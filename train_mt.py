import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import yaml
from types import SimpleNamespace
import shutil

import torch
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from engine_mt import JiTEngine
from models.adahdit_mt import image_transformer

from dataset import CUHKCRDataset, ISPRSDataset
import wandb
import math
from torchvision.utils import make_grid

torch.set_float32_matmul_precision('high')
os.environ.setdefault("USE_COMPILE", "0")
logger = get_logger(__name__)
USE_WANDB = True
if USE_WANDB:
    wandb.login(key="5762924b097784497ea7f44d7faff92bcb46d691")

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    file_logger = logging.getLogger(__name__)
    return file_logger


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_config(args=None):
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        if isinstance(d, list):
            return [dict_to_namespace(x) for x in d]
        return d

    config = dict_to_namespace(cfg_dict)
    if args.exp_name:
        config.logging.exp_name = args.exp_name
    elif not (hasattr(config.logging, "exp_name") and config.logging.exp_name):
        from datetime import datetime
        config.logging.exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return config


def main(args):
    config = load_config(args)
    use_wandb = USE_WANDB

    logging_dir = Path(config.logging.output_dir, config.logging.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.logging.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        mixed_precision=config.precision.mixed_precision,
        log_with=config.logging.report_to if use_wandb else None,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(config.logging.output_dir, exist_ok=True)
        save_dir = os.path.join(config.logging.output_dir, config.logging.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(save_dir, "config.yml"))
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        file_logger = create_logger(save_dir)
        file_logger.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if config.misc.seed is not None:
        set_seed(config.misc.seed + accelerator.process_index)

    def namespace_to_dict(obj):
        if isinstance(obj, dict):
            return {k: namespace_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, SimpleNamespace):
            return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(namespace_to_dict(v) for v in obj)
        return obj

    kwargs = namespace_to_dict(config.model)
    model = image_transformer.ImageTransformerDenoiserModelInterface(**kwargs).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    if accelerator.is_main_process:
        file_logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if config.precision.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.optimization.learning_rate),
        betas=(float(config.optimization.adam_beta1), float(config.optimization.adam_beta2)),
        weight_decay=float(config.optimization.adam_weight_decay),
        eps=float(config.optimization.adam_epsilon),
    )

    if 'changsha' in config.dataset.data_dir or 'guangzhou' in config.dataset.data_dir:
        train_dataset = CUHKCRDataset(config.dataset.data_dir, mode='train')
        val_dataset = CUHKCRDataset(config.dataset.data_dir, mode='test', val_size=config.dataset.val_size)
    elif 'Potsdam' in config.dataset.data_dir or 'Vaihingen' in config.dataset.data_dir:
        train_dataset = ISPRSDataset(config.dataset, mode='train')
        val_dataset = ISPRSDataset(config.dataset, mode='test', val_size=config.dataset.val_size)

    local_batch_size = int(config.dataset.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=config.misc.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.misc.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if accelerator.is_main_process:
        file_logger.info(f"Dataset contains {len(train_dataset):,} images ({config.dataset.data_dir})")

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    global_step = 0
    if config.logging.resume_step > 0:
        ckpt_name = str(config.logging.resume_step).zfill(7) + '.pt'
        ckpt = torch.load(
            f'{os.path.join(config.logging.output_dir, config.logging.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
        )

        def _strip_prefix(sd, prefixes=("module.", "model.")):
            if sd is None:
                return sd
            new_sd = {}
            for k, v in sd.items():
                nk = k
                for p in prefixes:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                new_sd[nk] = v
            return new_sd

        model_sd = _strip_prefix(ckpt.get('model'))
        ema_sd = _strip_prefix(ckpt.get('ema'))
        model.load_state_dict(model_sd)
        ema.load_state_dict(ema_sd)
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if accelerator.is_main_process and use_wandb:
        tracker_config = vars(copy.deepcopy(config))
        accelerator.init_trackers(
            project_name="JiTCR",
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{config.logging.exp_name}"}
            },
        )

    progress_bar = tqdm(
        range(0, config.optimization.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    engine = JiTEngine(
        model=model,
        config=config,
    )

    for epoch in range(config.optimization.epochs):
        engine.model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(engine.model):
                loss = engine(batch)
                loss_mean = loss.mean()

                accelerator.backward(loss_mean)
                if accelerator.sync_gradients:
                    params_to_clip = engine.model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, config.optimization.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, engine.model)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            if global_step % config.optimization.checkpointing_steps == 0 or global_step == 1:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": engine.model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "config": config,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    file_logger.info(f"Saved checkpoint to {checkpoint_path}")

            if global_step == 1 or (global_step % config.logging.sampling_steps == 0 and global_step > 0):
                engine.model.eval()
                engine.avg_metrics.reset()
                for batch_val in tqdm(val_dataloader, desc='Val'):
                    engine.test_step(batch_val)
                metrics = engine.avg_metrics.value()
                if use_wandb:
                    accelerator.log(metrics, step=global_step)
                    batch_log = {k: batch[k][:1] for k in batch}
                    image_results = engine.log_images(batch_log, sample=True)
                    accelerator.log({
                        "input": wandb.Image(array2grid(image_results['input'])),
                        "cloudy": wandb.Image(array2grid(image_results['cloudy'])),
                        "samples": wandb.Image(array2grid(image_results['samples'])),
                    }, step=global_step)
                elif accelerator.is_main_process:
                    file_logger.info(f"Validation metrics at step {global_step}: {metrics}")
                engine.model.train()

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
            }
            progress_bar.set_postfix(**logs)
            if use_wandb:
                accelerator.log(logs, step=global_step)

            if global_step >= config.optimization.max_train_steps:
                break
        if global_step >= config.optimization.max_train_steps:
            break

    model.eval()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        file_logger.info("Done!")
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/vaihingen_thick_adahdit_dinojoint_B2_100k.yml")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
