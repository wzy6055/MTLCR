import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import yaml
from types import SimpleNamespace
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.set_float32_matmul_precision('high')

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# from models.sit import SiT_models
# from models.sitcr import SiT_models
# from models.k_diffusion import image_transformer
# from loss import SILoss, SICRLoss
from engine import JiTEngine
# from model import JiT_models
# from models.hdit import image_transformer
# from utils import load_encoders
from models.hjit import HJiT_models

from dataset import CUHKCRDataset, ISPRSDataset
# from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)
wandb.login(key="5762924b097784497ea7f44d7faff92bcb46d691")

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def load_config(args=None):
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(x) for x in d]
        else:
            return d

    config = dict_to_namespace(cfg_dict)
    if args.exp_name:
        config.logging.exp_name = args.exp_name
    elif hasattr(config.logging, "exp_name") and config.logging.exp_name:
        pass  # yml 中已设置
    else:
        from datetime import datetime
        config.logging.exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return config


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    config = load_config(args)

    # set accelerator
    logging_dir = Path(config.logging.output_dir, config.logging.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.logging.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        mixed_precision=config.precision.mixed_precision,
        log_with=config.logging.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(config.logging.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(config.logging.output_dir, config.logging.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(save_dir, "config.yml"))
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if config.misc.seed is not None:
        set_seed(config.misc.seed + accelerator.process_index)

    # def namespace_to_dict(obj):
    #     if isinstance(obj, dict):
    #         return {k: namespace_to_dict(v) for k, v in obj.items()}
    #     from types import SimpleNamespace
    #     if isinstance(obj, SimpleNamespace):
    #         return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    #     if isinstance(obj, (list, tuple)):
    #         return type(obj)(namespace_to_dict(v) for v in obj)
    #     return obj
    # kwargs = namespace_to_dict(config.model)
    # model = image_transformer.ImageTransformerDenoiserModelInterface(**kwargs)
    # model = model.to(device)
    # model = DenoiserCR(**kwargs)
    # model = DenoiserCR(config)
    model = HJiT_models[config.model.model_name](
            input_size=config.dataset.resolution,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            attn_drop=config.model.attn_dropout,
            proj_drop=config.model.proj_dropout,
        )
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    if accelerator.is_main_process:
        logger.info(f"JiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
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

    # Setup data:
    if 'changsha' in config.dataset.data_dir or 'guangzhou' in config.dataset.data_dir:
        train_dataset = CUHKCRDataset(config.dataset.data_dir, mode='train')
        val_dataset = CUHKCRDataset(config.dataset.data_dir, mode='test', val_size=config.dataset.val_size)

    # ISPRS data
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
        logger.info(f"Dataset contains {len(train_dataset):,} images ({config.dataset.data_dir})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume:
    global_step = 0
    if config.logging.resume_step > 0:
        ckpt_name = str(config.logging.resume_step).zfill(7) + '.pt'
        ckpt = torch.load(
            f'{os.path.join(config.logging.output_dir, config.logging.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
        )

        # temp for resume
        def _strip_prefix(sd, prefixes=("module.", "model.")):
            """去掉 DataParallel/DDP 等保存时加的前缀"""
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
        model.load_state_dict(model_sd)  # 如仍有结构差异，可加 strict=False
        ema.load_state_dict(ema_sd)
        # temp for resume end

        # model.load_state_dict(ckpt['model'])
        # ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if accelerator.is_main_process:
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
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    engine = JiTEngine(
        model=model,
        config=config,
        # prediction=config.loss.prediction,
        # path_type=config.loss.path_type,
        # accelerator=accelerator,
        # weighting=config.loss.weighting
    )


    for epoch in range(config.optimization.epochs):
        engine.model.train()
        for batch in train_dataloader:

            with accelerator.accumulate(engine.model):

                loss = engine(batch)
                loss_mean = loss.mean()

                ## optimization
                accelerator.backward(loss_mean)
                if accelerator.sync_gradients:
                    params_to_clip = engine.model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, config.optimization.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, engine.model)  # change ema function

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if global_step % config.optimization.checkpointing_steps == 0 or global_step == 1:
                # if global_step % config.optimization.checkpointing_steps == 1:
                if accelerator.is_main_process:
                    checkpoint = {
                        # "model": engine.model.module.state_dict(),
                        "model": engine.model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "config": config,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            # val step
            if (global_step == 1 or (global_step % config.logging.sampling_steps == 0 and global_step > 0)):
                engine.model.eval()
                engine.avg_metrics.reset()
                for j, batch_val in enumerate(tqdm(val_dataloader, desc='Val')):
                    engine.test_step(batch_val)
                metrics = engine.avg_metrics.value()
                accelerator.log(metrics, step=global_step)
                batch_log = {k: batch[k][:1] for k in batch}
                image_results = engine.log_images(batch_log, sample=True)
                accelerator.log({"input": wandb.Image(array2grid(image_results['input'])),
                                 "cloudy": wandb.Image(array2grid(image_results['cloudy'])),
                                 "samples": wandb.Image(array2grid(image_results['samples'])),
                                 },
                                step=global_step)
                engine.model.train()

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(),
                # "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= config.optimization.max_train_steps:
                break
        if global_step >= config.optimization.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/vaihingen_thick_hjit_B2_fm_200k.yml")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
