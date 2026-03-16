import argparse
import os
from pathlib import Path
import json
import yaml
from types import SimpleNamespace

from PIL import Image
import numpy as np
import math
from torchvision.utils import make_grid, save_image
import torch
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.set_float32_matmul_precision('high')

from model import JiT_models
from engine import JiTEngine
from dataset import CUHKCRDataset, ISPRSDataset

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x

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
    return config

def prepare_batch(batch, device):
    """
    遍历 batch（dict/list/tuple），
    将 numpy.ndarray 转换为 torch.Tensor 并移动到 device 上。
    """
    if isinstance(batch, dict):
        return {k: prepare_batch(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(prepare_batch(v, device) for v in batch)
    elif isinstance(batch, np.ndarray):
        return torch.from_numpy(batch).to(device)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch  # 其他类型（如 int/str）保持不变

#################################################################################
#                                  Testing Loop                                #
#################################################################################

def main(args):
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available()
    device = "cuda"

    config = load_config(args)

    if config.misc.seed is not None:
        torch.manual_seed(config.misc.seed)

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
    # model = image_transformer.ImageTransformerDenoiserModelInterface(**kwargs).to(device)
    model = JiT_models[config.model.model_name](
            input_size=config.dataset.resolution,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            attn_drop=config.model.attn_dropout,
            proj_drop=config.model.proj_dropout,
        ).to(device)

    # Setup data:
    if 'changsha' in config.dataset.data_dir or 'guangzhou' in config.dataset.data_dir:
        test_dataset = CUHKCRDataset(config.dataset.data_dir, mode='test')

    # ISPRS data
    elif 'Potsdam' in config.dataset.data_dir or 'Vaihingen' in config.dataset.data_dir:
        test_dataset = ISPRSDataset(config.dataset, mode='test')

    # Setup data:
    # test_dataset = CRDataset(config.dataset.data_dir, mode='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.misc.num_workers,
        pin_memory=True,
    )
    print(f"Dataset contains {len(test_dataloader):,} images ({config.dataset.data_dir})")

    # load checkpoint:
    if args.ckpt_step:
        config_dir = Path(args.config).parent
        ckpt_step = args.ckpt_step
        ckpt_path = config_dir / f'checkpoints/{int(ckpt_step):07d}.pt'
    else:
        config_dir = Path(args.config).parent
        ckpt_dir = config_dir / "checkpoints"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"No checkpoints directory found: {ckpt_dir}")
        ckpts = sorted(ckpt_dir.glob("*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
        ckpt_path = str(ckpts[-1])
        ckpt_step = os.path.splitext(os.path.basename(str(ckpts[-1])))[0]
    print(f'Loading checkpoint file from: {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)['ema']
    model.load_state_dict(state_dict)

    save_root = Path(config.logging.output_dir) / config.logging.exp_name
    test_dir = save_root / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = test_dir / "preds"
    if args.pred:
        pred_dir.mkdir(exist_ok=True)
    print(f"The test output will be saved at: {test_dir}")

    # engine = CRDiffusionEngineSiT(
    #     model=model,
    #     prediction=config.loss.prediction,
    #     path_type=config.loss.path_type,
    #     weighting=config.loss.weighting
    # )
    engine = JiTEngine(
        model=model,
        config=config,
    )


    engine.model.eval()
    engine.avg_metrics.reset()

    for i, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        if i>= 10:
            break
        batch = prepare_batch(batch, device)
        engine(batch)
        engine.test_step(batch)
        if args.pred:
            results = engine.log_images(batch)
            sample = torch.clamp(255. * results.get("samples"), 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            save_path = f"{pred_dir}/{Path(os.path.basename(batch['filename'][0])).with_suffix('.png').name}"
            Image.fromarray(sample[0]).save(save_path)

    metrics = engine.avg_metrics.value()
    metrics = {k: (v.item() if torch.is_tensor(v) else float(v))
               for k, v in metrics.items()}
    with open(test_dir / f"{ckpt_step}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test metrics saved at: {metrics}")
    # print(metrics)


def parse_args():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("--config", type=str, default="exps/vaihingen_thick_jit_B16_fm_200k/config.yml")
    parser.add_argument("--ckpt-step", type=str, default=None)
    parser.add_argument("--pred", type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
