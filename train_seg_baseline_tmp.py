import argparse
import logging
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb


PALETTE = {
    0: (255, 255, 255),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (255, 0, 0),
    255: (0, 0, 0),
}
INVERT_PALETTE = {v: k for k, v in PALETTE.items()}


@dataclass
class LoggingConfig:
    output_dir: str = "exps"
    exp_name: str = "vaihingen_thick_latent_seg_baseline_tmp_100k"
    logging_dir: str = "logs"
    project_name: str = "JiTCR"
    report_to: str = "wandb"
    sampling_steps: int = 1000
    checkpointing_steps: int = 10000


@dataclass
class DatasetConfig:
    data_dir: str = "/224010098/DATASET/DACR/Vaihingen/"
    latent_dir: str = "./fm_feature/Vaihingen/dinov3_vitl16_sat493m"
    latent_name: str = "clear_latent.npy"
    resolution: int = 256
    batch_size: int = 4
    val_size: int = 100
    num_classes: int = 6
    ignore_index: int = 255


@dataclass
class ModelConfig:
    latent_dim: int = 1024
    mid_channels: int = 256
    dropout_p: float = 0.1
    up_mode: str = "bilinear"
    use_syncbn: bool = False


@dataclass
class OptimizationConfig:
    max_train_steps: int = 100000
    learning_rate: float = 1.0e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0


@dataclass
class MiscConfig:
    seed: int = 0
    num_workers: int = 4
    allow_tf32: bool = True
    use_amp: bool = True


@dataclass
class Config:
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_logger(logging_dir: str):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def convert_from_color(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for color, label in INVERT_PALETTE.items():
        mask = np.all(arr_3d == np.array(color).reshape(1, 1, 3), axis=2)
        arr_2d[mask] = label
    return arr_2d


def convert_to_color(arr_2d):
    if isinstance(arr_2d, torch.Tensor):
        arr_2d = arr_2d.detach().cpu().numpy()
    arr_2d = arr_2d.astype(np.uint8)
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for label, color in PALETTE.items():
        arr_3d[arr_2d == label] = color
    return arr_3d


def scale_01(batch_image):
    return (batch_image + 1.0) / 2.0


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return x


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1, act=True, use_syncbn=False):
        super().__init__()
        bn = nn.SyncBatchNorm if use_syncbn else nn.BatchNorm2d
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn = bn(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, ch, use_syncbn=False, up_mode="bilinear"):
        super().__init__()
        self.up_mode = up_mode
        self.refine = ConvBNAct(ch, ch, k=3, p=1, use_syncbn=use_syncbn)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode=self.up_mode, align_corners=False)
        x = self.refine(x)
        return x


class ProgressiveUpDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        mid_channels=256,
        use_syncbn=False,
        up_mode="bilinear",
        dropout_p=0.1,
    ):
        super().__init__()
        self.stem = ConvBNAct(in_channels, mid_channels, k=1, p=0, use_syncbn=use_syncbn)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()
        self.up_blocks = nn.ModuleList(
            [UpBlock(mid_channels, use_syncbn=use_syncbn, up_mode=up_mode) for _ in range(4)]
        )
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.dropout(x)
        for block in self.up_blocks:
            x = block(x)
        return self.classifier(x)


class ThickLatentSegBaseline(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.latent_dim = cfg.model.latent_dim
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.decoder = ProgressiveUpDecoder(
            in_channels=self.latent_dim,
            num_classes=cfg.dataset.num_classes,
            mid_channels=cfg.model.mid_channels,
            use_syncbn=cfg.model.use_syncbn,
            up_mode=cfg.model.up_mode,
            dropout_p=cfg.model.dropout_p,
        )

    def _latent_to_nchw(self, latent):
        if latent.dim() == 3:
            bsz, num_tokens, channels = latent.shape
            if channels != self.latent_dim:
                raise ValueError(f"Expected latent dim {self.latent_dim}, got {channels}")
            side = int(math.sqrt(num_tokens))
            if side * side != num_tokens:
                raise ValueError(f"Latent token count {num_tokens} cannot form a square grid")
            latent = self.latent_norm(latent)
            latent = latent.view(bsz, side, side, channels)
            return latent.permute(0, 3, 1, 2).contiguous()

        if latent.dim() == 4:
            if latent.shape[1] == self.latent_dim:
                return latent.contiguous()
            if latent.shape[-1] == self.latent_dim:
                latent = self.latent_norm(latent)
                return latent.permute(0, 3, 1, 2).contiguous()

        raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")

    def forward(self, latent):
        x = self._latent_to_nchw(latent)
        return self.decoder(x)


class CrossEntropy2dIgnore(nn.Module):
    def __init__(self, ignore_label=255):
        super().__init__()
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1, device=predict.device, dtype=predict.dtype)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        return F.cross_entropy(predict, target, weight=weight)


class mIoUAvgMeter:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confmat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, pred, target):
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        for lt, lp in zip(target, pred):
            mask = lt != self.ignore_index
            lt = lt[mask].reshape(-1)
            lp = lp[mask].reshape(-1)
            if lt.size == 0:
                continue
            self.confmat += np.bincount(
                self.num_classes * lt + lp,
                minlength=self.num_classes ** 2,
            ).reshape(self.num_classes, self.num_classes)

    def value(self):
        tp = np.diag(self.confmat).astype(np.float64)
        pos_gt = self.confmat.sum(axis=1).astype(np.float64)
        pos_pr = self.confmat.sum(axis=0).astype(np.float64)
        union = pos_gt + pos_pr - tp
        iou = tp / np.clip(union, 1, None)
        return {"mIoU": float(np.nanmean(iou))}


class ThickLatentSegDataset(Dataset):
    def __init__(self, cfg: Config, mode="train"):
        self.cfg = cfg
        self.data_dir = cfg.dataset.data_dir
        self.latent_path = os.path.join(cfg.dataset.latent_dir, cfg.dataset.latent_name)
        self.mode = mode
        self.mask_dir = os.path.join(self.data_dir, "seg")
        self.clear_dir = os.path.join(self.data_dir, "clear")
        self.cloudy_dir = os.path.join(self.data_dir, "thick")
        self.latent_db = np.load(self.latent_path, allow_pickle=True).item()

        txt_name = "train.txt" if mode == "train" else "test.txt"
        img_list = np.loadtxt(os.path.join(self.data_dir, txt_name), dtype=str)
        if mode != "train" and cfg.dataset.val_size is not None:
            img_list = img_list[: cfg.dataset.val_size]
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        file_name = self.img_list[idx].replace("tif", "png")
        clear = io.imread(os.path.join(self.clear_dir, file_name)).astype(np.float32)[:, :, :3]
        cloudy = io.imread(os.path.join(self.cloudy_dir, file_name)).astype(np.float32)[:, :, :3]
        clear = ((clear / 255.0) * 2 - 1).transpose(2, 0, 1)
        cloudy = ((cloudy / 255.0) * 2 - 1).transpose(2, 0, 1)

        mask = io.imread(os.path.join(self.mask_dir, file_name))
        if mask.ndim == 3:
            mask = convert_from_color(mask)
        mask = mask.astype(np.int64)

        latent = self.latent_db[file_name].astype(np.float32)

        return {
            "clear": torch.from_numpy(clear),
            "cloudy": torch.from_numpy(cloudy),
            "mask": torch.from_numpy(mask),
            "clear_latent": torch.from_numpy(latent),
            "filename": file_name,
        }


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


@torch.no_grad()
def validate(model, dataloader, criterion, device, cfg: Config):
    model.eval()
    miou_meter = mIoUAvgMeter(cfg.dataset.num_classes, cfg.dataset.ignore_index)
    losses = []
    vis_batch = None

    for batch in tqdm(dataloader, desc="Val", leave=False):
        latent = batch["clear_latent"].to(device)
        mask = batch["mask"].to(device)
        logits = model(latent)
        loss = criterion(logits, mask)
        losses.append(loss.item())
        miou_meter.update(logits, mask)
        if vis_batch is None:
            vis_batch = {
                "clear": batch["clear"][:1].clone(),
                "cloudy": batch["cloudy"][:1].clone(),
                "mask": batch["mask"][:1].clone(),
                "pred": logits.argmax(dim=1)[:1].cpu(),
            }

    metrics = {"seg_loss": float(np.mean(losses)), **miou_meter.value()}
    return metrics, vis_batch


def main(args):
    cfg = Config()
    if args.exp_name:
        cfg.logging.exp_name = args.exp_name
    if args.batch_size:
        cfg.dataset.batch_size = args.batch_size
    if args.max_train_steps:
        cfg.optimization.max_train_steps = args.max_train_steps
    if args.learning_rate:
        cfg.optimization.learning_rate = args.learning_rate

    set_seed(cfg.misc.seed)

    if cfg.misc.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(cfg.logging.output_dir, cfg.logging.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    file_logger = create_logger(save_dir)
    shutil.copyfile(__file__, os.path.join(save_dir, Path(__file__).name))

    train_dataset = ThickLatentSegDataset(cfg, mode="train")
    val_dataset = ThickLatentSegDataset(cfg, mode="test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.misc.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.misc.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    train_iter = cycle(train_loader)

    model = ThickLatentSegBaseline(cfg).to(device)
    criterion = CrossEntropy2dIgnore(ignore_label=cfg.dataset.ignore_index)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimization.learning_rate,
        betas=(cfg.optimization.adam_beta1, cfg.optimization.adam_beta2),
        weight_decay=cfg.optimization.adam_weight_decay,
        eps=cfg.optimization.adam_epsilon,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.misc.use_amp and device.type == "cuda")

    if cfg.logging.report_to == "wandb":
        wandb.login()
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.exp_name,
            config=asdict(cfg),
        )
        wandb.watch(model, log=None)

    file_logger.info(f"Experiment directory created at {save_dir}")
    file_logger.info(f"Train set: {len(train_dataset)} | Val set: {len(val_dataset)}")
    file_logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    progress_bar = tqdm(range(1, cfg.optimization.max_train_steps + 1), desc="Steps")
    model.train()

    for global_step in progress_bar:
        batch = next(train_iter)
        latent = batch["clear_latent"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.misc.use_amp and device.type == "cuda"):
            logits = model(latent)
            seg_loss = criterion(logits, mask)
            loss = seg_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimization.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        logs = {
            "loss": float(loss.detach().item()),
            "denoising_loss": 0.0,
            "seg_loss": float(seg_loss.detach().item()),
            "grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
        }
        progress_bar.set_postfix(**logs)
        if cfg.logging.report_to == "wandb":
            wandb.log(logs, step=global_step)

        if global_step % cfg.logging.checkpointing_steps == 0 or global_step == 1:
            ckpt_path = os.path.join(save_dir, "checkpoints", f"{global_step:07d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "step": global_step,
                    "config": asdict(cfg),
                },
                ckpt_path,
            )
            file_logger.info(f"Saved checkpoint to {ckpt_path}")

        if global_step == 1 or global_step % cfg.logging.sampling_steps == 0:
            metrics, vis_batch = validate(model, val_loader, criterion, device, cfg)
            file_logger.info(f"Validation metrics at step {global_step}: {metrics}")

            if cfg.logging.report_to == "wandb":
                log_images = {
                    "input": wandb.Image(array2grid(scale_01(vis_batch["clear"]))),
                    "cloudy": wandb.Image(array2grid(scale_01(vis_batch["cloudy"]))),
                    # This baseline has no restoration output; reuse clear-reference slot to keep panel keys aligned.
                    "samples": wandb.Image(array2grid(scale_01(vis_batch["clear"]))),
                    "mask_label": wandb.Image(convert_to_color(vis_batch["mask"][0])),
                    "mask_pred": wandb.Image(convert_to_color(vis_batch["pred"][0])),
                }
                wandb.log({**metrics, **log_images}, step=global_step)
            model.train()

    if cfg.logging.report_to == "wandb":
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Temporary thick-latent segmentation baseline")
    parser.add_argument("--exp-name", type=str, default='seg_thick')
    parser.add_argument("--batch-size", type=int, default='4')
    parser.add_argument("--max-train-steps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
