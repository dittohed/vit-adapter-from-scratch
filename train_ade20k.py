import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vit_adapter.datasets.ade20k import ADE20K
from vit_adapter.datasets.transforms import SegmentationTransform
from vit_adapter.models.segmentation_model import SegmentationModel
from vit_adapter.models.vit_adapter import ViTAdapterBackbone
from vit_adapter.models.upernet import UperNetHead
from vit_adapter.models.semantic_fpn import SemanticFPNHead
from vit_adapter.utils.metrics import compute_miou, compute_confusion_matrix

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser("ViT-Adapter ADE20K training (PyTorch Lightning)")
    parser.add_argument("--data-root", type=str, default="local/datasets/ADEChallengeData2016", help="ADE20K root with images/ and annotations/")
    parser.add_argument("--work-dir", type=str, default="./local/work_dir")
    parser.add_argument("--wandb-project", type=str, default="vit-adapter-ade20k")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-offline", action="store_true", default=False)

    parser.add_argument("--vit-name", type=str, default="deit_small_patch16_224")
    parser.add_argument("--no-vit-pretrained", dest="vit_pretrained", action="store_false")
    parser.add_argument("--spm-base-channels", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=150)
    parser.add_argument("--decode-head", type=str, default="semantic_fpn", choices=["upernet", "semantic_fpn"])
    parser.add_argument("--decode-channels", type=int, default=256)
    parser.add_argument("--semantic-fpn-dropout", type=float, default=0.0)

    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--no-reduce-zero-label", dest="reduce_zero_label", action="store_false")
    parser.add_argument("--ignore-index", type=int, default=255)

    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-iters", type=int, default=1500)
    parser.add_argument("--power", type=float, default=0.9)

    parser.add_argument("--max-iters", type=int, default=80000, help="Max optimizer steps")
    parser.add_argument("--epochs", type=int, default=160, help="Trainer max epochs (max-iters usually stops first)")
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true", default=False)

    parser.add_argument("--eval-interval", type=int, default=8000, help="Validate every N train batches")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--resume", type=str, default="", help="Path to a Lightning .ckpt to resume from")

    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Device backend (use 'cpu' for quick debugging even if GPUs are available)",
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use (usually 1 for debugging)")
    return parser.parse_args()


def build_param_groups(model: SegmentationModel, base_lr: float, weight_decay: float):
    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            decay = 0.0
        else:
            decay = weight_decay

        if decay not in groups:
            groups[decay] = {
                "params": [],
                "weight_decay": decay,
            }
        
        groups[decay]["params"].append(param)

    return [
        {
            "params": g["params"],
            "weight_decay": g["weight_decay"],
            "lr": base_lr,
        }
        for g in groups.values()
    ]


def poly_warmup_factor(step: int, total_steps: int, warmup_steps: int, power: float) -> float:
    step = step + 1  # So the first optimizer step has a small but non-zero LR

    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)

    return (1.0 - progress) ** power


@dataclass
class DataConfig:
    data_root: str
    crop_size: int
    scale_min: float
    scale_max: float
    reduce_zero_label: bool
    ignore_index: int
    batch_size: int
    num_workers: int
    pin_memory: bool


class ADE20KDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        train_tf = SegmentationTransform(
            mode="train",
            crop_size=self.cfg.crop_size,
            scale_range=(self.cfg.scale_min, self.cfg.scale_max),
            ignore_index=self.cfg.ignore_index,
        )
        val_tf = SegmentationTransform(
            mode="val",
            crop_size=self.cfg.crop_size,
            ignore_index=self.cfg.ignore_index,
        )
        self.train_dataset = ADE20K(
            self.cfg.data_root,
            split="train",
            transform=train_tf,
            reduce_zero_label=self.cfg.reduce_zero_label,
        )
        self.val_dataset = ADE20K(
            self.cfg.data_root,
            split="val",
            transform=val_tf,
            reduce_zero_label=self.cfg.reduce_zero_label,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )


class SegLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: SegmentationModel,
        num_classes: int,
        ignore_index: int,
        lr: float,
        weight_decay: float,
        warmup_iters: int,
        max_iters: int,
        power: float,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.power = power

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        # Register mIoU confusion matrix as a buffer so Lightning moves it
        # with the module across devices
        self.register_buffer(
            "val_cm",
            torch.zeros((num_classes, num_classes), dtype=torch.int64),
            persistent=False,
        )

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.criterion(logits, masks)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.criterion(logits, masks)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        preds = logits.argmax(dim=1)
        self.val_cm += compute_confusion_matrix(preds, masks, self.num_classes, self.ignore_index)

    def on_validation_epoch_end(self):
        # Aggregate the confusion matrices across ranks before computing mIoU
        cm = self.val_cm
        if getattr(self.trainer, "world_size", 1) > 1:
            cm = self.all_gather(cm).sum(dim=0)

        miou, _ = compute_miou(cm)
        self.log("val_miou", miou, prog_bar=True, sync_dist=False)
        self.val_cm.zero_()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            build_param_groups(self.model, self.lr, self.weight_decay),
            lr=self.lr,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: poly_warmup_factor(
                step, total_steps=self.max_iters, warmup_steps=self.warmup_iters, power=self.power
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)

    accelerator = args.accelerator
    pin_memory = accelerator in {"gpu", "auto"} and torch.cuda.is_available()

    data_cfg = DataConfig(
        data_root=args.data_root,
        crop_size=args.crop_size,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        reduce_zero_label=args.reduce_zero_label,
        ignore_index=args.ignore_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    dm = ADE20KDataModule(data_cfg)

    backbone = ViTAdapterBackbone(
        vit_name=args.vit_name, pretrained=args.vit_pretrained, base_channels=args.spm_base_channels
    )

    if args.decode_head == "upernet":
        decode_head = UperNetHead(
            in_channels=backbone.out_channels, channels=args.decode_channels, num_classes=args.num_classes
        )
    else:
        decode_head = SemanticFPNHead(
            in_channels=backbone.out_channels,
            fpn_out_channels=args.decode_channels,
            semantic_out_channels=args.decode_channels // 2,
            num_classes=args.num_classes,
            dropout=args.semantic_fpn_dropout,
        )
    
    seg_model = SegmentationModel(backbone, decode_head)
    lit_model = SegLightningModule(
        model=seg_model,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_iters=args.warmup_iters,
        max_iters=args.max_iters,
        power=args.power,
    )

    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name if args.wandb_name else None,
        save_dir=args.work_dir,
        offline=args.wandb_offline,
        log_model=False,
    )
    ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}-{step}-{val_miou:.4f}",
        monitor="val_miou",
        mode="max",
        save_last=True,
        save_top_k=1,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    if accelerator == "cpu" and args.amp:
        print("NOTE: --amp is not supported on CPU; falling back to full precision (32-bit).")
    precision = "16-mixed" if (args.amp and accelerator != "cpu") else 32

    trainer = pl.Trainer(
        default_root_dir=args.work_dir,
        accelerator=accelerator,
        devices=args.devices,
        strategy="ddp" if (accelerator == "gpu" and args.devices > 1) else "auto",
        sync_batchnorm=True if (accelerator == "gpu" and args.devices > 1) else False,
        max_steps=args.max_iters,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accum_steps,
        gradient_clip_val=args.grad_clip,
        precision=precision,
        log_every_n_steps=args.log_interval,
        val_check_interval=args.eval_interval,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        enable_checkpointing=True,
    )

    ckpt_path = args.resume if args.resume else None
    trainer.fit(lit_model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
