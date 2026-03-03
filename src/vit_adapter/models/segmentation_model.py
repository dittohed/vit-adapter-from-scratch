import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from vit_adapter.utils.metrics import compute_miou, compute_confusion_matrix
from vit_adapter.utils.utils import build_param_groups, poly_warmup_factor


class SegmentationModel(nn.Module):
    def __init__(self, backbone: nn.Module, decode_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decode_head(feats)

        # Both UperNet and SemanticFPN produce 1/4 scale logits, so still need to upsample
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return logits


class SegLightningModule(pl.LightningModule):
    # Declared for static type checkers; the value is created via `register_buffer`
    val_cm: torch.Tensor

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
