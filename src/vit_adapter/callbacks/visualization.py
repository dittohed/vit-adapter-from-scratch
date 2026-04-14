import torch
import wandb
from lightning.pytorch.callbacks import Callback

from vit_adapter.utils.visualization import denormalize, make_palette, colorize_mask


class VisualizationCallback(Callback):
    """Logs input / GT / prediction images to WandB every ``every_n_steps`` global steps."""

    def __init__(self, every_n_steps: int, num_classes: int = 150,
                 ignore_index: int = 255, max_samples: int = 4):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.max_samples = max_samples
        pal_size = max(num_classes + 1, ignore_index + 1)
        self.palette = make_palette(pal_size, black_index=ignore_index)
        self._fixed_batch = None

    def _get_fixed_batch(self, trainer, pl_module):
        if self._fixed_batch is None:
            ds = trainer.datamodule.train_dataset
            n = min(self.max_samples, len(ds))
            
            gen = torch.Generator().manual_seed(0)
            idxs = torch.randperm(len(ds), generator=gen)[:n].tolist()
            imgs, masks = zip(*(ds[i] for i in idxs))

            device = pl_module.device
            self._fixed_batch = (torch.stack(imgs).to(device), torch.stack(masks).to(device))

        return self._fixed_batch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps <= 0:
            return
        if trainer.global_step % self.every_n_steps != 0:
            return

        imgs, masks = self._get_fixed_batch(trainer, pl_module)
        n = min(self.max_samples, imgs.shape[0])

        with torch.no_grad():
            logits = pl_module(imgs[:n])
            preds = logits.argmax(dim=1)

        step = trainer.global_step
        log_images = {}
        for i in range(n):
            img_rgb = denormalize(imgs[i]).cpu()
            gt_rgb = colorize_mask(masks[i].cpu(), self.palette)
            pred_rgb = colorize_mask(preds[i].cpu(), self.palette)

            strip = torch.cat([img_rgb, gt_rgb, pred_rgb], dim=2)
            strip_np = strip.permute(1, 2, 0).numpy()
            log_images[f"train_vis/sample_{i}_step_{step}"] = wandb.Image(
                strip_np, caption=f"step {step} | sample {i} | input | GT | pred"
            )

        trainer.logger.experiment.log({**log_images, "global_step": step})
