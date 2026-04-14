import torch

from vit_adapter.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img_chw: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization. Input/output: (3,H,W) float32."""
    mean = torch.tensor(IMAGENET_MEAN, device=img_chw.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img_chw.device).view(3, 1, 1)
    return (img_chw * std + mean).clamp(0, 1)


def make_palette(size: int, black_index: int) -> torch.Tensor:
    """Deterministic color palette. Returns (size, 3) float tensor."""
    idx = torch.arange(size, dtype=torch.int64) + 1
    pal = torch.stack([(idx * 37) % 255, (idx * 17) % 255, (idx * 29) % 255], dim=1).float() / 255.0
    pal[black_index] = 0.0
    return pal


def colorize_mask(mask_hw: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """(H,W) int64 mask -> (3,H,W) float RGB using palette."""
    mask_hw = mask_hw.clamp(0, palette.shape[0] - 1)
    h, w = mask_hw.shape
    return palette[mask_hw.view(-1)].view(h, w, 3).permute(2, 0, 1).contiguous()
