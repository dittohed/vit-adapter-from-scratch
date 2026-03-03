import argparse
import os

from pathlib import Path
from PIL import Image
import torch

from torchvision.utils import save_image

from vit_adapter.datasets.ade20k import ADE20K
from vit_adapter.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD, SegmentationTransform


def parse_args():
    p = argparse.ArgumentParser("Visualize ADE20K samples (original vs transformed)")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--scale-min", type=float, default=0.5)
    p.add_argument("--scale-max", type=float, default=2.0)
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument("--no-reduce-zero-label", dest="reduce_zero_label", action="store_false")
    p.add_argument("--out-dir", type=Path, default="./local/work_dir/viz")
    return p.parse_args()


def denormalize(img_chw: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=img_chw.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img_chw.device).view(3, 1, 1)
    return (img_chw * std + mean).clamp(0, 1)


def make_palette(size: int, black_index: int) -> torch.Tensor:
    """
    Create a color palette for visualizing segmentation masks. 

    The palette is deterministic and assigns a unique color to each class index, 
    except for the specified `black_index` which is set to black (0,0,0).
    """

    idx = torch.arange(size, dtype=torch.int64) + 1  # Offset so class 0 isn't black by default
    pal = torch.stack([(idx * 37) % 255, (idx * 17) % 255, (idx * 29) % 255], dim=1).float() / 255.0
    pal[black_index] = 0.0

    return pal


def colorize_mask(mask_hw: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """
    Convert a segmentation mask ([H,W] int64) with class indices to 
    an RGB image ([3,H,W] float) using the provided palette.
    """

    mask_hw = mask_hw.clamp(0, palette.shape[0] - 1)
    h, w = mask_hw.shape
    return palette[mask_hw.view(-1)].view(h, w, 3).permute(2, 0, 1).contiguous()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    tf = SegmentationTransform(
        mode="train" if args.split == "train" else "val",
        crop_size=args.crop_size,
        scale_range=(args.scale_min, args.scale_max),
        ignore_index=args.ignore_index,
    )
    ds = ADE20K(
        root=args.data_root,
        split=args.split,
        transform=tf,
        reduce_zero_label=args.reduce_zero_label,
    )

    pal_size = max(151, int(args.ignore_index) + 1)
    palette = make_palette(pal_size, black_index=args.ignore_index)

    idxs = torch.randperm(len(ds))[:args.num_samples].tolist()

    for j, idx in enumerate(idxs):
        # Save original image for reference
        img_path, _ = ds.samples[int(idx)]
        Image.open(img_path).convert("RGB").save(args.out_dir / f"{idx:06d}_0_orig.png")

        # Save transformed image and mask
        img_t, mask_t = ds[int(idx)]
        img_vis = denormalize(img_t)
        mask_rgb = colorize_mask(mask_t, palette)
        save_image(img_vis, args.out_dir / f"{idx:06d}_1_img.png")
        save_image(mask_rgb, args.out_dir / f"{idx:06d}_2_mask.png")

    print(f"Saved {args.num_samples} samples to {args.out_dir}")


if __name__ == "__main__":
    main()

