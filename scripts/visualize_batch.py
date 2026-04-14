import argparse
import os

from pathlib import Path
from PIL import Image
import torch

from torchvision.utils import save_image

from vit_adapter.datasets.ade20k import ADE20K
from vit_adapter.datasets.transforms import SegmentationTransform
from vit_adapter.utils.visualization import denormalize, make_palette, colorize_mask


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

