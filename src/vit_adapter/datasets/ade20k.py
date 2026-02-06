import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from vit_adapter.datasets.transforms import SegmentationTransform


class ADE20K(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: SegmentationTransform | None = None,
        reduce_zero_label: bool = True,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.reduce_zero_label = reduce_zero_label

        img_dir, ann_dir = self._resolve_dirs(root, split)
        self.samples = self._load_samples(img_dir, ann_dir)
        if not self.samples:
            raise RuntimeError(f"No ADE20K samples found in {img_dir} and {ann_dir}")

    def _resolve_dirs(self, root: str, split: str) -> Tuple[str, str]:
        if split not in {"train", "val", "validation"}:
            raise ValueError("split must be 'train' or 'val'")
        img_split = "training" if split == "train" else "validation"
        ann_split = "training" if split == "train" else "validation"
        img_dir = os.path.join(root, "images", img_split)
        ann_dir = os.path.join(root, "annotations", ann_split)
        return img_dir, ann_dir

    def _load_samples(self, img_dir: str, ann_dir: str) -> List[Tuple[str, str]]:
        samples: List[Tuple[str, str]] = []
        for name in sorted(os.listdir(ann_dir)):
            if not name.endswith(".png"):
                continue
            base = name[:-4]
            ann_path = os.path.join(ann_dir, name)
            img_path = os.path.join(img_dir, base + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, base + ".png")
            if not os.path.exists(img_path):
                continue
            samples.append((img_path, ann_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path)
        mask = self._encode_mask(mask)
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask

    def _encode_mask(self, mask: Image.Image) -> Image.Image:
        if not self.reduce_zero_label:
            return mask
        mask_np = np.array(mask, dtype=np.int64)
        mask_np[mask_np == 0] = 255
        mask_np[mask_np != 255] -= 1
        return Image.fromarray(mask_np.astype(np.uint8))
