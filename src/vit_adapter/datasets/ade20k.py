from typing import List, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from vit_adapter.datasets.transforms import SegmentationTransform


class ADE20K(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: SegmentationTransform | None = None,
        # If True, the label 0 (background) will be set to 255, while all other labels will 
        # be reduced by 1 which plays well with CrossEntropyLoss(ignore_index=255) 
        # and having only 150 output classes (instead of 151 with background)
        reduce_zero_label: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.reduce_zero_label = reduce_zero_label

        img_dir, ann_dir = self._resolve_dirs(self.root, split)
        self.samples = self._load_samples(img_dir, ann_dir)
        if not self.samples:
            raise RuntimeError(f"No ADE20K samples found in {img_dir} and {ann_dir}")

    def _resolve_dirs(self, root: Path, split: str) -> Tuple[Path, Path]:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        split_full_name = "training" if split == "train" else "validation"

        img_dir = root / "images" / split_full_name
        ann_dir = root / "annotations" / split_full_name

        return img_dir, ann_dir

    def _load_samples(self, img_dir: Path, ann_dir: Path) -> List[Tuple[Path, Path]]:
        samples: List[Tuple[Path, Path]] = []

        for ann_path in sorted(ann_dir.glob("*.png")):
            img_path = img_dir / (ann_path.stem + ".jpg")
            samples.append((img_path, ann_path))

        return samples

    def _preprocess_mask(self, mask: Image.Image) -> Image.Image:
        if not self.reduce_zero_label:
            return mask
        
        mask_np = np.array(mask, dtype=np.int64)
        mask_np[mask_np == 0] = 255
        mask_np[mask_np != 255] -= 1

        return Image.fromarray(mask_np.astype(np.uint8))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path)
        mask = self._preprocess_mask(mask)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask