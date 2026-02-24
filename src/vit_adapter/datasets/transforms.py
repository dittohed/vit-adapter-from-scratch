from __future__ import annotations

import numpy as np
from PIL import Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationTransform:
    def __init__(
        self,
        mode: str,
        crop_size: int = 512,
        scale_range: tuple[float, float] = (0.5, 2.0),
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        ignore_index: int = 255,
    ):
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")

        self.mode = mode
        self.crop_size = (crop_size, crop_size)
        self.scale_range = scale_range
        self.ignore_index = ignore_index

        self.crop = T.RandomCrop(
            size=self.crop_size,
            pad_if_needed=True,
            fill={tv_tensors.Image: 0, tv_tensors.Mask: int(ignore_index)},
        )
        self.flip = T.RandomHorizontalFlip(p=0.5)
        self.to_dtype = T.ToDtype(torch.float32, scale=True)
        self.normalize = T.Normalize(mean=mean, std=std)
        self.to_pure = T.ToPureTensor()

    def __call__(self, img: Image.Image, mask: Image.Image):
        img_t = F.to_image(img)
        mask_t = tv_tensors.Mask(mask)

        if self.mode == "train":
            img_t, mask_t = self._random_scale(img_t, mask_t)
            img_t, mask_t = self.crop(img_t, mask_t)
            img_t, mask_t = self.flip(img_t, mask_t)
        else:
            img_t = F.resize(
                img_t,
                size=list(self.crop_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask_t = F.resize(mask_t, size=list(self.crop_size), interpolation=InterpolationMode.NEAREST)

        img_t = self.normalize(self.to_dtype(img_t))
        img_t = self.to_pure(img_t)
        mask_t = self.to_pure(mask_t).to(torch.int64).squeeze(0)

        return img_t, mask_t

    def _random_scale(self, img: tv_tensors.Image, mask: tv_tensors.Mask):
        min_s, max_s = self.scale_range
        scale = float(torch.empty(()).uniform_(min_s, max_s))
        h, w = int(img.shape[-2]), int(img.shape[-1])
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        img = F.resize(
            img,
            size=[new_h, new_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = F.resize(mask, size=[new_h, new_w], interpolation=InterpolationMode.NEAREST)

        return img, mask
