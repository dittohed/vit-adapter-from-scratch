import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
import torch


def _to_tuple(size):
    if isinstance(size, int):
        return (size, size)
    return tuple(size)


class SegmentationTransform:
    def __init__(
        self,
        mode: str,
        crop_size=512,
        scale_range=(0.5, 2.0),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        ignore_index=255,
    ):
        self.mode = mode
        self.crop_size = _to_tuple(crop_size)
        self.scale_range = scale_range
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.ignore_index = ignore_index

    def __call__(self, img: Image.Image, mask: Image.Image):
        if self.mode == "train":
            img, mask = self._random_scale(img, mask)
            img, mask = self._random_crop(img, mask)
            img, mask = self._random_flip(img, mask)
        else:
            img = img.resize(self.crop_size[::-1], Image.BILINEAR)
            mask = mask.resize(self.crop_size[::-1], Image.NEAREST)

        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = (img - self.mean) / self.std

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img, mask

    def _random_scale(self, img: Image.Image, mask: Image.Image):
        scale = random.uniform(*self.scale_range)
        new_w = max(1, int(round(img.width * scale)))
        new_h = max(1, int(round(img.height * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        return img, mask

    def _random_crop(self, img: Image.Image, mask: Image.Image):
        crop_h, crop_w = self.crop_size
        pad_h = max(crop_h - img.height, 0)
        pad_w = max(crop_w - img.width, 0)
        if pad_h > 0 or pad_w > 0:
            img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=self.ignore_index)

        x_max = img.width - crop_w
        y_max = img.height - crop_h
        x1 = random.randint(0, max(0, x_max))
        y1 = random.randint(0, max(0, y_max))
        img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        return img, mask

    def _random_flip(self, img: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask
