from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int) -> nn.GroupNorm:
    # Prefer 32 groups when possible (common default), but fall back gracefully
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


class FPN(nn.Module):
    """
    Implementation of "vanilla" FPN from "Feature Pyramid Networks for Object Detection":
    1) normalize each pyramid level to `out_channels` channels via 1x1 conv
    2) top-down upsample + element-wise add to a next level
    3) 3x3 conv to reduce aliasing

    Compared to the original, this implementation adds extra GroupNorm+ReLU after all convs.

    Expects features ordered high->low resolution, e.g. [1/4, 1/8, 1/16, 1/32].
    Returns features in the same order, with `out_channels` channels for each level.
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()

        self.lateral_convs = nn.ModuleList()  # 1x1 convs to unify channels to `out_channels`
        self.output_convs = nn.ModuleList()  # 3x3 convs to reduce aliasing after fusion

        for ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, kernel_size=1, bias=False),
                    _gn(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    _gn(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(feats) == len(self.lateral_convs), "Number of input features must match in_channels"

        laterals = [lat(x) for x, lat in zip(feats, self.lateral_convs)]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=target_size, mode="bilinear", align_corners=False
            )

        outs = [conv(x) for x, conv in zip(laterals, self.output_convs)]

        return outs


class SemanticFPNHead(nn.Module):
    """
    Implementation of "semantic" FPN head from "Panoptic Feature Pyramid Networks"
    that first runs "vanilla" FPN and then merges its outputs:
    1) run vanilla FPN to get 4 feature maps at 1/4, 1/8, 1/16, 1/32 scales
    2) for each FPN level, apply K upsampling stages to reach 1/4 scale with 128 channels:
        3x3 conv -> GroupNorm -> ReLU -> 2x bilinear upsample
      where K is 3/2/1/0 for strides 32/16/8/4 respectively.
    3) element-wise sum all 1/4 maps
    4) final 1x1 conv to produce per-class logits at 1/4 scale

    Expects features ordered high->low resolution, e.g. [1/4, 1/8, 1/16, 1/32].
    The output should be upsampled back from 1/4 to full input resolution.
    """

    def __init__(
        self,
        in_channels: List[int],
        fpn_out_channels: int = 256,  # No. of "vanilla" FPN output channels
        semantic_out_channels: int = 128,  # No. of "semantic" FPN output channels
        num_classes: int = 150,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fpn = FPN(in_channels=in_channels, out_channels=fpn_out_channels)
        self.dropout = dropout
        self.classifier = nn.Conv2d(semantic_out_channels, num_classes, kernel_size=1)

        # Build scale heads
        self.scale_heads = nn.ModuleList()  # Each head produces 128 channels at 1/4 scale from a single FPN level

        for i in range(len(in_channels)):
            layers: List[nn.Module] = []

            if i == 0:
                layers.extend(
                    [
                        nn.Conv2d(fpn_out_channels, semantic_out_channels, kernel_size=3, padding=1, bias=False),
                        _gn(semantic_out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
            else:
                in_ch = fpn_out_channels
                for _ in range(i):
                    layers.extend(
                        [
                            nn.Conv2d(in_ch, semantic_out_channels, kernel_size=3, padding=1, bias=False),
                            _gn(semantic_out_channels),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        ]
                    )
                    in_ch = semantic_out_channels

            self.scale_heads.append(nn.Sequential(*layers))

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # Vanilla FPN part
        fpn_feats = self.fpn(feats)
        target_size = fpn_feats[0].shape[-2:]

        # Semantic FPN part
        # 1) Upsample all FPN outputs to 1/4 scale
        outs = []
        for feat, head in zip(fpn_feats, self.scale_heads):
            out = head(feat)
            if out.shape[-2:] != target_size:
                print(f"Warning: unexpected feature map size {out.shape[-2:]} after scale head, resizing to match target size {target_size}...")
                out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
            outs.append(out)

        # 2) Element-wise sum all 1/4 maps
        x = outs[0]
        for out in outs[1:]:
            x = x + out

        # 3) Final classifier on 1/4 scale
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)

        return x
