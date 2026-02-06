from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, pool_scales: Iterable[int], out_channels: int):
        super().__init__()
        self.pool_scales = pool_scales
        self.convs = nn.ModuleList()
        for scale in pool_scales:
            self.convs.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [x]
        for conv in self.convs:
            out = conv(x)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        return self.bottleneck(x)


class UperNetHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        channels: int = 256,
        pool_scales=(1, 2, 3, 6),
        num_classes: int = 150,
    ):
        super().__init__()
        assert len(in_channels) == 4, "UperNet expects 4 feature levels"
        self.in_channels = in_channels
        self.channels = channels

        self.ppm = PyramidPoolingModule(in_channels[-1], pool_scales, channels)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for ch in in_channels[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = feats
        ppm_out = self.ppm(c4)

        laterals = [self.lateral_convs[0](c1), self.lateral_convs[1](c2), self.lateral_convs[2](c3)]
        laterals.append(ppm_out)

        for i in range(2, -1, -1):
            target_size = laterals[i].shape[-2:]
            laterals[i] = laterals[i] + F.interpolate(laterals[i + 1], size=target_size, mode="bilinear", align_corners=False)

        fpn_outs = [
            self.fpn_convs[0](laterals[0]),
            self.fpn_convs[1](laterals[1]),
            self.fpn_convs[2](laterals[2]),
            laterals[3],
        ]

        target_size = fpn_outs[0].shape[-2:]
        for i in range(1, 4):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=target_size, mode="bilinear", align_corners=False)

        x = torch.cat(fpn_outs, dim=1)
        x = self.fpn_bottleneck(x)
        x = self.classifier(x)
        return x
