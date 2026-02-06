from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticFPNHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        channels: int = 256,
        num_classes: int = 150,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for ch in in_channels:
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

        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels * len(in_channels), channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = dropout
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        laterals = [lat(feat) for feat, lat in zip(feats, self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=target_size, mode="bilinear", align_corners=False
            )

        outs = [fpn(lat) for lat, fpn in zip(laterals, self.fpn_convs)]
        target_size = outs[0].shape[-2:]
        for i in range(1, len(outs)):
            outs[i] = F.interpolate(outs[i], size=target_size, mode="bilinear", align_corners=False)

        x = torch.cat(outs, dim=1)
        x = self.bottleneck(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x
