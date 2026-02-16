import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationModel(nn.Module):
    def __init__(self, backbone: nn.Module, decode_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decode_head(feats)

        # Both UperNet and SemanticFPN produce 1/4 scale logits, so still need to upsample
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return logits
