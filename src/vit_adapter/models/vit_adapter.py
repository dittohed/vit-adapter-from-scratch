import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int, stride: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SpatialPriorModule(nn.Module):
    """
    Spatial Prior Module (SPM) that produces multi-scale features at 1/8, 1/16, 1/32.
    Mirrors the paper's conv stem + stride-2 convs, followed by 1x1 projections to embed_dim.
    """

    def __init__(self, in_chans: int = 3, base_channels: int = 64, embed_dim: int = 384):
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_relu(in_chans, base_channels, kernel=3, stride=2, padding=1),
            _conv_bn_relu(base_channels, base_channels, kernel=3, stride=1, padding=1),
            _conv_bn_relu(base_channels, base_channels * 2, kernel=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.down1 = _conv_bn_relu(base_channels * 2, base_channels * 4, kernel=3, stride=2, padding=1)
        self.down2 = _conv_bn_relu(base_channels * 4, base_channels * 8, kernel=3, stride=2, padding=1)
        self.down3 = _conv_bn_relu(base_channels * 8, base_channels * 8, kernel=3, stride=2, padding=1)

        self.proj1 = nn.Conv2d(base_channels * 4, embed_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(base_channels * 8, embed_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(base_channels * 8, embed_dim, kernel_size=1)
        self.out_channels = [embed_dim, embed_dim, embed_dim]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)  # 1/4
        c2 = self.down1(x)  # 1/8
        c3 = self.down2(c2)  # 1/16
        c4 = self.down3(c3)  # 1/32
        c2 = self.proj1(c2)
        c3 = self.proj2(c3)
        c4 = self.proj3(c4)
        return [c2, c3, c4]


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query, key_value, key_value, need_weights=False)
        return self.proj_drop(out)


class SpatialFeatureInjector(nn.Module):
    """
    Injector: inject spatial priors into ViT tokens via cross-attention and layer-scale gamma.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_values: float = 0.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, vit_tokens: torch.Tensor, sp_tokens: torch.Tensor) -> torch.Tensor:
        out = self.attn(self.norm_q(vit_tokens), self.norm_kv(sp_tokens))
        return vit_tokens + out * self.gamma


class ConvFFN(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.25, drop: float = 0.0):
        super().__init__()
        hidden_dim = max(1, int(dim * hidden_ratio))
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, spatial_shapes: Sequence[Tuple[int, int]]):
        x = self.fc1(x)
        x = self.act(x)
        if spatial_shapes is not None:
            x = self._dwconv(x, spatial_shapes)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _dwconv(self, x: torch.Tensor, spatial_shapes: Sequence[Tuple[int, int]]):
        b, n, c = x.shape
        outs = []
        start = 0
        for (h, w) in spatial_shapes:
            length = h * w
            part = x[:, start : start + length, :]
            part = part.transpose(1, 2).reshape(b, c, h, w)
            part = self.dwconv(part)
            part = part.flatten(2).transpose(1, 2)
            outs.append(part)
            start += length
        return torch.cat(outs, dim=1)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.25, drop: float = 0.0):
        super().__init__()
        hidden_dim = max(1, int(dim * hidden_ratio))
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, spatial_shapes: Sequence[Tuple[int, int]] | None = None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatialFeatureExtractor(nn.Module):
    """
    Extractor: pull information from ViT tokens into spatial tokens via cross-attention + FFN.
    Uses ConvFFN (depthwise conv) by default to better mirror the paper's CFFN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 0.25,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop: float = 0.0,
        use_cffn: bool = True,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm_ffn = nn.LayerNorm(dim)
        self.use_cffn = use_cffn
        if use_cffn:
            self.ffn = ConvFFN(dim, hidden_ratio=ffn_ratio, drop=drop)
        else:
            self.ffn = MLP(dim, hidden_ratio=ffn_ratio, drop=drop)

    def forward(
        self,
        sp_tokens: torch.Tensor,
        vit_tokens: torch.Tensor,
        spatial_shapes: Sequence[Tuple[int, int]],
    ) -> torch.Tensor:
        sp_tokens = sp_tokens + self.attn(self.norm_q(sp_tokens), self.norm_kv(vit_tokens))
        sp_tokens = sp_tokens + self.ffn(self.norm_ffn(sp_tokens), spatial_shapes if self.use_cffn else None)
        return sp_tokens


class ViTAdapterBackbone(nn.Module):
    def __init__(
        self,
        vit_name: str = "deit_small_patch16_224",
        pretrained: bool = True,
        base_channels: int = 64,
        interaction_indexes: Sequence[int] | None = None,
        num_interactions: int = 4,
        num_heads: int | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_ratio: float = 0.25,
        use_cffn: bool = True,
        extra_extractors: int = 2,
        init_values: float = 0.0,
        freeze_vit: bool = False,
    ):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=pretrained)
        self.vit.reset_classifier(0)
        # timm ViTs are often configured with strict 224x224 input resolution. For segmentation we
        # train/eval on larger crops (e.g. 512x512), so relax the check and only require divisibility
        # by patch size.
        if hasattr(self.vit, "patch_embed") and hasattr(self.vit.patch_embed, "strict_img_size"):
            self.vit.patch_embed.strict_img_size = False
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.embed_dim = self.vit.embed_dim
        self.depth = len(self.vit.blocks)
        self.num_heads = self._default_num_heads(self.embed_dim) if num_heads is None else num_heads

        self.spm = SpatialPriorModule(in_chans=3, base_channels=base_channels, embed_dim=self.embed_dim)
        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
        )

        if interaction_indexes is None:
            interaction_indexes = self._default_stage_indexes(self.depth, num_interactions)
        self.interaction_indexes = list(interaction_indexes)

        self.injectors = nn.ModuleList(
            [
                SpatialFeatureInjector(
                    self.embed_dim,
                    self.num_heads,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    init_values=init_values,
                )
                for _ in self.interaction_indexes
            ]
        )
        self.extractors = nn.ModuleList(
            [
                SpatialFeatureExtractor(
                    self.embed_dim,
                    self.num_heads,
                    ffn_ratio=ffn_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    use_cffn=use_cffn,
                )
                for _ in self.interaction_indexes
            ]
        )
        self.extra_extractors = nn.ModuleList(
            [
                SpatialFeatureExtractor(
                    self.embed_dim,
                    self.num_heads,
                    ffn_ratio=ffn_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    use_cffn=use_cffn,
                )
                for _ in range(max(0, extra_extractors))
            ]
        )

        self.out_channels = [self.embed_dim] * 4
        self.spm.out_channels = self.out_channels

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

    def _default_stage_indexes(self, depth: int, num_stages: int) -> List[int]:
        return [((i + 1) * depth // num_stages) - 1 for i in range(num_stages)]

    def _default_num_heads(self, embed_dim: int) -> int:
        if embed_dim <= 384:
            return 6
        if embed_dim <= 768:
            return 12
        return 16

    def _resize_pos_embed(self, pos_embed: torch.Tensor, h: int, w: int) -> torch.Tensor:
        n = pos_embed.shape[1]
        if n == 0:
            return pos_embed
        orig_size = int(math.sqrt(n))
        new_size = (h // self.patch_size, w // self.patch_size)
        if orig_size == new_size[0] and orig_size == new_size[1]:
            return pos_embed
        pos = pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=new_size, mode="bicubic", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[1])
        return pos

    def _get_pos_embed(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        pos_embed = self.vit.pos_embed
        if pos_embed is None:
            return torch.zeros(1, (h // self.patch_size) * (w // self.patch_size), self.embed_dim, device=device)
        
        # Drop [CLS] token, not needed for segmentation
        if pos_embed.shape[1] == self.vit.patch_embed.num_patches + 1:
            pos_embed = pos_embed[:, 1:]
        
        # Interpolate classical ViT learnable positional embeddings to higher resolution
        pos_embed = self._resize_pos_embed(pos_embed, h, w)
        return pos_embed.to(device)

    def _flatten_spatial(self, feats: Sequence[torch.Tensor]):
        tokens = []
        shapes: List[Tuple[int, int]] = []
        for lvl, feat in enumerate(feats):
            b, c, h, w = feat.shape
            shapes.append((h, w))
            t = feat.flatten(2).transpose(1, 2)
            t = t + self.level_embed[lvl].view(1, 1, -1)
            tokens.append(t)
        return torch.cat(tokens, dim=1), shapes

    def _unflatten_spatial(self, tokens: torch.Tensor, shapes: Sequence[Tuple[int, int]]):
        outs = []
        start = 0
        for (h, w) in shapes:
            length = h * w
            part = tokens[:, start : start + length, :]
            b, _, c = part.shape
            part = part.transpose(1, 2).reshape(b, c, h, w)
            outs.append(part)
            start += length
        return outs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, _, h, w = x.shape
        
        # Extract "spatial priors" and flatten to tokens
        sp_feats = self.spm(x)
        sp_tokens, spatial_shapes = self._flatten_spatial(sp_feats)

        # Prepare ViT tokens for transformer blocks
        tokens = self.vit.patch_embed(x)
        pos_embed = self._get_pos_embed(h, w, tokens.device)
        tokens = tokens + pos_embed
        tokens = self.vit.pos_drop(tokens)

        # Backbone-adapter interactions
        start = 0
        for idx, end in enumerate(self.interaction_indexes):
            tokens = self.injectors[idx](tokens, sp_tokens)
            for blk in self.vit.blocks[start : end + 1]:
                tokens = blk(tokens)
            sp_tokens = self.extractors[idx](sp_tokens, tokens, spatial_shapes)

            if idx == len(self.interaction_indexes) - 1 and len(self.extra_extractors) > 0:
                for extra in self.extra_extractors:
                    sp_tokens = extra(sp_tokens, tokens, spatial_shapes)

            start = end + 1

        sp_feats = self._unflatten_spatial(sp_tokens, spatial_shapes)
        c2, c3, c4 = sp_feats
        c1 = self.up4(c2)
        return [c1, c2, c3, c4]
