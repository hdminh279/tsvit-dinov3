"""
Temporal Transformer module inherited from DeepSatModels (TSViT).
Models temporal vegetation phenology across time-series timestamps using Day-of-Year (DOY) encodings
and dedicated per-class temporal queries.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .dinov3_backbone import PreNorm, Attention, FeedForward


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer from TSViT.
    Processes temporal sequence of each patch token along the time dimension T.
    """
    def __init__(
        self,
        dim: int = 192,
        num_classes: int = 19,
        depth: int = 4,
        heads: int = 4,
        dim_head: int = 48,
        mlp_ratio: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.depth = depth

        # DOY 1-365 embedding: Linear projection from 366-dim one-hot vector
        self.to_temporal_pos_embedding = nn.Linear(366, dim)

        # Learnable temporal queries (one per crop class)
        self.temporal_class_tokens = nn.Parameter(torch.randn(1, num_classes, dim) * 0.02)

        # Multi-layer Transformer blocks
        self.layers = nn.ModuleList([])
        mlp_dim = dim * mlp_ratio
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, D] where N = num_patches (e.g. 144)
            doy: [B, T] in range [0, 1) or [0, 365]
        Returns:
            Class token features per patch: [B, N, num_classes, D]
        """
        B, T, N, D = x.shape

        # 1. Compute DOY temporal position encoding (vectorized for ONNX tracing)
        doy_scaled = torch.where(doy <= 1.05, doy * 365.0001, doy)
        doy_days = doy_scaled.long().clamp(0, 365)

        doy_one_hot = F.one_hot(doy_days, num_classes=366).float()  # [B, T, 366]
        temporal_pos = self.to_temporal_pos_embedding(doy_one_hot)   # [B, T, D]

        # 2. Add temporal position embedding to tokens
        # x is [B, T, N, D], temporal_pos is [B, T, 1, D]
        x = x + temporal_pos.unsqueeze(2)

        # Reshape to [(B * N), T, D] for temporal attention
        x = rearrange(x, 'b t n d -> (b n) t d')

        # 3. Prepend learnable temporal class tokens: shape [(B * N), num_classes, D]
        cls_tokens = repeat(self.temporal_class_tokens, '1 c d -> (b n) c d', b=B, n=N)
        x = torch.cat([cls_tokens, x], dim=1)  # [(B * N), num_classes + T, D]

        # 4. Apply Temporal Self-Attention layers
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = self.norm(x)

        # 5. Extract only the updated class query tokens [num_classes]
        cls_out = x[:, :self.num_classes, :]  # [(B * N), num_classes, D]
        cls_out = rearrange(cls_out, '(b n) c d -> b n c d', b=B, n=N)

        return cls_out
