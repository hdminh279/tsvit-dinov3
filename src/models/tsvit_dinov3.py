"""
TSViT-DINOv3: Factorized Temporal-Spatial Vision Transformer with DINO-v3 Backbone
for Multi-Temporal Satellite Crop Classification.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from einops import rearrange

from .spectral_adapter import SpectralAdapter
from .temporal_transformer import TemporalTransformer
from .dinov3_backbone import DINOv3SpatialEncoder


class TSViT_DINOv3(nn.Module):
    """
    Combines:
      1. Spectral Adapter (10 bands -> token dimension dim)
      2. Factorized Temporal Transformer (over timestamps T with DOY embeddings)
      3. DINO-v3 Spatial Encoder (over spatial patches N)
      4. Dense Segmentation Head (decoding to [B, num_classes, 24, 24])
    """
    def __init__(
        self,
        img_res: int = 24,
        patch_size: int = 2,
        in_channels: int = 10,
        num_classes: int = 19,
        dim: int = 192,
        temporal_depth: int = 4,
        spatial_depth: int = 4,
        heads: int = 4,
        dim_head: int = 48,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        dinov3_model_name: Optional[str] = "facebook/dinov3-vits16-pretrain-lvd1689m"
    ):
        super().__init__()
        self.img_res = img_res
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dim = dim

        self.num_patches_1d = img_res // patch_size
        self.num_patches = self.num_patches_1d ** 2

        # 1. 10-Band Spectral Adapter
        self.spectral_adapter = SpectralAdapter(
            in_channels=in_channels,
            dim=dim,
            patch_size=patch_size
        )

        # 2. Temporal Transformer (from TSViT)
        self.temporal_transformer = TemporalTransformer(
            dim=dim,
            num_classes=num_classes,
            depth=temporal_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        # 3. DINO-v3 Spatial Transformer Backbone
        self.spatial_backbone = DINOv3SpatialEncoder(
            dim=dim,
            depth=spatial_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            model_name=dinov3_model_name
        )

        # 4. Dense Segmentation Head: projects token dim to sub-patch pixels
        self.seg_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_size * patch_size)
        )

    def forward(self, x: torch.Tensor, doy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for crop segmentation.
        Args:
            x: [B, T, 10, 24, 24] normalized multi-temporal satellite imagery
            doy: [B, T] day of year values (optional, defaults to evenly spaced)
        Returns:
            Logits tensor: [B, num_classes, 24, 24]
        """
        B, T, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert H == self.img_res and W == self.img_res, f"Expected ({self.img_res}, {self.img_res}), got ({H}, {W})"

        if doy is None:
            # Fallback DOY: linearly spaced days if not provided
            device = x.device
            doy = torch.linspace(0.0, 1.0, T, device=device).unsqueeze(0).expand(B, -1)

        # 1. Spectral Projection: [B, T, num_patches, dim]
        tokens = self.spectral_adapter(x)

        # 2. Temporal Self-Attention over T with DOY: [B, num_patches, num_classes, dim]
        cls_tokens = self.temporal_transformer(tokens, doy)

        # 3. Spatial Attention using DINO-v3 Backbone across patches
        # Reshape to [(B * num_classes), num_patches, dim]
        spatial_in = rearrange(cls_tokens, 'b n c d -> (b c) n d')
        spatial_out = self.spatial_backbone(spatial_in)

        # 4. Dense Decoder: project each patch token to (patch_size x patch_size) sub-pixels
        logits_flat = self.seg_head(spatial_out)  # [(B * num_classes), num_patches, p1*p2]
        
        # Reshape back to spatial mask [B, num_classes, H, W]
        logits = rearrange(
            logits_flat,
            '(b c) (h1 w1) (p1 p2) -> b c (h1 p1) (w1 p2)',
            b=B,
            c=self.num_classes,
            h1=self.num_patches_1d,
            w1=self.num_patches_1d,
            p1=self.patch_size,
            p2=self.patch_size
        )

        return logits
