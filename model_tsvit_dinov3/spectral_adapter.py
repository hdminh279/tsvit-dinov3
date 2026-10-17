"""
Spectral Adapter for 10-band Sentinel-2 imagery.
Projects 10 multi-spectral bands to token dimension dim via 2D patch convolution.
"""

import torch
import torch.nn as nn
from einops import rearrange


class SpectralAdapter(nn.Module):
    """
    Projects 10 Sentinel-2 bands to embedding dimension dim.
    Converts [B, T, 10, 24, 24] -> [B, T, 144, dim] with patch_size=2.
    """
    def __init__(self, in_channels: int = 10, dim: int = 192, patch_size: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(num_groups=min(8, dim), num_channels=dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 10, H, W]
        Returns:
            [B, T, num_patches, dim]
        """
        B, T, C, H, W = x.shape
        out = self.proj(x.reshape(B * T, C, H, W))
        return rearrange(out, '(b t) d h w -> b t (h w) d', b=B, t=T)
