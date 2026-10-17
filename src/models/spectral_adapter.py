"""
Spectral Adapter for 10-band Sentinel-2 imagery.
Maps multi-spectral satellite channels into latent token representations suitable for DINO-v3.
"""

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class SpectralAdapter(nn.Module):
    """
    Projects 10 Sentinel-2 bands to embedding dimension dim.
    Supports either 2D Patch Convolution or Linear Patch Projection.
    """
    def __init__(
        self,
        in_channels: int = 10,
        dim: int = 192,
        patch_size: int = 2,
        adapter_type: str = "conv"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.patch_size = patch_size
        self.adapter_type = adapter_type

        if adapter_type == "conv":
            # 2D Conv projection: [B*T, 10, H, W] -> [B*T, dim, H/p, W/p]
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
                nn.GroupNorm(num_groups=min(8, dim), num_channels=dim),
                nn.GELU()
            )
        elif adapter_type == "linear":
            # Linear projection: patch_dim = in_channels * patch_size * patch_size
            patch_dim = in_channels * patch_size * patch_size
            self.proj = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim)
            )
        elif adapter_type == "rgb_synth":
            # 10 bands to 3 channels (RGB synthesis) for direct input to standard 3-channel backbones
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1),
                nn.BatchNorm2d(3),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 10, H, W]
        Returns:
            If 'conv': [B, T, num_patches, dim]
            If 'linear': [B, T, num_patches, dim]
            If 'rgb_synth': [B*T, 3, H, W]
        """
        B, T, C, H, W = x.shape
        x_reshaped = x.reshape(B * T, C, H, W)

        if self.adapter_type == "conv":
            out = self.proj(x_reshaped)  # [B*T, dim, h_p, w_p]
            _, d, hp, wp = out.shape
            out = rearrange(out, '(b t) d h w -> b t (h w) d', b=B, t=T)
            return out
        elif self.adapter_type == "linear":
            out = self.proj(x_reshaped)  # [B*T, num_patches, dim]
            out = rearrange(out, '(b t) n d -> b t n d', b=B, t=T)
            return out
        elif self.adapter_type == "rgb_synth":
            return self.proj(x_reshaped)
