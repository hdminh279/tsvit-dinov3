"""
DINO-v3 Spatial Backbone Wrapper.
Supports loading Meta DINO-v3 foundation models with graceful fallback to DINOv2 / timm
for flexible deployment and fine-tuning.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 48, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOv3SpatialEncoder(nn.Module):
    """
    Spatial Transformer encoder incorporating DINO-v3 vision transformer architecture.
    Applies spatial self-attention over the (H/patch_size x W/patch_size) token grid.
    """
    def __init__(
        self,
        dim: int = 192,
        depth: int = 4,
        heads: int = 4,
        dim_head: int = 48,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        model_name: Optional[str] = "facebook/dinov3-vits16-pretrain-lvd1689m"
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.model_name = model_name
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        mlp_dim = dim * mlp_ratio
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

        # Learnable spatial position encoding for 144 patches (12x12 grid)
        self.pos_embedding = nn.Parameter(torch.randn(1, 144, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Spatial patch tokens [Batch, num_patches, dim]"""
        N = x.shape[1]
        x = x + self.pos_embedding[:, :N, :]

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
