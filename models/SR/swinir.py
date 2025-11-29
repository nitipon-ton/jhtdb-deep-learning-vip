"""
Lightweight SwinIR implementation for turbulence super-resolution.

This adapts the SwinIR SR architecture to a single-scale, small variant
appropriate for 32×32 → 128×128 super-resolution.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition (B, H, W, C) feature map into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition."""
    B = int(windows.shape[0] // (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # Get pairwise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, dropout=drop)

    def create_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        cnt = 0
        for h in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
            for w in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        shortcut = x

        x = self.norm1(x.view(B, -1, C)).view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_r, 0, pad_b)
        shifted_x = F.pad(shifted_x, pad_values)
        _, Hp, Wp, _ = shifted_x.shape

        windows = window_partition(shifted_x, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, C)

        attn_mask = self.create_mask(Hp, Wp, shifted_x.device)
        attn_windows = self.attn(windows, attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        shifted_x = shifted_x[:, :H, :W, :]
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x.view(B, -1, C))).view(B, H, W, C))
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        window_size: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        blocks = []
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class Upsample(nn.Module):
    def __init__(self, scale: int, num_feat: int):
        super().__init__()
        modules = []
        for _ in range(int(scale).bit_length() - 1):
            modules += [
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SwinIRLite(nn.Module):
    """
    Lightweight SwinIR for one-channel turbulence SR.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        embed_dim: int = 64,
        depths: int = 6,
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        scale: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.scale = scale

        self.conv_first = nn.Conv2d(in_ch, embed_dim, 3, 1, 1)
        self.body = BasicLayer(
            dim=embed_dim,
            depth=depths,
            window_size=window_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.trunk_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsampler = Upsample(scale=scale, num_feat=embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, out_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        B, C, H, W = feat.shape
        x_body = feat.permute(0, 2, 3, 1).contiguous()
        x_body = self.body(x_body)
        x_body = x_body.permute(0, 3, 1, 2).contiguous()
        feat = feat + self.trunk_conv(x_body)
        out = self.upsampler(feat)
        out = self.conv_last(out)
        return out


if __name__ == "__main__":
    net = SwinIRLite(scale=4)
    x = torch.randn(1, 1, 32, 32)
    y = net(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
