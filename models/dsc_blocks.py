"""
Building blocks for Hybrid DSC/MS network (Fukami et al. 2019).

Matches the original Keras implementation - standard convolutions, no BatchNorm.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'")


class ConvBlock(nn.Module):
    """
    Standard convolution block: Conv2d + Activation (NO BatchNorm).

    Matches Fukami's original implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True  # Fukami uses bias
        )
        self.act = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.act(x)


class MultiScaleBlock(nn.Module):
    """
    Multi-scale convolution block matching Fukami et al. (2019).

    Architecture:
        Branch 1: input → [16, 8, 8] with 5×5 kernels (3 layers)
        Branch 2: input → [16, 8, 8] with 9×9 kernels (3 layers)
        Branch 3: input → [16, 8, 8] with 13×13 kernels (3 layers)

        Concat: [8, 8, 8, input] = 25 channels
        Fuse:   25 → 8 → 3 (7×7, then 5×5)
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 16,
        small_channels: int = 8,
        out_channels: int = 3,
        kernel_sizes: Iterable[int] = (5, 9, 13),
        activation: str = "relu",
    ):
        super().__init__()

        # Create 3 parallel branches with different kernel sizes
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                ConvBlock(in_channels, mid_channels, kernel_size=k, activation=activation),
                ConvBlock(mid_channels, small_channels, kernel_size=k, activation=activation),
                ConvBlock(small_channels, small_channels, kernel_size=k, activation=activation),
            )
            self.branches.append(branch)

        # Fusion: concat all branches + input, then fuse
        fuse_in = small_channels * len(self.branches) + in_channels
        self.fuse = nn.Sequential(
            ConvBlock(fuse_in, small_channels, kernel_size=7, activation=activation),
            ConvBlock(small_channels, out_channels, kernel_size=5, activation=activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process each branch
        features = [branch(x) for branch in self.branches]

        # Add input to features and concatenate
        features.append(x)
        fused = torch.cat(features, dim=1)

        # Final fusion
        return self.fuse(fused)


class UpsampleBlock(nn.Module):
    """
    Upsample block: nearest-neighbor upsample (2×) followed by conv.

    Matches Fukami's UpSampling2D((2,2)) followed by Conv2D.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.scale = scale
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return self.conv(x)


class PixelShuffleUpsample(nn.Module):
    """
    Learned upsampling using sub-pixel convolution (PixelShuffle).

    This is MUCH better than bicubic for SR - the network learns how to upsample!

    For 2× upsampling:
        - Conv to scale²×C channels
        - Rearrange to (H×scale, W×scale, C)

    Example: 128×128×64 → conv → 128×128×256 → shuffle → 256×256×64
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.scale = scale

        # Conv to generate scale² × out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (scale ** 2),
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.shuffle = nn.PixelShuffle(scale)
        self.act = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.shuffle(x)  # Rearrange to higher resolution
        return self.act(x)
