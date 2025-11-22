"""
UNet variant inspired by the SR architecture used in
`3d-sr-micrometeorology`. The original version operates on 3D voxels with
gated/partial convolutions and voxel shuffle. This module adapts the
high-level structure to 2D turbulence SR by:

- Replacing 3D convolutions with standard 2D convolutions.
- Using PixelShuffle-based upsampling inside each decoder block.
- Preserving the multi-stage encoder/decoder depth and latent conv stack.

The model upsamples the LR input once at the beginning (bilinear) and learns a
refined HR prediction with wide residual paths.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def _conv_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int | None = None,
    bias: bool = True,
    activation: Optional[str] = "relu",
) -> nn.Sequential:
    padding = kernel_size // 2 if padding is None else padding
    layers: List[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    ]
    if activation is not None:
        act_layer: nn.Module
        if activation == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            act_layer = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation '{activation}'")
        layers.append(act_layer)
    return nn.Sequential(*layers)


class DownBlock2D(nn.Module):
    """Downsampling block: stride-2 conv followed by (n-1) conv layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        assert n_layers >= 1
        layers = [
            _conv_act(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                activation=activation,
            )
        ]
        for _ in range(n_layers - 1):
            layers.append(_conv_act(out_channels, out_channels, activation=activation))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock2D(nn.Module):
    """Upsampling block: PixelShuffle + conv stack with skip concat."""

    def __init__(
        self,
        in_low_channels: int,
        in_skip_channels: int,
        out_channels: int,
        n_layers: int = 2,
    ):
        super().__init__()
        assert n_layers >= 1
        self.up = nn.Sequential(
            nn.Conv2d(in_low_channels, in_low_channels * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2),
        )
        layers = [
            _conv_act(
                in_low_channels + in_skip_channels,
                out_channels,
                activation="leaky_relu",
            )
        ]
        for _ in range(n_layers - 1):
            layers.append(_conv_act(out_channels, out_channels, activation="leaky_relu"))
        self.block = nn.Sequential(*layers)

    def forward(self, coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up(coarse)
        merged = torch.cat([up, skip], dim=1)
        return self.block(merged)


class MeteorologyUNetSR(nn.Module):
    """
    2D adaptation of the meteorology UNet.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        scale_factor: SR scale (HR/LR).
        feature_channels: List[int] defining channel width per stage
            (length 4 or 5). Example: [32, 64, 128, 256].
        n_blocks_per_stage: number of conv layers per down/up block.
        num_latent_layers: number of conv layers at bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scale_factor: int = 4,
        feature_channels: Optional[List[int]] = None,
        n_blocks_per_stage: int = 2,
        num_latent_layers: int = 2,
        pre_upsample_mode: str = "bilinear",
    ):
        super().__init__()
        if feature_channels is None:
            feature_channels = [32, 64, 128, 256, 256]

        if len(feature_channels) != 5:
            raise ValueError("feature_channels must have length 5 for this implementation.")

        self.scale_factor = scale_factor
        if scale_factor == 1:
            self.pre_up = nn.Identity()
        else:
            align = pre_upsample_mode in {"bilinear", "bicubic"}
            self.pre_up = nn.Upsample(
                scale_factor=scale_factor,
                mode=pre_upsample_mode,
                align_corners=False if align else None,
            )

        c0, c1, c2, c3, c4 = feature_channels

        self.stem = _conv_act(in_channels, c0, activation="relu")

        self.down1 = DownBlock2D(c0, c1, n_layers=n_blocks_per_stage)
        self.down2 = DownBlock2D(c1, c2, n_layers=n_blocks_per_stage)
        self.down3 = DownBlock2D(c2, c3, n_layers=n_blocks_per_stage)
        self.down4 = DownBlock2D(c3, c4, n_layers=n_blocks_per_stage)

        latent_in = c4
        latent_layers = []
        for _ in range(num_latent_layers):
            latent_layers.append(_conv_act(latent_in, latent_in, activation="leaky_relu"))
        self.latent = nn.Sequential(*latent_layers) if latent_layers else nn.Identity()

        self.up3 = UpBlock2D(latent_in, c3, c3, n_layers=n_blocks_per_stage)
        self.up2 = UpBlock2D(c3, c2, c2, n_layers=n_blocks_per_stage)
        self.up1 = UpBlock2D(c2, c1, c1, n_layers=n_blocks_per_stage)
        self.up0 = UpBlock2D(c1, c0, c0, n_layers=n_blocks_per_stage)

        self.head = nn.Sequential(
            _conv_act(c0, c0, activation="relu"),
            nn.Conv2d(c0, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hr = self.pre_up(x)
        x0 = self.stem(x_hr)
        x1 = self.down1(x0)  # 1/2
        x2 = self.down2(x1)  # 1/4
        x3 = self.down3(x2)  # 1/8
        x4 = self.down4(x3)  # 1/16

        latent = self.latent(x4)

        y3 = self.up3(latent, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)
        return self.head(y0)
