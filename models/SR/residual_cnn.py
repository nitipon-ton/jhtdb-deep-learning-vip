"""Full-resolution residual CNN for two-dimensional turbulence super-resolution.

The model mirrors the project's earlier 3D velocity CNN in two dimensions:

1. Upsample the low-resolution vorticity with a fixed interpolation baseline.
2. Process that full-resolution field with periodic residual convolutions.
3. Predict a correction and add it to the interpolation baseline.

The returned tensor is the reconstructed high-resolution field, so this model
can use the same datasets, losses, and training loop as the other SR models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalization(kind: str, channels: int, group_norm_groups: int) -> nn.Module:
    kind = kind.lower()
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "group":
        if channels % group_norm_groups != 0:
            raise ValueError(
                "hidden_channels must be divisible by group_norm_groups; "
                f"got {channels} and {group_norm_groups}."
            )
        return nn.GroupNorm(group_norm_groups, channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(
        f"Unknown normalization '{kind}'. Use batch, instance, group, or none."
    )


def _activation(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "silu":
        return nn.SiLU(inplace=True)
    if kind == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation '{kind}'. Use relu, silu, or gelu.")


class CircularConv2d(nn.Module):
    """A same-size 2D convolution with periodic boundary padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}.")

        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode="circular",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBlock2d(nn.Module):
    """Periodic Conv -> Norm -> Act -> Conv -> Norm residual block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        normalization: str,
        group_norm_groups: int,
        activation: str,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}.")

        self.conv1 = CircularConv2d(channels, channels, kernel_size, dilation)
        self.norm1 = _normalization(normalization, channels, group_norm_groups)
        self.act1 = _activation(activation)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = CircularConv2d(channels, channels, kernel_size, dilation)
        self.norm2 = _normalization(normalization, channels, group_norm_groups)
        self.act2 = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.act2(x + residual)


class ResidualCorrectionCNN2d(nn.Module):
    """Upsample first, then predict a full-resolution vorticity correction."""

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 32,
        num_residual_blocks: int = 4,
        kernel_size: int = 3,
        dilation: int = 1,
        normalization: str = "batch",
        group_norm_groups: int = 8,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        upsample_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        if scale <= 1:
            raise ValueError(f"scale must be greater than one, got {scale}.")
        if in_channels != out_channels:
            raise ValueError(
                "Residual baseline addition requires matching input/output channels; "
                f"got {in_channels} and {out_channels}."
            )
        if hidden_channels <= 0 or num_residual_blocks <= 0:
            raise ValueError("hidden_channels and num_residual_blocks must be positive.")
        if upsample_mode not in {"bilinear", "bicubic"}:
            raise ValueError("upsample_mode must be bilinear or bicubic.")

        self.scale = scale
        self.upsample_mode = upsample_mode
        self.input_projection = nn.Sequential(
            CircularConv2d(in_channels, hidden_channels, kernel_size, dilation),
            _normalization(normalization, hidden_channels, group_norm_groups),
            _activation(activation),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock2d(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    normalization=normalization,
                    group_norm_groups=group_norm_groups,
                    activation=activation,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_residual_blocks)
            ]
        )
        self.output_projection = CircularConv2d(
            hidden_channels, out_channels, kernel_size, dilation
        )

    def forward(self, low_resolution: torch.Tensor) -> torch.Tensor:
        baseline = F.interpolate(
            low_resolution,
            scale_factor=self.scale,
            mode=self.upsample_mode,
            align_corners=False,
        )
        features = self.input_projection(baseline)
        correction = self.output_projection(self.blocks(features))
        return baseline + correction

