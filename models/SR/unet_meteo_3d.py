"""Three-dimensional port of the meteorology-inspired UNet.

The 2D :class:`models.SR.unet_meteo.MeteorologyUNetSR` replaces the 3D
micrometeorology convolutions with 2D convolutions and ``PixelShuffle``.  This
module performs the inverse adaptation for the 128^3 velocity cubes used by
``superresolution-nn``:

- ``Conv3d`` with circular padding, matching the periodic turbulence domain.
- Voxel-shuffle upsampling via einops (PyTorch has no native 3D PixelShuffle).
- Optional trilinear pre-upsampling for the direct coarse -> fine variant.

With ``scale_factor=1`` the model operates at full resolution and predicts the
same residual correction as ``superresolution.models.cnn.SuperResolutionCNN``.
Set ``residual=True`` to add the input back inside the model and return the
reconstructed field instead.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange


def _activation(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    if kind == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{kind}'")


def _normalization(kind: str, channels: int, num_groups: int) -> nn.Module:
    kind = kind.lower()
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if kind == "group":
        if channels % num_groups != 0:
            raise ValueError(
                "Channel count must be divisible by group_norm_groups; "
                f"got {channels} channels and {num_groups} groups."
            )
        return nn.GroupNorm(num_groups, channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(
        f"Unknown normalization '{kind}'. Use batch, instance, group, or none."
    )


class CircularConv3d(nn.Module):
    """Same-size 3D convolution with periodic boundary padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}."
            )
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            padding_mode="circular",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _conv_norm_act(
    in_channels: int,
    out_channels: int,
    *,
    stride: int = 1,
    normalization: str = "batch",
    num_groups: int = 8,
    activation: str = "relu",
) -> nn.Sequential:
    return nn.Sequential(
        CircularConv3d(in_channels, out_channels, stride=stride),
        _normalization(normalization, out_channels, num_groups),
        _activation(activation),
    )


class VoxelShuffle3d(nn.Module):
    """Three-dimensional pixel shuffle: (B, C*r^3, D, H, W) -> (B, C, rD, rH, rW)."""

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}.")
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.scale
        if x.shape[1] % (r**3) != 0:
            raise ValueError(
                f"Channel count {x.shape[1]} is not divisible by r^3={r**3}."
            )
        return rearrange(
            x,
            "b (c r1 r2 r3) d h w -> b c (d r1) (h r2) (w r3)",
            r1=r,
            r2=r,
            r3=r,
        )


class DownBlock3D(nn.Module):
    """Stride-2 convolution followed by ``n_layers - 1`` same-size convs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int = 2,
        normalization: str = "batch",
        num_groups: int = 8,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {n_layers}.")
        layers: List[nn.Module] = [
            _conv_norm_act(
                in_channels,
                out_channels,
                stride=2,
                normalization=normalization,
                num_groups=num_groups,
                activation=activation,
            )
        ]
        for _ in range(n_layers - 1):
            layers.append(
                _conv_norm_act(
                    out_channels,
                    out_channels,
                    normalization=normalization,
                    num_groups=num_groups,
                    activation=activation,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock3D(nn.Module):
    """Voxel-shuffle upsampling, skip concatenation, then a conv stack."""

    def __init__(
        self,
        in_low_channels: int,
        in_skip_channels: int,
        out_channels: int,
        n_layers: int = 2,
        normalization: str = "batch",
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {n_layers}.")
        self.up = nn.Sequential(
            CircularConv3d(in_low_channels, in_low_channels * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            VoxelShuffle3d(scale=2),
        )
        layers: List[nn.Module] = [
            _conv_norm_act(
                in_low_channels + in_skip_channels,
                out_channels,
                normalization=normalization,
                num_groups=num_groups,
                activation="leaky_relu",
            )
        ]
        for _ in range(n_layers - 1):
            layers.append(
                _conv_norm_act(
                    out_channels,
                    out_channels,
                    normalization=normalization,
                    num_groups=num_groups,
                    activation="leaky_relu",
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(coarse)
        merged = torch.cat([upsampled, skip], dim=1)
        return self.block(merged)


class MeteorologyUNetSR3D(nn.Module):
    """3D encoder/decoder with five feature stages.

    Args:
        in_channels: Input channels (3 for ``u, v, w``).
        out_channels: Output channels (3 for ``u, v, w``).
        scale_factor: Input -> output spatial scale. ``1`` operates at full
            resolution (residual-correction mode); ``4`` pre-upsamples from a
            coarse grid.
        feature_channels: Five channel widths, one per stage.
        n_blocks_per_stage: Convolutions per down/up block.
        num_latent_layers: Convolutions at the bottleneck.
        pre_upsample_mode: ``trilinear`` or ``nearest`` when ``scale_factor > 1``.
        normalization: ``batch``, ``instance``, ``group``, or ``none``.
        group_norm_groups: Groups for ``normalization="group"``.
        residual: If true, add the (pre-upsampled) input to the output and
            return the reconstructed field; otherwise return the correction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        scale_factor: int = 1,
        feature_channels: Optional[List[int]] = None,
        n_blocks_per_stage: int = 2,
        num_latent_layers: int = 2,
        pre_upsample_mode: str = "trilinear",
        normalization: str = "batch",
        group_norm_groups: int = 8,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if feature_channels is None:
            feature_channels = [32, 64, 128, 256, 256]
        if len(feature_channels) != 5:
            raise ValueError("feature_channels must have length 5.")
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be at least 1, got {scale_factor}.")
        if n_blocks_per_stage < 1:
            raise ValueError("n_blocks_per_stage must be at least 1.")
        if num_latent_layers < 0:
            raise ValueError("num_latent_layers must be non-negative.")

        self.scale_factor = scale_factor
        self.residual = residual
        self.feature_channels = list(feature_channels)

        if scale_factor == 1:
            self.pre_up = nn.Identity()
        else:
            mode = pre_upsample_mode.lower()
            if mode not in {"trilinear", "nearest"}:
                raise ValueError(
                    "pre_upsample_mode must be trilinear or nearest when "
                    f"scale_factor > 1, got '{pre_upsample_mode}'."
                )
            align_corners = False if mode == "trilinear" else None
            self.pre_up = nn.Upsample(
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
            )

        c0, c1, c2, c3, c4 = feature_channels
        self.stem = _conv_norm_act(
            in_channels,
            c0,
            normalization=normalization,
            num_groups=group_norm_groups,
            activation="relu",
        )
        self.down1 = DownBlock3D(
            c0, c1, n_blocks_per_stage, normalization, group_norm_groups, "relu"
        )
        self.down2 = DownBlock3D(
            c1, c2, n_blocks_per_stage, normalization, group_norm_groups, "relu"
        )
        self.down3 = DownBlock3D(
            c2, c3, n_blocks_per_stage, normalization, group_norm_groups, "relu"
        )
        self.down4 = DownBlock3D(
            c3, c4, n_blocks_per_stage, normalization, group_norm_groups, "relu"
        )

        latent_layers: List[nn.Module] = []
        for _ in range(num_latent_layers):
            latent_layers.append(
                _conv_norm_act(
                    c4,
                    c4,
                    normalization=normalization,
                    num_groups=group_norm_groups,
                    activation="leaky_relu",
                )
            )
        self.latent = nn.Sequential(*latent_layers) if latent_layers else nn.Identity()

        self.up3 = UpBlock3D(
            c4, c3, c3, n_blocks_per_stage, normalization, group_norm_groups
        )
        self.up2 = UpBlock3D(
            c3, c2, c2, n_blocks_per_stage, normalization, group_norm_groups
        )
        self.up1 = UpBlock3D(
            c2, c1, c1, n_blocks_per_stage, normalization, group_norm_groups
        )
        self.up0 = UpBlock3D(
            c1, c0, c0, n_blocks_per_stage, normalization, group_norm_groups
        )

        self.head = nn.Sequential(
            _conv_norm_act(
                c0,
                c0,
                normalization=normalization,
                num_groups=group_norm_groups,
                activation="relu",
            ),
            CircularConv3d(c0, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected (B, C, D, H, W), got {tuple(x.shape)}")
        x_hr = self.pre_up(x)
        spatial = x_hr.shape[-3:]
        if min(spatial) < 32:
            raise ValueError(
                "MeteorologyUNetSR3D needs spatial dimensions of at least 32 "
                "after pre-upsampling because five stride-2 stages shrink them "
                f"by 16x; got {tuple(spatial)}."
            )
        if any(size % 16 != 0 for size in spatial):
            raise ValueError(
                "Spatial dimensions must be divisible by 16 so encoder and "
                f"decoder sizes match; got {tuple(spatial)}."
            )
        x0 = self.stem(x_hr)  # full resolution
        x1 = self.down1(x0)  # 1/2
        x2 = self.down2(x1)  # 1/4
        x3 = self.down3(x2)  # 1/8
        x4 = self.down4(x3)  # 1/16

        latent = self.latent(x4)

        y3 = self.up3(latent, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)
        out = self.head(y0)
        if self.residual:
            out = out + x_hr
        return out
