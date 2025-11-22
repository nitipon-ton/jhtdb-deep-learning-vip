"""
Hybrid Down-sampled Skip-Connection / Multi-Scale model (Fukami et al. 2019).

The implementation below mirrors the original Keras reference code shared by
Kai Fukami.  The network operates on HR-sized inputs, so for super-resolution
we first upsample the LR field and then predict a refined HR correction.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

__all__ = ["FukamiHDSCMSBackbone", "HybridDSCMSR"]


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    activation: bool = True,
) -> nn.Sequential:
    """Utility: Conv2d (+ optional ReLU) with same padding."""
    padding = kernel_size // 2
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class FukamiHDSCMSBackbone(nn.Module):
    """
    Pure PyTorch transcription of hDSC/MS (Fukami et al. 2019).

    The architecture contains two parallel branches:
      1. DSC branch – cascaded max-pooling blocks with skip connections.
      2. MS branch – three multi-scale convolution paths fused together.

    The network keeps spatial resolution fixed; use HybridDSCMSR for SR tasks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dsc_width: int = 32,
        ms_mid: int = 16,
        ms_small: int = 8,
        ms_out: int = 3,
    ):
        super().__init__()
        act = nn.ReLU(inplace=True)

        # DSC branch pools with stride 8/4/2 and upsamples with factor 2 (nearest).
        self.pool8 = nn.MaxPool2d(kernel_size=8, stride=8, ceil_mode=True)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.dsc1 = nn.Sequential(
            _conv_block(in_channels, dsc_width, 3),
            _conv_block(dsc_width, dsc_width, 3),
        )
        self.dsc2 = nn.Sequential(
            _conv_block(in_channels + dsc_width, dsc_width, 3),
            _conv_block(dsc_width, dsc_width, 3),
        )
        self.dsc3 = nn.Sequential(
            _conv_block(in_channels + dsc_width, dsc_width, 3),
            _conv_block(dsc_width, dsc_width, 3),
        )
        self.dsc4 = nn.Sequential(
            _conv_block(in_channels + dsc_width, dsc_width, 3),
            _conv_block(dsc_width, dsc_width, 3),
        )

        # MS branch (paths with kernel sizes 5/9/13 as in the reference code)
        self.ms1 = nn.Sequential(
            _conv_block(in_channels, ms_mid, 5),
            _conv_block(ms_mid, ms_small, 5),
            _conv_block(ms_small, ms_small, 5),
        )
        self.ms2 = nn.Sequential(
            _conv_block(in_channels, ms_mid, 9),
            _conv_block(ms_mid, ms_small, 9),
            _conv_block(ms_small, ms_small, 9),
        )
        self.ms3 = nn.Sequential(
            _conv_block(in_channels, ms_mid, 13),
            _conv_block(ms_mid, ms_small, 13),
            _conv_block(ms_small, ms_small, 13),
        )
        self.ms_fuse = nn.Sequential(
            _conv_block(ms_small * 3 + in_channels, ms_small, 7),
            _conv_block(ms_small, ms_out, 5),
        )

        self.final = _conv_block(dsc_width + ms_out, out_channels, 3, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DSC branch
        d1 = self.pool8(x)
        y1 = self.dsc1(d1)
        y1u = self.up(y1)

        d2 = self.pool4(x)
        y2 = self.dsc2(torch.cat([y1u, d2], dim=1))
        y2u = self.up(y2)

        d3 = self.pool2(x)
        y3 = self.dsc3(torch.cat([y2u, d3], dim=1))
        y3u = self.up(y3)

        y_dsc = self.dsc4(torch.cat([y3u, x], dim=1))

        # MS branch
        ms_concat = torch.cat([self.ms1(x), self.ms2(x), self.ms3(x), x], dim=1)
        y_ms = self.ms_fuse(ms_concat)

        return self.final(torch.cat([y_dsc, y_ms], dim=1))


class HybridDSCMSR(nn.Module):
    """
    Wrapper that upsamples LR inputs before passing through the Fukami backbone.

    Args:
        in_ch:  Input channels (e.g., 1 for omega_z).
        out_ch: Output channels.
        scale:  Super-resolution scale factor (hr_size / lr_size).
        upsample_mode: 'bilinear', 'bicubic', or 'nearest' for the pre-upsampling.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        scale: int = 4,
        upsample_mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear",
        dsc_width: int = 32,
        ms_mid: int = 16,
        ms_small: int = 8,
        ms_out: int = 3,
        lr_size: Optional[int] = None,
        hr_size: Optional[int] = None,
    ):
        super().__init__()
        if scale <= 0 and lr_size and hr_size:
            scale = hr_size // lr_size
        self.scale = max(1, scale)

        if self.scale == 1:
            self.pre_up = nn.Identity()
        else:
            if upsample_mode in {"bilinear", "bicubic"}:
                self.pre_up = nn.Upsample(
                    scale_factor=self.scale,
                    mode=upsample_mode,
                    align_corners=False,
                )
            else:
                self.pre_up = nn.Upsample(scale_factor=self.scale, mode=upsample_mode)

        self.backbone = FukamiHDSCMSBackbone(
            in_channels=in_ch,
            out_channels=out_ch,
            dsc_width=dsc_width,
            ms_mid=ms_mid,
            ms_small=ms_small,
            ms_out=ms_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hr = self.pre_up(x)
        return self.backbone(x_hr)
