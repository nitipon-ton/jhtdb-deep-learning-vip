"""
Hybrid DSC/MS Network (Fukami et al. 2019)

Exact PyTorch implementation of the architecture from:
"Super-resolution reconstruction of turbulent flows with machine learning"
K. Fukami, K. Fukagata, & K. Taira, Journal of Fluid Mechanics, 2019

Key features:
- Hybrid Down-sampled Skip-Connection (DSC) branch
- Multi-Scale (MS) branch with parallel convolutions
- NO pre-upsampling (works on native resolution)
- Standard Conv2d (not depthwise separable)
- NO BatchNorm
- For 4× SR: 128×128 input → 512×512 output
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dsc_blocks import ConvBlock, MultiScaleBlock, UpsampleBlock, PixelShuffleUpsample


class HDSCMS(nn.Module):
    """
    Hybrid DSC/MS backbone matching Fukami et al. (2019).

    Architecture:
        DSC branch: 4 levels with progressively smaller downsampling
            Level 1: pool(8×) → conv → up(2×)
            Level 2: pool(4×) → concat → conv → up(2×)
            Level 3: pool(2×) → concat → conv → up(2×)
            Level 4: input → concat → conv

        MS branch: 3 parallel paths with different kernel sizes
            Path 1: 5×5 kernels
            Path 2: 9×9 kernels
            Path 3: 13×13 kernels

        Final: concat(DSC, MS) → conv → output

    Args:
        in_ch: Input channels (1 for vorticity)
        out_ch: Output channels (1 for vorticity)
        dsc_width: Channel width for DSC branch (Fukami: 32)
        ms_mid: Mid channels for MS branch (Fukami: 16)
        ms_small: Small channels for MS branch (Fukami: 8)
        ms_out: Final MS output channels (Fukami: 3)
        activation: Activation function
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        dsc_width: int = 32,
        ms_mid: int = 16,
        ms_small: int = 8,
        ms_out: int = 3,
        activation: str = "relu",
    ):
        super().__init__()

        # DSC Branch - Down-sampled skip connections
        # Level 1: pool(8×) → 2 convs → up(2×)
        self.pool8 = nn.AvgPool2d(8, ceil_mode=True)
        self.dsc_8 = nn.Sequential(
            ConvBlock(in_ch, dsc_width, kernel_size=3, activation=activation),
            ConvBlock(dsc_width, dsc_width, kernel_size=3, activation=activation),
        )
        self.up_8 = UpsampleBlock(dsc_width, dsc_width, scale=2, activation=activation)

        # Level 2: pool(4×) → concat with up_8 → 2 convs → up(2×)
        self.pool4 = nn.AvgPool2d(4, ceil_mode=True)
        self.dsc_4 = nn.Sequential(
            ConvBlock(in_ch + dsc_width, dsc_width, kernel_size=3, activation=activation),
            ConvBlock(dsc_width, dsc_width, kernel_size=3, activation=activation),
        )
        self.up_4 = UpsampleBlock(dsc_width, dsc_width, scale=2, activation=activation)

        # Level 3: pool(2×) → concat with up_4 → 2 convs → up(2×)
        self.pool2 = nn.AvgPool2d(2, ceil_mode=True)
        self.dsc_2 = nn.Sequential(
            ConvBlock(in_ch + dsc_width, dsc_width, kernel_size=3, activation=activation),
            ConvBlock(dsc_width, dsc_width, kernel_size=3, activation=activation),
        )
        self.up_2 = UpsampleBlock(dsc_width, dsc_width, scale=2, activation=activation)

        # Level 4: input → concat with up_2 → 2 convs
        self.dsc_final = nn.Sequential(
            ConvBlock(in_ch + dsc_width, dsc_width, kernel_size=3, activation=activation),
            ConvBlock(dsc_width, dsc_width, kernel_size=3, activation=activation),
        )

        # MS Branch - Multi-scale parallel convolutions
        self.ms = MultiScaleBlock(
            in_channels=in_ch,
            mid_channels=ms_mid,
            small_channels=ms_small,
            out_channels=ms_out,
            kernel_sizes=(5, 9, 13),
            activation=activation,
        )

        # Final fusion: concat(DSC, MS) → conv
        self.final = nn.Conv2d(dsc_width + ms_out, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid DSC/MS network.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W) - same spatial size!
        """
        # DSC Branch
        # Level 1: downsample 8×, process, upsample 2×
        d8 = self.pool8(x)
        y8 = self.dsc_8(d8)
        y8u = self.up_8(y8)

        # Level 2: downsample 4×, concat skip, process, upsample 2×
        d4 = self.pool4(x)
        y4 = self.dsc_4(torch.cat([y8u, d4], dim=1))
        y4u = self.up_4(y4)

        # Level 3: downsample 2×, concat skip, process, upsample 2×
        d2 = self.pool2(x)
        y2 = self.dsc_2(torch.cat([y4u, d2], dim=1))
        y2u = self.up_2(y2)

        # Level 4: concat with input, final processing
        y_dsc = self.dsc_final(torch.cat([y2u, x], dim=1))

        # MS Branch
        y_ms = self.ms(x)

        # Fuse both branches
        fused = torch.cat([y_dsc, y_ms], dim=1)
        out = self.final(fused)

        return out


class HybridDSCMSR(nn.Module):
    """
    Hybrid DSC/MS Super-Resolution wrapper with LEARNED upsampling.

    For 4× SR (128→512):
        - Input: LR 128×128
        - Backbone: processes at 128×128 (NO pre-upsampling!)
        - Learned upsample: 128→256→512 (PixelShuffle 2×2)
        - Output: 512×512

    Key difference from bicubic: Network LEARNS how to upsample!

    Args:
        in_ch: Input channels
        out_ch: Output channels
        scale: Upsampling scale factor (4 for 128→512)
        lr_size: LR input size (128)
        hr_size: HR output size (512)
        dsc_width: DSC branch width (default 64 for good capacity)
        ms_mid: MS mid channels (16)
        ms_small: MS small channels (8)
        ms_out: MS output channels (3)
        act: Activation function
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        scale: int = 4,
        lr_size: int | None = None,
        hr_size: int | None = None,
        dsc_width: int = 64,
        ms_mid: int = 16,
        ms_small: int = 8,
        ms_out: int = 3,
        act: str = "relu",
    ):
        super().__init__()
        self.scale = scale
        self.lr_size = lr_size
        self.hr_size = hr_size if hr_size is not None else (
            lr_size * scale if lr_size is not None else None
        )

        # Backbone network (works at LR resolution!)
        self.backbone = HDSCMS(
            in_ch=in_ch,
            out_ch=dsc_width,  # Output more channels for upsampling
            dsc_width=dsc_width,
            ms_mid=ms_mid,
            ms_small=ms_small,
            ms_out=ms_out,
            activation=act,
        )

        # Learned upsampling layers (2× then 2× = 4×)
        # 128→256→512
        if scale == 4:
            self.upsample = nn.Sequential(
                PixelShuffleUpsample(dsc_width, dsc_width, scale=2, activation=act),  # 128→256
                PixelShuffleUpsample(dsc_width, dsc_width, scale=2, activation=act),  # 256→512
                nn.Conv2d(dsc_width, out_ch, kernel_size=3, padding=1),  # Final 1-channel output
            )
        elif scale == 8:
            # For 8× SR: 64→128→256→512
            self.upsample = nn.Sequential(
                PixelShuffleUpsample(dsc_width, dsc_width, scale=2, activation=act),
                PixelShuffleUpsample(dsc_width, dsc_width, scale=2, activation=act),
                PixelShuffleUpsample(dsc_width, dsc_width, scale=2, activation=act),
                nn.Conv2d(dsc_width, out_ch, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Only scale=4 or scale=8 supported, got {scale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: LR → backbone → LEARNED upsample → HR.

        No more bicubic! Network learns optimal upsampling for turbulence.

        Args:
            x: LR input (B, C, H, W) where H=W=128

        Returns:
            HR output (B, C, H', W') where H'=W'=512
        """
        # Process at native LR resolution (128×128)
        features = self.backbone(x)  # → (B, dsc_width, 128, 128)

        # Learned upsampling to HR
        out = self.upsample(features)  # → (B, 1, 512, 512)

        return out
