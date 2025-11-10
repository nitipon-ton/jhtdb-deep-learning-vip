from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_groupnorm: bool = True):
        super().__init__()
        # Use GroupNorm instead of BatchNorm for small batch sizes
        # GroupNorm is batch-size independent and works well for SR
        if use_groupnorm:
            # Use 8 groups (or out_ch if fewer channels)
            num_groups = min(8, out_ch) if out_ch >= 8 else out_ch
            norm1 = nn.GroupNorm(num_groups, out_ch)
            norm2 = nn.GroupNorm(num_groups, out_ch)
        else:
            norm1 = nn.BatchNorm2d(out_ch)
            norm2 = nn.BatchNorm2d(out_ch)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm1,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            norm2,
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_groupnorm: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch, use_groupnorm=use_groupnorm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetSRx4(nn.Module):
    """Compact U-Net for x4 SR on single-channel fields."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 32, scale: int = 4, use_groupnorm: bool = True):
        super().__init__()
        self.scale = scale
        self.enc1 = ConvBlock(in_ch, base_ch, use_groupnorm=use_groupnorm)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, use_groupnorm=use_groupnorm)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, use_groupnorm=use_groupnorm)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8, use_groupnorm=use_groupnorm)

        self.up2 = UpBlock(base_ch * 8, base_ch * 4, use_groupnorm=use_groupnorm)
        self.up1 = UpBlock(base_ch * 4, base_ch * 2, use_groupnorm=use_groupnorm)
        self.up0 = UpBlock(base_ch * 2, base_ch, use_groupnorm=use_groupnorm)

        self.out = nn.Conv2d(base_ch, out_ch * (scale**2), kernel_size=1)
        self.pixshuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))

        d2 = self.up2(bn, e3)
        d1 = self.up1(d2, e2)
        d0 = self.up0(d1, e1)
        out = self.out(d0)
        return self.pixshuffle(out)


class UNetSRx8(nn.Module):
    """
    U-Net for x8 SR on single-channel turbulence fields.

    Uses PixelShuffle for x8 upsampling (outputs 64 channels for x8).
    Designed for 64×64 LR → 512×512 HR.

    Uses GroupNorm instead of BatchNorm for stability with small batch sizes.

    Args:
        in_ch: Input channels (default: 1)
        out_ch: Output channels (default: 1)
        base_ch: Base channel width (default: 24 for ~4.4M params)
        scale: Upsampling scale (must be 8)
        use_groupnorm: Use GroupNorm instead of BatchNorm (default: True)
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 24, scale: int = 8, use_groupnorm: bool = True):
        super().__init__()
        assert scale == 8, "UNetSRx8 only supports scale=8"
        self.scale = scale

        # Encoder (downsampling path)
        self.enc1 = ConvBlock(in_ch, base_ch, use_groupnorm=use_groupnorm)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, use_groupnorm=use_groupnorm)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, use_groupnorm=use_groupnorm)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, use_groupnorm=use_groupnorm)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16, use_groupnorm=use_groupnorm)

        # Decoder (upsampling path with skip connections)
        self.up3 = UpBlock(base_ch * 16, base_ch * 8, use_groupnorm=use_groupnorm)
        self.up2 = UpBlock(base_ch * 8, base_ch * 4, use_groupnorm=use_groupnorm)
        self.up1 = UpBlock(base_ch * 4, base_ch * 2, use_groupnorm=use_groupnorm)
        self.up0 = UpBlock(base_ch * 2, base_ch, use_groupnorm=use_groupnorm)

        # Output: PixelShuffle x8 requires 64 output channels (8^2 = 64)
        self.out = nn.Conv2d(base_ch, out_ch * (scale**2), kernel_size=1)
        self.pixshuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, in_ch, 64, 64)

        Returns:
            Output tensor (B, out_ch, 512, 512)
        """
        # Encoder
        e1 = self.enc1(x)           # (B, base_ch, 64, 64)
        e2 = self.enc2(self.pool(e1))  # (B, base_ch*2, 32, 32)
        e3 = self.enc3(self.pool(e2))  # (B, base_ch*4, 16, 16)
        e4 = self.enc4(self.pool(e3))  # (B, base_ch*8, 8, 8)

        # Bottleneck
        bn = self.bottleneck(self.pool(e4))  # (B, base_ch*16, 4, 4)

        # Decoder with skip connections
        d3 = self.up3(bn, e4)  # (B, base_ch*8, 8, 8)
        d2 = self.up2(d3, e3)  # (B, base_ch*4, 16, 16)
        d1 = self.up1(d2, e2)  # (B, base_ch*2, 32, 32)
        d0 = self.up0(d1, e1)  # (B, base_ch, 64, 64)

        # Output projection and PixelShuffle x8
        out = self.out(d0)  # (B, out_ch*64, 64, 64)
        return self.pixshuffle(out)  # (B, out_ch, 512, 512)

