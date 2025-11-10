"""
Enhanced Deep Super-Resolution (EDSR) Network

Implementation based on:
"Enhanced Deep Residual Networks for Single Image Super-Resolution"
Lim et al., CVPRW 2017

Key features:
- Residual blocks WITHOUT BatchNorm (improves SR quality)
- Global residual learning (add LR to upsampled output)
- PixelShuffle for learned upsampling
- Scalable architecture (control depth and width)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block for EDSR (no BatchNorm!).

    Architecture:
        input → Conv → ReLU → Conv → (×res_scale) → (+input) → output
    """

    def __init__(
        self,
        n_channels: int = 64,
        kernel_size: int = 3,
        res_scale: float = 1.0,
    ):
        super().__init__()
        self.res_scale = res_scale

        self.body = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x) * self.res_scale
        return x + res


class Upsampler(nn.Module):
    """
    Efficient sub-pixel upsampler using PixelShuffle.

    For scale=4: Uses 2 stages of 2× upsampling
    For scale=8: Uses 3 stages of 2× upsampling
    """

    def __init__(
        self,
        scale: int,
        n_channels: int = 64,
    ):
        super().__init__()

        modules = []
        if scale in [2, 4, 8]:
            for _ in range(scale // 2):
                modules.append(nn.Conv2d(n_channels, 4 * n_channels, 3, padding=1))
                modules.append(nn.PixelShuffle(2))
        elif scale == 3:
            modules.append(nn.Conv2d(n_channels, 9 * n_channels, 3, padding=1))
            modules.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale: {scale}. Use 2, 3, 4, or 8.")

        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class EDSR(nn.Module):
    """
    Enhanced Deep Super-Resolution Network.

    Architecture:
        Input → Conv (feature extraction)
          ↓
        Residual blocks (deep features, no BN!)
          ↓
        Conv (feature fusion)
          ↓
        Upsampler (PixelShuffle)
          ↓
        Conv (reconstruction)
          ↓
        Output (+ global residual from bicubic input)

    Args:
        scale: Upsampling factor (2, 3, 4, or 8)
        n_resblocks: Number of residual blocks (16 for EDSR, 32 for EDSR+)
        n_feats: Number of feature channels (64 for EDSR, 256 for EDSR+)
        res_scale: Residual scaling factor (0.1 for deep networks)
        in_channels: Input channels (1 for vorticity)
        out_channels: Output channels (1 for vorticity)
    """

    def __init__(
        self,
        scale: int = 4,
        n_resblocks: int = 16,
        n_feats: int = 64,
        res_scale: float = 1.0,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.scale = scale

        # Initial feature extraction
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)

        # Residual blocks
        body = []
        for _ in range(n_resblocks):
            body.append(ResidualBlock(n_feats, kernel_size=3, res_scale=res_scale))
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*body)

        # Upsampling
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with global residual learning.

        Args:
            x: LR input (B, C, H, W)

        Returns:
            SR output (B, C, H*scale, W*scale)
        """
        # Bicubic upsample input for global residual
        x_upsampled = nn.functional.interpolate(
            x,
            scale_factor=self.scale,
            mode='bicubic',
            align_corners=False
        )

        # Extract features
        x_feat = self.head(x)

        # Process through residual blocks
        res = self.body(x_feat)
        res += x_feat  # Global residual connection

        # Upsample and reconstruct
        x_sr = self.tail(res)

        # Add global residual (bicubic baseline)
        x_sr += x_upsampled

        return x_sr


class EDSRLite(nn.Module):
    """
    Lightweight EDSR for faster training and lower memory.

    Uses fewer blocks and channels than full EDSR.
    Good for initial experiments and benchmarking.

    Args:
        scale: Upsampling factor (2, 4, or 8)
        n_resblocks: Number of residual blocks (default: 8)
        n_feats: Number of feature channels (default: 32)
        in_channels: Input channels (1 for vorticity)
        out_channels: Output channels (1 for vorticity)
    """

    def __init__(
        self,
        scale: int = 4,
        n_resblocks: int = 8,
        n_feats: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()

        # Reuse EDSR architecture with lighter settings
        self.model = EDSR(
            scale=scale,
            n_resblocks=n_resblocks,
            n_feats=n_feats,
            res_scale=1.0,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Convenience function for getting param counts
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test EDSR
    print("=" * 70)
    print("Testing EDSR Models")
    print("=" * 70)

    # Full EDSR
    model_full = EDSR(scale=4, n_resblocks=16, n_feats=64)
    x = torch.randn(1, 1, 128, 128)
    y = model_full(x)
    print(f"\nEDSR (16 blocks, 64 feats):")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {count_parameters(model_full):,}")

    # Lite EDSR
    model_lite = EDSRLite(scale=4, n_resblocks=8, n_feats=32)
    y_lite = model_lite(x)
    print(f"\nEDSRLite (8 blocks, 32 feats):")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y_lite.shape}")
    print(f"  Params: {count_parameters(model_lite):,}")

    print("\n" + "=" * 70)
