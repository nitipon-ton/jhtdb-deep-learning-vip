"""
ESRGAN-style generator network (RRDBNet) adapted for turbulence SR.

Reference:
  - Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"

This module implements only the generator (RRDBNet). The training notebook
will treat it as a deterministic SR network, without GAN losses for now.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ESRGANGenerator", "ESRGANDiscriminator"]


class ResidualDenseBlock(nn.Module):
    """
    Residual dense block used in RRDB.

    Consists of 5 conv layers with dense connections and local residual scaling.
    """

    def __init__(
        self,
        channels: int = 64,
        growth_channels: int = 32,
        bias: bool = True,
        scale: float = 0.2,
    ):
        super().__init__()
        self.scale = scale
        in_chs = [channels + i * growth_channels for i in range(5)]

        self.conv1 = nn.Conv2d(in_chs[0], growth_channels, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(in_chs[1], growth_channels, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(in_chs[2], growth_channels, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(in_chs[3], growth_channels, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(in_chs[4], channels, 3, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = x
        out1 = self.lrelu(self.conv1(inputs))
        out2 = self.lrelu(self.conv2(torch.cat([inputs, out1], dim=1)))
        out3 = self.lrelu(self.conv3(torch.cat([inputs, out1, out2], dim=1)))
        out4 = self.lrelu(self.conv4(torch.cat([inputs, out1, out2, out3], dim=1)))
        out5 = self.conv5(torch.cat([inputs, out1, out2, out3, out4], dim=1))
        return inputs + out5 * self.scale


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB).

    Stacks three ResidualDenseBlocks with another residual scaling.
    """

    def __init__(
        self,
        channels: int = 64,
        growth_channels: int = 32,
        bias: bool = True,
        scale: float = 0.2,
    ):
        super().__init__()
        self.block1 = ResidualDenseBlock(channels, growth_channels, bias, scale)
        self.block2 = ResidualDenseBlock(channels, growth_channels, bias, scale)
        self.block3 = ResidualDenseBlock(channels, growth_channels, bias, scale)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        return x + out * self.scale


class ESRGANGenerator(nn.Module):
    """
    RRDBNet-style generator for single-image SR.

    Args:
        in_channels:  Input channels (1 for vorticity).
        out_channels: Output channels.
        nf:           Base number of feature maps.
        gc:           Growth channels inside dense blocks.
        num_blocks:   Number of RRDB blocks.
        scale:        Upsampling factor (supports 2, 4, or 8 via 2× stages).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        nf: int = 64,
        gc: int = 32,
        num_blocks: int = 23,
        scale: int = 4,
    ):
        super().__init__()
        if scale not in {2, 4, 8}:
            raise ValueError(f"ESRGANGenerator currently supports scale=2/4/8, got scale={scale}")

        self.scale = scale
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)

        trunk = [RRDB(nf, gc) for _ in range(num_blocks)]
        self.trunk = nn.Sequential(*trunk)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampling via repeated PixelShuffle x2
        up_layers = []
        num_upsamples = int(scale).bit_length() - 1  # e.g. scale=4 -> 2 stages
        for _ in range(num_upsamples):
            up_layers += [
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsampler = nn.Sequential(*up_layers)

        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(feat))
        feat = feat + trunk
        out = self.upsampler(feat)
        out = self.hr_conv(out)
        out = self.conv_last(out)
        return out


class ESRGANDiscriminator(nn.Module):
    """
    Patch discriminator similar to ESRGAN's VGG-style discriminator.

    Args:
        in_channels: Number of channels in the HR field (1 for omega_z).
        base_channels: Width of the first conv stage; doubles every stride-2 block.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()

        def block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        layers = []
        channels = base_channels
        layers.append(block(in_channels, channels, stride=1))
        layers.append(block(channels, channels, stride=2))
        layers.append(block(channels, channels * 2, stride=1))
        channels *= 2
        layers.append(block(channels, channels, stride=2))
        layers.append(block(channels, channels * 2, stride=1))
        channels *= 2
        layers.append(block(channels, channels, stride=2))
        layers.append(block(channels, channels * 2, stride=1))
        channels *= 2
        layers.append(block(channels, channels, stride=2))

        self.features = nn.Sequential(*layers)

        # Spatial resolution after four stride-2 ops starting from 128 is 8×8.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 70)
    print("Testing ESRGANGenerator (RRDBNet)")
    print("=" * 70)

    net = ESRGANGenerator(scale=4, num_blocks=8)
    x = torch.randn(1, 1, 32, 32)
    y = net(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters:   {_count_parameters(net):,}")
    print("=" * 70)

    disc = ESRGANDiscriminator()
    score = disc(y)
    print(f"Discriminator output shape: {score.shape}")
    print(f"Discriminator params: {_count_parameters(disc):,}")
    print("=" * 70)
