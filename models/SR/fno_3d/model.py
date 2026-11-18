"""Self-contained 3D hybrid Fourier Neural Operator for velocity-residual SR.

Inputs and outputs are (B, 3, D, H, W). The model predicts the correction,
not the reconstructed DNS field: DNS_prediction = input + model(input).

Training data in the reference project is already cubic-spline upsampled to
128^3, so the network does NOT perform another upsampling operation.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv3d(nn.Module):
    """3D spectral convolution using four non-overlapping low-frequency corners.

    The rFFT retains only nonnegative frequencies on the last (W) axis.
    Both signs of low frequencies are needed on the first two axes.
    """

    def __init__(self, width: int, modes: Sequence[int]):
        super().__init__()
        if len(modes) != 3 or any(int(m) < 1 for m in modes):
            raise ValueError('modes must have 3 positive integers: [D, H, W]')
        self.width = int(width)
        self.modes = tuple(int(m) for m in modes)
        shape = (self.width, self.width, *self.modes)
        scale = 1 / (self.width * self.width)
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.complex(torch.randn(shape), torch.randn(shape)))
            for _ in range(4)
        ])

    @staticmethod
    def multiply(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bcxyz,coxyz->boxyz', x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        md, mh, mw = self.modes
        if md > d // 2 or mh > h // 2 or mw > w // 2 + 1:
            raise ValueError(
                f'Fourier modes {self.modes} exceed non-overlapping frequency '
                f'limits for spatial shape {(d, h, w)}; try smaller modes.'
            )

        # Keep the FFT in float32 even when AMP autocast uses fp16/bf16.
        # FFT low-precision support varies across GPUs and resolutions.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm='ortho')
            out = torch.zeros(
                (b, c, d, h, w // 2 + 1),
                dtype=xf.dtype, device=x.device
            )
            out[:, :, :md, :mh, :mw] = self.multiply(
                xf[:, :, :md, :mh, :mw], self.weights[0]
            )
            out[:, :, -md:, :mh, :mw] = self.multiply(
                xf[:, :, -md:, :mh, :mw], self.weights[1]
            )
            out[:, :, :md, -mh:, :mw] = self.multiply(
                xf[:, :, :md, -mh:, :mw], self.weights[2]
            )
            out[:, :, -md:, -mh:, :mw] = self.multiply(
                xf[:, :, -md:, -mh:, :mw], self.weights[3]
            )
            y = torch.fft.irfftn(out, s=(d, h, w), dim=(-3, -2, -1), norm='ortho')
        return y.to(dtype=x.dtype)


class HybridFNOBlock3d(nn.Module):
    def __init__(
        self, width: int, modes: Sequence[int], local_kernel: int = 3,
        local_path: bool = True, residual_blocks: bool = True,
    ):
        super().__init__()
        if local_kernel not in (1, 3, 5):
            raise ValueError('local_kernel must be 1, 3, or 5')
        self.spectral = SpectralConv3d(width, modes)
        self.pointwise = nn.Conv3d(width, width, 1)
        self.local = (
            nn.Conv3d(width, width, local_kernel,
                      padding=local_kernel // 2, padding_mode='circular')
            if local_path else None
        )
        self.residual_blocks = residual_blocks
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.spectral(x) + self.pointwise(x)
        if self.local is not None:
            z = z + self.local(x)
        if self.residual_blocks:
            z = z + x
        return self.activation(z)


class HybridFNO3D(nn.Module):
    """Map preinterpolated three-channel velocity cubes to residual corrections.

    width: learned feature channels
    modes: retained 3D Fourier mode counts [D,H,W]
    depth: number of hybrid Fourier/local blocks
    coordinates: append normalized grid coordinates (off by default, especially
        for homogeneous periodic turbulence where translation equivariance helps).
    """
    def __init__(
        self, in_channels: int = 3, out_channels: int = 3,
        width: int = 16, modes: Sequence[int] = (8, 8, 8), depth: int = 3,
        local_kernel: int = 3, local_path: bool = True,
        residual_blocks: bool = True, coordinates: bool = False,
        zero_init_head: bool = True,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, width, depth) < 1:
            raise ValueError('in_channels, out_channels, width and depth must be positive')
        self.coordinates = coordinates
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lift = nn.Conv3d(in_channels + (3 if coordinates else 0), width, 1)
        self.blocks = nn.ModuleList([
            HybridFNOBlock3d(width, modes, local_kernel, local_path, residual_blocks)
            for _ in range(depth)
        ])
        self.project = nn.Sequential(
            nn.Conv3d(width, width, 1),
            nn.GELU(),
            nn.Conv3d(width, out_channels, 1),
        )
        if zero_init_head:
            nn.init.zeros_(self.project[-1].weight)
            nn.init.zeros_(self.project[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError(f'Expected (B, {self.in_channels}, D, H, W), got {tuple(x.shape)}')
        if self.coordinates:
            d, h, w = x.shape[-3:]
            axes = [torch.arange(n, device=x.device, dtype=x.dtype) / max(n, 1)
                    for n in (d, h, w)]
            coords = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=0)
            coords = coords.unsqueeze(0).expand(x.size(0), -1, -1, -1, -1)
            x = torch.cat((x, coords), dim=1)
        features = self.lift(x)
        for block in self.blocks:
            features = block(features)
        return self.project(features)

    @classmethod
    def from_config(cls, cfg: dict) -> 'HybridFNO3D':
        return cls(**cfg)
