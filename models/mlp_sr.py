"""
Fully-connected super-resolution models.

These models operate on flattened LR patches and predict the corresponding HR
patch.  They are useful as light baselines or for quick experiments where
convolutional inductive biases are not required.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(
        self,
        lr_size: int,
        hr_size: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        activation: str = "relu",
    ):
        super().__init__()
        self.lr_size = lr_size
        self.hr_size = hr_size
        act = nn.ReLU if activation == "relu" else nn.GELU

        dims = [lr_size * lr_size] + [hidden_dim] * num_layers + [hr_size * hr_size]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != dims[-1]:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        out = self.net(x.view(b, -1))
        return out.view(b, 1, self.hr_size, self.hr_size)


class MLPSuperResolution(nn.Module):
    def __init__(
        self,
        lr_size: int,
        hr_size: int,
        hidden_dims: Sequence[int] = (1024, 2048, 2048, 1024),
        dropout_p: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.lr_size = lr_size
        self.hr_size = hr_size
        act = nn.ReLU if activation == "relu" else nn.GELU

        dims = [lr_size * lr_size] + list(hidden_dims) + [hr_size * hr_size]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != dims[-1]:
                layers.append(act())
                layers.append(nn.Dropout(dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        out = self.net(x.view(b, -1))
        return out.view(b, 1, self.hr_size, self.hr_size)
