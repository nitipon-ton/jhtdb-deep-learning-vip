from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float | None = None) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target)
    if max_val is None:
        max_val = torch.maximum(target.max(), pred.max()).clamp(min=1.0)
    return 20 * torch.log10(max_val) - 10 * torch.log10(mse + 1e-8)


def ssim_placeholder(_: torch.Tensor, __: torch.Tensor) -> torch.Tensor:
    """Placeholder SSIM implementation for quick smoke tests."""
    return torch.tensor(0.0)

