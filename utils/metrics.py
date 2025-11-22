from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float | None = None) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target)
    if max_val is None:
        max_val = torch.maximum(target.max(), pred.max()).clamp(min=1.0)
    return 20 * torch.log10(max_val) - 10 * torch.log10(mse + 1e-8)


def _create_ssim_window(window_size: int, channel: int, device, dtype) -> torch.Tensor:
    sigma = 1.5
    gauss = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    gauss = torch.exp(-(gauss**2) / (2 * sigma**2))
    gauss = (gauss / gauss.sum()).unsqueeze(1)
    window = gauss @ gauss.t()
    window = window / window.sum()
    return window.expand(channel, 1, window_size, window_size)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    Structural Similarity Index (single-channel, batched).

    Args:
        pred: (B, C, H, W) or (C, H, W) tensor
        target: same shape as pred
        data_range: Value range of the data. If None, inferred from target/pred
        window_size: Size of Gaussian window
        k1, k2: SSIM stabilization constants

    Returns:
        Scalar tensor with mean SSIM over batch.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    b, c, h, w = pred.shape
    if h < window_size or w < window_size:
        raise ValueError("Input spatial dimensions must be >= window_size")

    if data_range is None:
        data_range = torch.max(
            torch.stack(
                [
                    pred.max() - pred.min(),
                    target.max() - target.min(),
                    torch.tensor(1.0, device=pred.device, dtype=pred.dtype),
                ]
            )
        ).item()

    window = _create_ssim_window(window_size, c, pred.device, pred.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=c)
    mu_y = F.conv2d(target, window, padding=padding, groups=c)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, window, padding=padding, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=padding, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=c) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / (denominator + 1e-12)
    return ssim_map.mean()

