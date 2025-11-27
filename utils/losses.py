"""
Custom loss components for turbulence super-resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.fft


class GradientLoss(nn.Module):
    """L2 loss on spatial gradients (helps sharpen shear layers)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        return torch.mean((pred_dx - target_dx) ** 2) + torch.mean(
            (pred_dy - target_dy) ** 2
        )


class SpectralLoss(nn.Module):
    """
    Penalise differences in the azimuthally averaged energy spectrum.

    The implementation is intentionally simple – it computes the FFT magnitude,
    averages radially using a set of concentric rings, and applies an L2 loss.
    """

    def __init__(self, num_bins: int = 20, log_bins: bool = True):
        super().__init__()
        self.num_bins = num_bins
        self.log_bins = log_bins

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_spec = torch.fft.rfft2(pred, norm="ortho").abs() ** 2
        target_spec = torch.fft.rfft2(target, norm="ortho").abs() ** 2

        _, _, h, w_half = pred_spec.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, device=pred.device),
            torch.arange(w_half, device=pred.device),
            indexing="ij",
        )
        radii = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
        r_max = radii.max().item()

        if self.log_bins:
            # MPS doesn't support logspace - compute on CPU then move
            log_end = torch.log10(torch.tensor(r_max + 1.0)).item()
            bins = torch.logspace(0, log_end, self.num_bins + 1).to(pred.device)
        else:
            bins = torch.linspace(0.0, r_max + 1.0, self.num_bins + 1, device=pred.device)

        loss = pred_spec.new_zeros(())
        for i in range(self.num_bins):
            mask = (radii >= bins[i]) & (radii < bins[i + 1])
            if mask.sum() == 0:
                continue
            pred_band = pred_spec[..., mask].mean()
            target_band = target_spec[..., mask].mean()
            loss = loss + (pred_band - target_band) ** 2
        return loss / self.num_bins


class TotalVariationLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return (dx.abs().mean() + dy.abs().mean()) / 2


class RelativisticAverageGANLoss(nn.Module):
    """
    Relativistic average GAN (RaGAN) loss used in ESRGAN.

    Provides generator and discriminator losses using BCE-with-logits.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def generator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Relativistic generator loss.

        Args:
            real_pred: Discriminator logits on real samples.
            fake_pred: Discriminator logits on fake samples (G output).
        """
        real_label = torch.zeros_like(real_pred)
        fake_label = torch.ones_like(fake_pred)

        mean_fake = torch.mean(fake_pred)
        mean_real = torch.mean(real_pred)

        loss_real = self.criterion(real_pred - mean_fake.detach(), real_label)
        loss_fake = self.criterion(fake_pred - mean_real.detach(), fake_label)
        return 0.5 * (loss_real + loss_fake)

    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Relativistic discriminator loss.

        Args:
            real_pred: Discriminator logits on real samples.
            fake_pred: Discriminator logits on fake samples.
        """
        real_label = torch.ones_like(real_pred)
        fake_label = torch.zeros_like(fake_pred)

        mean_fake = torch.mean(fake_pred.detach())
        mean_real = torch.mean(real_pred.detach())

        loss_real = self.criterion(real_pred - mean_fake, real_label)
        loss_fake = self.criterion(fake_pred - mean_real, fake_label)
        return 0.5 * (loss_real + loss_fake)
