"""
Specialised metrics for analysing 2D turbulence fields.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def energy_spectrum(field: np.ndarray, num_bins: int = 24, log_bins: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an azimuthal 1D energy spectrum."""
    fft = np.fft.rfft2(field)
    energy = np.abs(fft) ** 2
    ny, nx_half = energy.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx_half), indexing="ij")
    radii = np.sqrt(xx**2 + yy**2)
    r_max = radii.max()
    if log_bins:
        bins = np.logspace(0, np.log10(r_max + 1), num_bins + 1)
    else:
        bins = np.linspace(0, r_max + 1, num_bins + 1)
    spectrum = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (radii >= bins[i]) & (radii < bins[i + 1])
        if np.any(mask):
            spectrum[i] = energy[mask].mean()
    centres = 0.5 * (bins[:-1] + bins[1:])
    return centres, spectrum


def enstrophy(vorticity: np.ndarray) -> float:
    return float(np.mean(vorticity**2))


def dissipation(velocity: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> float:
    ux, uy = velocity
    dux_dx = np.gradient(ux, dx, axis=-1)
    dux_dy = np.gradient(ux, dy, axis=-2)
    duy_dx = np.gradient(uy, dx, axis=-1)
    duy_dy = np.gradient(uy, dy, axis=-2)
    strain_sq = 2 * (dux_dx**2 + duy_dy**2) + (dux_dy + duy_dx) ** 2
    return float(np.mean(strain_sq))


def two_point_correlation(field: np.ndarray, axis: int = -1) -> np.ndarray:
    mean_field = field - field.mean()
    corr = np.correlate(mean_field.flatten(), mean_field.flatten(), mode="full")
    corr = corr[corr.size // 2 :]
    corr /= corr[0]
    return corr


def summary_report(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    corr = float(np.corrcoef(pred.ravel(), target.ravel())[0, 1])
    return {"mse": mse, "mae": mae, "corr": corr}
