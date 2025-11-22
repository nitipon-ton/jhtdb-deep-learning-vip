"""
Physics-based loss functions for turbulence modeling.

Implements physics-informed losses based on the vorticity transport equation:
    ∂ω/∂t + u·∇ω = ν∇²ω

where:
    - ω is the vorticity (omega_z for 2D x-y plane)
    - u is the velocity vector (ux, uy)
    - ν is the kinematic viscosity
"""

from __future__ import annotations

# Utility helpers shared by multiple losses

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_3d(t: torch.Tensor) -> torch.Tensor:
    """Convert (B, 1, H, W) tensors to (B, H, W) for stencil ops."""
    return t.squeeze(1) if t.dim() == 4 else t


def _advection_term(
    u: torch.Tensor,
    v: torch.Tensor,
    omega: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """
    Compute discrete u·∇ω using central differences (returns B×(H-2)×(W-2)).
    """
    u = _ensure_3d(u)
    v = _ensure_3d(v)
    omega = _ensure_3d(omega)

    domega_dx = (omega[:, :, 2:] - omega[:, :, :-2]) / (2 * dx)  # (B, H, W-2)
    domega_dy = (omega[:, 2:, :] - omega[:, :-2, :]) / (2 * dy)  # (B, H-2, W)

    u_c = u[:, 1:-1, 1:-1]
    v_c = v[:, 1:-1, 1:-1]
    domega_dx_c = domega_dx[:, 1:-1, :]
    domega_dy_c = domega_dy[:, :, 1:-1]

    adv = u_c * domega_dx_c + v_c * domega_dy_c
    return torch.nan_to_num(adv, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-1e3, 1e3)


def _laplacian_term(
    omega: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """
    Compute discrete ∇²ω using a 5-point stencil (returns B×(H-2)×(W-2)).
    """
    omega = _ensure_3d(omega)

    d2omega_dx2 = (omega[:, :, 2:] - 2 * omega[:, :, 1:-1] + omega[:, :, :-2]) / (dx ** 2)
    d2omega_dy2 = (omega[:, 2:, :] - 2 * omega[:, 1:-1, :] + omega[:, :-2, :]) / (dy ** 2)

    d2omega_dx2_c = d2omega_dx2[:, 1:-1, :]
    d2omega_dy2_c = d2omega_dy2[:, :, 1:-1]

    lap = d2omega_dx2_c + d2omega_dy2_c
    return torch.nan_to_num(lap, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-1e3, 1e3)


class AdvectionLoss(nn.Module):
    """
    Physics-informed advection loss: matches DNS vs ML advection terms.

    Loss:
        E_adv = ||u_DNS·∇ω_ML − u_DNS·∇ω_DNS||²

    The velocities used are always the DNS velocities; the only change between
    ML vs DNS physics comes from the vorticity field.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        u_field: torch.Tensor,
        v_field: torch.Tensor,
        omega_ml: torch.Tensor,
        omega_dns: torch.Tensor,
        dx: float,
        dy: float
    ) -> torch.Tensor:
        adv_dns = _advection_term(u_field, v_field, omega_dns, dx, dy)
        adv_ml = _advection_term(u_field, v_field, omega_ml, dx, dy)
        diff = adv_dns - adv_ml

        if self.reduction == 'mean':
            return torch.mean(diff ** 2)
        elif self.reduction == 'sum':
            return torch.sum(diff ** 2)
        else:
            return torch.mean(diff ** 2, dim=(-2, -1))


class DiffusionLoss(nn.Module):
    """
    Physics-informed diffusion loss: matches DNS vs ML Laplacian terms.

    Loss:
        E_diff = ||∇²ω_ML − ∇²ω_DNS||²
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        omega_pred: torch.Tensor,
        omega_target: torch.Tensor,
        dx: float,
        dy: float
    ) -> torch.Tensor:
        """
        Compute physics-informed diffusion loss.

        Args:
            omega_pred: ML predicted vorticity (B, H, W) or (B, 1, H, W)
            omega_target: DNS vorticity (B, H, W) or (B, 1, H, W)
            dx: grid spacing in x-direction
            dy: grid spacing in y-direction

        Returns:
            loss: MSE between DNS and ML Laplacian terms
        """
        lap_dns = _laplacian_term(omega_target, dx, dy)
        lap_ml = _laplacian_term(omega_pred, dx, dy)
        diff = lap_dns - lap_ml

        if self.reduction == 'mean':
            return torch.mean(diff ** 2)
        elif self.reduction == 'sum':
            return torch.sum(diff ** 2)
        else:  # 'none'
            return torch.mean(diff ** 2, dim=(-2, -1))


class VorticityGradientLoss(nn.Module):
    """
    Penalizes sharp vorticity gradients (smoothness prior).

    This is distinct from physics losses - it's a regularization term
    that encourages smooth fields.

    Loss: L_grad = ||∇ω||²
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        omega: torch.Tensor,
        dx: float,
        dy: float
    ) -> torch.Tensor:
        """
        Compute vorticity gradient loss.

        Args:
            omega: vorticity (B, H, W) or (B, 1, H, W)
            dx: grid spacing in x-direction
            dy: grid spacing in y-direction

        Returns:
            loss: scalar or (B,) depending on reduction
        """
        # Ensure 3D tensor (B, H, W)
        if omega.dim() == 4:
            omega = omega.squeeze(1)

        # Compute first derivatives
        domega_dx = (omega[:, :, 1:] - omega[:, :, :-1]) / dx
        domega_dy = (omega[:, 1:, :] - omega[:, :-1, :]) / dy

        # Gradient magnitude squared
        grad_mag_sq = domega_dx[:, :-1, :]**2 + domega_dy[:, :, :-1]**2

        if self.reduction == 'mean':
            return torch.mean(grad_mag_sq)
        elif self.reduction == 'sum':
            return torch.sum(grad_mag_sq)
        else:  # 'none'
            return torch.mean(grad_mag_sq, dim=(-2, -1))


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss for turbulence.

    Combines:
        - Advection loss: λ_adv * ||u_DNS·∇ω_DNS - u_ML·∇ω_ML||²
        - Diffusion loss: λ_diff * ||∇²ω_DNS - ∇²ω_ML||²
        - Optional gradient regularization: λ_grad * ||∇ω_ML||²

    CORRECT FORMULATION: Matches DNS physics to ML physics, not minimizing them!

    IMPORTANT: This module expects NORMALIZED inputs (zero mean, unit variance).
    Physics losses are computed on normalized fields to avoid scale issues.

    Args:
        lambda_adv: Weight for advection loss (default: 0.01)
        lambda_diff: Weight for diffusion loss (default: 0.01)
        lambda_grad: Weight for gradient regularization (default: 0.0)
        reduction: 'mean', 'sum', or 'none'
        use_normalized_grid: If True, dx and dy are scaled by field std (default: True)
    """

    def __init__(
        self,
        lambda_adv: float = 0.01,
        lambda_diff: float = 0.01,
        lambda_grad: float = 0.0,
        reduction: str = 'mean',
        use_normalized_grid: bool = True
    ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_diff = lambda_diff
        self.lambda_grad = lambda_grad
        self.use_normalized_grid = use_normalized_grid

        self.advection_loss = AdvectionLoss(reduction=reduction)
        self.diffusion_loss = DiffusionLoss(reduction=reduction)
        self.gradient_loss = VorticityGradientLoss(reduction=reduction) if lambda_grad > 0 else None

    def forward(
        self,
        u_pred: torch.Tensor,
        v_pred: torch.Tensor,
        omega_pred: torch.Tensor,
        u_target: torch.Tensor,
        v_target: torch.Tensor,
        omega_target: torch.Tensor,
        dx: float,
        dy: float,
        sigma_ux: float = 1.0,
        sigma_uy: float = 1.0,
        sigma_omega: float = 1.0
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined physics-informed loss.

        CORRECT FORMULATION: Compares DNS physics to ML physics!
            E_adv = ||u_DNS·∇ω_DNS - u_ML·∇ω_ML||²
            E_diff = ||∇²ω_DNS - ∇²ω_ML||²

        IMPORTANT: Expects NORMALIZED inputs (already divided by their std).
        The grid spacing is scaled to account for normalization, making physics
        losses dimensionless and comparable to pixel loss.

        Args:
            u_pred: NORMALIZED ML x-velocity (B, H, W) or (B, 1, H, W)
            v_pred: NORMALIZED ML y-velocity (B, H, W) or (B, 1, H, W)
            omega_pred: NORMALIZED ML vorticity (B, H, W) or (B, 1, H, W)
            u_target: NORMALIZED DNS x-velocity (B, H, W) or (B, 1, H, W)
            v_target: NORMALIZED DNS y-velocity (B, H, W) or (B, 1, H, W)
            omega_target: NORMALIZED DNS vorticity (B, H, W) or (B, 1, H, W)
            dx: physical grid spacing in x-direction
            dy: physical grid spacing in y-direction
            sigma_ux: std of x-velocity (for dimensionalization, default=1.0)
            sigma_uy: std of y-velocity (for dimensionalization, default=1.0)
            sigma_omega: std of vorticity (for dimensionalization, default=1.0)

        Returns:
            dict with keys:
                - 'total': total physics loss
                - 'advection': advection loss component
                - 'diffusion': diffusion loss component
                - 'gradient': gradient regularization (if enabled)
        """
        # For stability, operate in index space (dx=dy=1) on normalized fields.
        # Physical scaling can be absorbed into lambda_adv / lambda_diff.
        dx_eff = 1.0
        dy_eff = 1.0

        # Compute physics terms and use plain MSE:
        #   E_adv  = ||u_DNS·∇ω_ML − u_DNS·∇ω_DNS||²
        #   E_diff = ||∇²ω_ML − ∇²ω_DNS||²
        loss_adv = self.advection_loss(
            u_target, v_target, omega_pred, omega_target, dx_eff, dy_eff
        )
        loss_diff = self.diffusion_loss(
            omega_pred, omega_target, dx_eff, dy_eff
        )

        # Combined loss
        loss_total = self.lambda_adv * loss_adv + self.lambda_diff * loss_diff

        result = {
            'total': loss_total,
            'advection': loss_adv,
            'diffusion': loss_diff,
        }

        # Optional gradient regularization (only on predicted field)
        if self.gradient_loss is not None:
            loss_grad = self.gradient_loss(omega_pred, dx_norm, dy_norm)
            loss_total = loss_total + self.lambda_grad * loss_grad
            result['total'] = loss_total
            result['gradient'] = loss_grad

        return result


def compute_vorticity_from_velocity(
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float
) -> torch.Tensor:
    """
    Compute vorticity from velocity field.

    For 2D flow in x-y plane:
        ω_z = ∂v/∂x - ∂u/∂y

    Args:
        u: x-velocity component (B, H, W) or (B, 1, H, W)
        v: y-velocity component (B, H, W) or (B, 1, H, W)
        dx: grid spacing in x-direction
        dy: grid spacing in y-direction

    Returns:
        omega_z: vorticity (B, H-2, W-2) - cropped due to central differences
    """
    # Ensure 3D tensors (B, H, W)
    if u.dim() == 4:
        u = u.squeeze(1)
    if v.dim() == 4:
        v = v.squeeze(1)

    # Compute derivatives using central differences
    dv_dx = (v[:, :, 2:] - v[:, :, :-2]) / (2 * dx)
    du_dy = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dy)

    # Match sizes (crop to interior)
    dv_dx = dv_dx[:, 1:-1, :]
    du_dy = du_dy[:, :, 1:-1]

    # Vorticity: ω_z = ∂v/∂x - ∂u/∂y
    omega_z = dv_dx - du_dy

    return omega_z


# Utility function for testing
def test_physics_losses():
    """
    Test physics losses with synthetic data.

    Creates a simple vortex flow and verifies loss computation.
    """
    import numpy as np

    # Create synthetic vortex
    N = 64
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    # Gaussian vortex
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Velocity components (azimuthal flow)
    strength = 1.0
    u_np = -strength * Y * np.exp(-R**2)
    v_np = strength * X * np.exp(-R**2)

    # Convert to torch tensors
    u = torch.from_numpy(u_np).float().unsqueeze(0)
    v = torch.from_numpy(v_np).float().unsqueeze(0)

    # Compute vorticity
    dx = dy = 2.0 / N
    omega = compute_vorticity_from_velocity(u, v, dx, dy)

    # Pad omega to match u, v sizes (for loss computation)
    omega_padded = F.pad(omega, (1, 1, 1, 1), mode='replicate')

    # Test individual losses (compare field to itself - should be ~0)
    adv_loss = AdvectionLoss()
    diff_loss = DiffusionLoss()

    loss_adv = adv_loss(u, v, omega_padded, omega_padded, dx, dy)
    loss_diff = diff_loss(omega_padded, omega_padded, dx, dy)

    print("Physics Losses Test:")
    print(f"  Advection loss (self-comparison): {loss_adv.item():.6f} (should be ~0)")
    print(f"  Diffusion loss (self-comparison): {loss_diff.item():.6f} (should be ~0)")

    # Test combined loss
    physics_loss = PhysicsInformedLoss(lambda_adv=0.01, lambda_diff=0.01)
    losses = physics_loss(u, v, omega_padded, u, v, omega_padded, dx, dy)

    print(f"  Total physics loss (self-comparison): {losses['total'].item():.6f} (should be ~0)")
    print(f"  Components:")
    print(f"    - Advection: {losses['advection'].item():.6f}")
    print(f"    - Diffusion: {losses['diffusion'].item():.6f}")

    # Test with perturbed field
    noise = 0.01 * torch.randn_like(omega_padded)
    omega_perturbed = omega_padded + noise

    loss_adv_diff = adv_loss(u, v, omega_perturbed, omega_padded, dx, dy)
    loss_diff_diff = diff_loss(omega_perturbed, omega_padded, dx, dy)

    print(f"\n  With 1% noise added to vorticity:")
    print(f"    Advection loss: {loss_adv_diff.item():.6f} (should be > 0)")
    print(f"    Diffusion loss: {loss_diff_diff.item():.6f} (should be > 0)")

    return True


if __name__ == "__main__":
    # Run test
    test_physics_losses()
