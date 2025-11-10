"""
Physics utilities for fluid dynamics analysis.

Provides functions for computing derived quantities like vorticity, Q-criterion,
strain rate, and other flow diagnostics.
"""

from __future__ import annotations

import numpy as np


def compute_vorticity_2d(
    ux: np.ndarray,
    uy: np.ndarray,
    dx: float | None = None,
    dy: float | None = None,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute 2D vorticity (ω_z) from 2D velocity field.

    For a 2D flow in the x-y plane, vorticity is computed as:
        ω_z = ∂v/∂x - ∂u/∂y

    Args:
        ux: x-component of velocity, shape (Nx, Ny)
        uy: y-component of velocity, shape (Nx, Ny)
        dx: Grid spacing in x (if uniform). Computed from x_coords if not provided.
        dy: Grid spacing in y (if uniform). Computed from y_coords if not provided.
        x_coords: 1D array of x coordinates (if non-uniform grid)
        y_coords: 1D array of y coordinates (if non-uniform grid)

    Returns:
        Vorticity field ω_z of shape (Nx, Ny)

    Example:
        >>> ux = np.random.rand(64, 64)
        >>> uy = np.random.rand(64, 64)
        >>> omega_z = compute_vorticity_2d(ux, uy, dx=0.1, dy=0.1)
    """
    # Determine grid spacing
    if dx is None:
        if x_coords is None:
            raise ValueError("Either dx or x_coords must be provided")
        dx = np.mean(np.diff(x_coords))

    if dy is None:
        if y_coords is None:
            raise ValueError("Either dy or y_coords must be provided")
        dy = np.mean(np.diff(y_coords))

    # Compute derivatives using centered finite differences
    # duy/dx (derivative of uy with respect to x)
    duy_dx = np.gradient(uy, dx, axis=0)

    # dux/dy (derivative of ux with respect to y)
    dux_dy = np.gradient(ux, dy, axis=1)

    # Vorticity: ω_z = ∂v/∂x - ∂u/∂y
    omega_z = duy_dx - dux_dy

    return omega_z


def compute_vorticity_3d(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D vorticity vector from 3D velocity field.

    Vorticity is computed as ω = ∇ × u:
        ω_x = ∂w/∂y - ∂v/∂z
        ω_y = ∂u/∂z - ∂w/∂x
        ω_z = ∂v/∂x - ∂u/∂y

    Args:
        ux: x-component of velocity, shape (Nx, Ny, Nz)
        uy: y-component of velocity, shape (Nx, Ny, Nz)
        uz: z-component of velocity, shape (Nx, Ny, Nz)
        dx: Grid spacing in x
        dy: Grid spacing in y
        dz: Grid spacing in z

    Returns:
        Tuple of (ω_x, ω_y, ω_z) each of shape (Nx, Ny, Nz)
    """
    # Compute gradients
    dux_dy = np.gradient(ux, dy, axis=1)
    dux_dz = np.gradient(ux, dz, axis=2)

    duy_dx = np.gradient(uy, dx, axis=0)
    duy_dz = np.gradient(uy, dz, axis=2)

    duz_dx = np.gradient(uz, dx, axis=0)
    duz_dy = np.gradient(uz, dy, axis=1)

    # Compute vorticity components
    omega_x = duz_dy - duy_dz
    omega_y = dux_dz - duz_dx
    omega_z = duy_dx - dux_dy

    return omega_x, omega_y, omega_z


def compute_vorticity_magnitude(
    omega_x: np.ndarray,
    omega_y: np.ndarray | None = None,
    omega_z: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute magnitude of vorticity vector.

    Args:
        omega_x: x-component of vorticity (or 2D vorticity)
        omega_y: y-component of vorticity (optional, for 3D)
        omega_z: z-component of vorticity (optional, for 3D)

    Returns:
        Vorticity magnitude |ω|
    """
    if omega_y is None and omega_z is None:
        # 2D case: just return absolute value
        return np.abs(omega_x)
    elif omega_y is not None and omega_z is not None:
        # 3D case: compute vector magnitude
        return np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    else:
        raise ValueError("Provide either just omega_x (2D) or all three components (3D)")


def compute_strain_rate_2d(
    ux: np.ndarray,
    uy: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Compute 2D strain rate magnitude.

    Strain rate tensor S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)

    For 2D flow:
        S_11 = ∂u/∂x
        S_22 = ∂v/∂y
        S_12 = 0.5 * (∂u/∂y + ∂v/∂x)

    Magnitude: |S| = sqrt(2 * S_ij * S_ij)

    Args:
        ux: x-component of velocity, shape (Nx, Ny)
        uy: y-component of velocity, shape (Nx, Ny)
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        Strain rate magnitude of shape (Nx, Ny)
    """
    # Compute velocity gradients
    dux_dx = np.gradient(ux, dx, axis=0)
    dux_dy = np.gradient(ux, dy, axis=1)
    duy_dx = np.gradient(uy, dx, axis=0)
    duy_dy = np.gradient(uy, dy, axis=1)

    # Strain rate tensor components
    S11 = dux_dx
    S22 = duy_dy
    S12 = 0.5 * (dux_dy + duy_dx)

    # Magnitude
    magnitude = np.sqrt(2 * (S11**2 + S22**2 + 2 * S12**2))

    return magnitude


def compute_q_criterion_2d(
    ux: np.ndarray,
    uy: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Compute Q-criterion for 2D flow.

    Q-criterion identifies vortical structures: Q = (Ω² - S²) / 2
    where Ω is the vorticity magnitude and S is the strain rate magnitude.

    Positive Q indicates regions where rotation dominates over strain.

    Args:
        ux: x-component of velocity, shape (Nx, Ny)
        uy: y-component of velocity, shape (Nx, Ny)
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        Q-criterion field of shape (Nx, Ny)
    """
    # Compute vorticity
    omega_z = compute_vorticity_2d(ux, uy, dx, dy)

    # Compute strain rate
    strain = compute_strain_rate_2d(ux, uy, dx, dy)

    # Q-criterion
    Q = 0.5 * (omega_z**2 - strain**2)

    return Q


def compute_enstrophy(omega: np.ndarray) -> float:
    """
    Compute enstrophy (integrated vorticity squared).

    Enstrophy = ∫ ω² dV

    This is a measure of the total rotation in the flow.

    Args:
        omega: Vorticity field (any shape)

    Returns:
        Enstrophy value
    """
    return float(np.mean(omega**2))


def compute_divergence_2d(
    ux: np.ndarray,
    uy: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Compute divergence of 2D velocity field.

    ∇·u = ∂u/∂x + ∂v/∂y

    For incompressible flow, divergence should be zero.

    Args:
        ux: x-component of velocity, shape (Nx, Ny)
        uy: y-component of velocity, shape (Nx, Ny)
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        Divergence field of shape (Nx, Ny)
    """
    dux_dx = np.gradient(ux, dx, axis=0)
    duy_dy = np.gradient(uy, dy, axis=1)

    divergence = dux_dx + duy_dy

    return divergence


def add_vorticity_to_plane(
    plane_data: dict,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    dx: float | None = None,
    dy: float | None = None,
) -> dict:
    """
    Add vorticity field to a plane data dictionary.

    This is a convenience function for augmenting JHTDB plane data with
    computed vorticity.

    Args:
        plane_data: Dictionary with 'data' and 'components' keys
        x_coords: Coordinates along primary axis
        y_coords: Coordinates along secondary axis
        dx: Grid spacing along primary axis
        dy: Grid spacing along secondary axis

    Returns:
        Updated plane_data dictionary with vorticity added

    Example:
        >>> plane_data = {
        ...     'data': np.random.rand(3, 64, 64),  # (ux, uy, uz)
        ...     'components': ['ux', 'uy', 'uz'],
        ... }
        >>> plane_data = add_vorticity_to_plane(plane_data, dx=0.1, dy=0.1)
        >>> print(plane_data['components'])
        ['ux', 'uy', 'uz', 'omega_z']
    """
    data = plane_data["data"]
    components = plane_data["components"]

    # Find velocity components
    ux_idx = None
    uy_idx = None

    for i, comp in enumerate(components):
        if "ux" in comp.lower() or comp.lower() == "u":
            ux_idx = i
        elif "uy" in comp.lower() or comp.lower() == "v":
            uy_idx = i

    if ux_idx is None or uy_idx is None:
        raise ValueError(
            f"Could not find velocity components in {components}. "
            "Expected 'ux' and 'uy' or similar."
        )

    # Extract velocity fields
    ux = data[ux_idx]
    uy = data[uy_idx]

    # Compute vorticity
    omega_z = compute_vorticity_2d(ux, uy, dx, dy, x_coords, y_coords)

    # Add to data
    data_augmented = np.concatenate([data, omega_z[None, :, :]], axis=0)

    plane_data_updated = plane_data.copy()
    plane_data_updated["data"] = data_augmented
    plane_data_updated["components"] = components + ["omega_z"]

    return plane_data_updated


# Utility functions for flow statistics

def compute_reynolds_stress_2d(
    ux: np.ndarray,
    uy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Reynolds stress components for 2D flow.

    Reynolds stress: -<u'_i u'_j> where u' is fluctuation from mean.

    Args:
        ux: x-velocity field (can be ensemble or time series)
        uy: y-velocity field

    Returns:
        Tuple of (R_xx, R_yy, R_xy) Reynolds stress components
    """
    ux_mean = np.mean(ux)
    uy_mean = np.mean(uy)

    ux_prime = ux - ux_mean
    uy_prime = uy - uy_mean

    R_xx = np.mean(ux_prime * ux_prime)
    R_yy = np.mean(uy_prime * uy_prime)
    R_xy = np.mean(ux_prime * uy_prime)

    return R_xx, R_yy, R_xy


def compute_tke_2d(ux: np.ndarray, uy: np.ndarray) -> float:
    """
    Compute turbulent kinetic energy for 2D flow.

    TKE = 0.5 * <u'² + v'²>

    Args:
        ux: x-velocity field
        uy: y-velocity field

    Returns:
        Turbulent kinetic energy
    """
    R_xx, R_yy, _ = compute_reynolds_stress_2d(ux, uy)
    tke = 0.5 * (R_xx + R_yy)
    return float(tke)
