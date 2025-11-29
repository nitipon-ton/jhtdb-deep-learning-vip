"""
Dataset utilities for turbulence super-resolution.

Implements TemporalStackDataset with physics-aware augmentations following
the channel flow SR implementation guide.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple, List


class TemporalStackDataset(Dataset):
    """
    Dataset for turbulence SR with optional temporal stacking (MTPC).

    Key features:
    - Snapshot-wise organization (no temporal leakage)
    - Automatic LR creation via average pooling
    - Support for N-frame temporal stacking
    - Physics-aware augmentations with sign maps
    - Global or per-snapshot normalization

    Args:
        HR_seq: High-resolution sequence (T, H, W)
        N: Number of temporal frames to stack (1, 3, or 5)
        dt: Spacing between frames
        scale: Downsampling factor (HR_size / LR_size)
        normalize: "global" or "per_snapshot"
        mu: Global mean (if normalize="global")
        sigma: Global std (if normalize="global")
        component: Component name for sign map ("ux", "uy", "uz", "omega_z")
        transform_config: Augmentation configuration dictionary
    """

    def __init__(
        self,
        HR_seq: np.ndarray,
        N: int = 1,
        dt: int = 1,
        scale: int = 4,
        normalize: str = "global",
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
        component: str = "omega_z",
        transform_config: Optional[Dict] = None,
    ):
        self.HR = torch.from_numpy(HR_seq).float()  # (T, H, W)
        self.N = N
        self.dt = dt
        self.scale = scale
        self.norm = normalize
        self.mu = mu
        self.sigma = sigma
        self.component = component
        self.transform_config = transform_config or {}

        # Valid indices (ensure we have N past frames)
        self.T = len(HR_seq)
        self.valid_indices = list(range((N - 1) * dt, self.T))

        # Precompute LR sequence
        self.LR = self._make_lr_sequence(self.HR)

    def _make_lr_sequence(self, hr_seq: torch.Tensor) -> torch.Tensor:
        """Create LR sequence via average pooling."""
        # Add batch and channel dims: (T, H, W) -> (T, 1, H, W)
        hr_batch = hr_seq.unsqueeze(1)
        lr_batch = F.avg_pool2d(hr_batch, kernel_size=self.scale, stride=self.scale)
        return lr_batch.squeeze(1)  # (T, H_lr, W_lr)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        if self.norm == "global":
            if self.mu is None or self.sigma is None:
                raise ValueError("Global normalization requires mu and sigma")
            return (x - self.mu) / (self.sigma + 1e-8)
        else:  # per_snapshot
            mu = x.mean(dim=(-2, -1), keepdim=True)
            sigma = x.std(dim=(-2, -1), keepdim=True)
            return (x - mu) / (sigma + 1e-8)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample with optional temporal stacking.

        Returns:
            lr_stack: (N, H_lr, W_lr) stacked LR frames
            hr_target: (H_hr, W_hr) HR target at time t
        """
        t = self.valid_indices[idx]

        # Get temporal indices: [t-(N-1)*dt, ..., t-dt, t]
        times = [t - k * self.dt for k in range(self.N - 1, -1, -1)]

        # Stack LR frames
        lr_frames = [self.LR[ti] for ti in times]
        lr_stack = torch.stack(lr_frames, dim=0)  # (N, H_lr, W_lr)

        # Get HR target
        hr_target = self.HR[t]  # (H_hr, W_hr)

        # Normalize
        lr_stack = self._normalize(lr_stack)
        hr_target = self._normalize(hr_target.unsqueeze(0)).squeeze(0)

        # Apply augmentations
        if self.transform_config.get("enabled", False):
            lr_stack, hr_target = self._apply_augmentations(lr_stack, hr_target)

        return lr_stack, hr_target

    def _apply_augmentations(
        self, lr_stack: torch.Tensor, hr_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply physics-aware augmentations to LR stack and HR target.

        Transformations are applied identically to both LR and HR.
        """
        cfg = self.transform_config

        # Random 180-degree rotation
        if cfg.get("rot180", False) and torch.rand(1) < 0.5:
            lr_stack = torch.flip(lr_stack, (-1, -2))
            hr_target = torch.flip(hr_target, (-1, -2))

        # Random horizontal flip (with sign map)
        if cfg.get("flipx", False) and torch.rand(1) < 0.5:
            lr_stack = torch.flip(lr_stack, (-1,))
            hr_target = torch.flip(hr_target, (-1,))
            # Apply sign map
            sign_map = cfg.get("sign_maps", {}).get(self.component, {})
            sign_x = sign_map.get("flipx", 1)
            lr_stack = lr_stack * sign_x
            hr_target = hr_target * sign_x

        # Random vertical flip (with sign map)
        if cfg.get("flipy", False) and torch.rand(1) < 0.5:
            lr_stack = torch.flip(lr_stack, (-2,))
            hr_target = torch.flip(hr_target, (-2,))
            # Apply sign map
            sign_map = cfg.get("sign_maps", {}).get(self.component, {})
            sign_y = sign_map.get("flipy", 1)
            lr_stack = lr_stack * sign_y
            hr_target = hr_target * sign_y

        # Random cyclic shift (periodic boundaries)
        if cfg.get("cyclic_shift", False):
            shift_x = torch.randint(-lr_stack.shape[-1] // 4, lr_stack.shape[-1] // 4, (1,)).item()
            shift_y = torch.randint(-lr_stack.shape[-2] // 4, lr_stack.shape[-2] // 4, (1,)).item()
            lr_stack = torch.roll(lr_stack, shifts=(shift_y, shift_x), dims=(-2, -1))
            # Scale shift for HR
            hr_shift_x = shift_x * self.scale
            hr_shift_y = shift_y * self.scale
            hr_target = torch.roll(hr_target, shifts=(hr_shift_y, hr_shift_x), dims=(-2, -1))

        # Optional noise injection (LR only)
        if cfg.get("noise", {}).get("enabled", False):
            noise_std = cfg["noise"].get("std", 0.01)
            lr_stack = lr_stack + torch.randn_like(lr_stack) * noise_std

        return lr_stack, hr_target


def create_snapshot_splits(
    total_snapshots: int,
    train_snapshots: int,
    val_snapshots: int,
    test_snapshots: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create snapshot-wise splits (no temporal leakage).

    Args:
        total_snapshots: Total number of snapshots
        train_snapshots: Number of training snapshots
        val_snapshots: Number of validation snapshots
        test_snapshots: Number of test snapshots

    Returns:
        train_indices, val_indices, test_indices
    """
    if train_snapshots + val_snapshots + test_snapshots > total_snapshots:
        raise ValueError("Split sizes exceed total snapshots")

    train_indices = list(range(0, train_snapshots))
    val_indices = list(range(train_snapshots, train_snapshots + val_snapshots))
    test_indices = list(range(
        train_snapshots + val_snapshots,
        train_snapshots + val_snapshots + test_snapshots
    ))

    return train_indices, val_indices, test_indices


def compute_global_stats(hr_data: np.ndarray, train_indices: List[int]) -> Tuple[float, float]:
    """
    Compute global mean and std over training snapshots only.

    Args:
        hr_data: Full HR data (T, H, W)
        train_indices: Indices of training snapshots

    Returns:
        mu, sigma
    """
    train_data = hr_data[train_indices]
    mu = float(train_data.mean())
    sigma = float(train_data.std())
    return mu, sigma


class MultiFieldDataset(Dataset):
    """
    Dataset for physics-informed SR that returns ALL velocity components.

    Unlike TemporalStackDataset which returns only one component, this dataset
    returns ux, uy, and omega_z for computing physics-informed losses.

    Key features:
    - Returns all fields: ux, uy, omega_z at both LR and HR resolutions
    - Snapshot-wise organization (no temporal leakage)
    - Automatic LR creation via average pooling
    - Support for N-frame temporal stacking
    - Physics-aware augmentations with sign maps
    - Global or per-snapshot normalization

    Args:
        ux_seq: x-velocity component (T, H, W)
        uy_seq: y-velocity component (T, H, W)
        omega_seq: vorticity component (T, H, W)
        N: Number of temporal frames to stack (1, 3, or 5)
        dt: Spacing between frames
        scale: Downsampling factor (HR_size / LR_size)
        normalize: "global" or "per_snapshot"
        stats: Dict with normalization stats per component (if global normalization)
        transform_config: Augmentation configuration dictionary
    """

    def __init__(
        self,
        ux_seq: np.ndarray,
        uy_seq: np.ndarray,
        omega_seq: np.ndarray,
        N: int = 1,
        dt: int = 1,
        scale: int = 4,
        normalize: str = "global",
        stats: Optional[Dict[str, Tuple[float, float]]] = None,
        transform_config: Optional[Dict] = None,
    ):
        # Store HR sequences
        self.ux_HR = torch.from_numpy(ux_seq).float()  # (T, H, W)
        self.uy_HR = torch.from_numpy(uy_seq).float()  # (T, H, W)
        self.omega_HR = torch.from_numpy(omega_seq).float()  # (T, H, W)

        self.N = N
        self.dt = dt
        self.scale = scale
        self.norm = normalize
        self.stats = stats or {}
        self.transform_config = transform_config or {}

        # Valid indices (ensure we have N past frames)
        self.T = len(ux_seq)
        self.valid_indices = list(range((N - 1) * dt, self.T))

        # Precompute LR sequences for all components
        self.ux_LR = self._make_lr_sequence(self.ux_HR)
        self.uy_LR = self._make_lr_sequence(self.uy_HR)
        self.omega_LR = self._make_lr_sequence(self.omega_HR)

    def _make_lr_sequence(self, hr_seq: torch.Tensor) -> torch.Tensor:
        """Create LR sequence via average pooling."""
        hr_batch = hr_seq.unsqueeze(1)  # (T, 1, H, W)
        lr_batch = F.avg_pool2d(hr_batch, kernel_size=self.scale, stride=self.scale)
        return lr_batch.squeeze(1)  # (T, H_lr, W_lr)

    def _normalize(self, x: torch.Tensor, component: str) -> torch.Tensor:
        """Apply normalization to a specific component."""
        if self.norm == "global":
            if component not in self.stats:
                raise ValueError(f"Global normalization requires stats for {component}")
            mu, sigma = self.stats[component]
            return (x - mu) / (sigma + 1e-8)
        else:  # per_snapshot
            mu = x.mean(dim=(-2, -1), keepdim=True)
            sigma = x.std(dim=(-2, -1), keepdim=True)
            return (x - mu) / (sigma + 1e-8)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample with ALL fields.

        Returns:
            dict with keys:
                - 'lr_omega': LR vorticity for SR input (N, H_lr, W_lr)
                - 'hr_omega': HR vorticity target (H_hr, W_hr)
                - 'hr_ux': HR x-velocity for physics loss (H_hr, W_hr)
                - 'hr_uy': HR y-velocity for physics loss (H_hr, W_hr)
                - 'lr_ux': LR x-velocity (optional, for consistency)
                - 'lr_uy': LR y-velocity (optional, for consistency)
        """
        t = self.valid_indices[idx]

        # Get temporal indices: [t-(N-1)*dt, ..., t-dt, t]
        times = [t - k * self.dt for k in range(self.N - 1, -1, -1)]

        # Stack LR frames for omega (input to SR model)
        lr_omega_frames = [self.omega_LR[ti] for ti in times]
        lr_omega_stack = torch.stack(lr_omega_frames, dim=0)  # (N, H_lr, W_lr)

        # Get HR targets at time t
        hr_omega = self.omega_HR[t]  # (H_hr, W_hr)
        hr_ux = self.ux_HR[t]  # (H_hr, W_hr)
        hr_uy = self.uy_HR[t]  # (H_hr, W_hr)

        # Also get LR velocity (optional, for completeness)
        lr_ux = self.ux_LR[t]  # (H_lr, W_lr)
        lr_uy = self.uy_LR[t]  # (H_lr, W_lr)

        # Normalize all components
        lr_omega_stack = self._normalize(lr_omega_stack, 'omega_z')
        hr_omega = self._normalize(hr_omega.unsqueeze(0), 'omega_z').squeeze(0)
        hr_ux = self._normalize(hr_ux.unsqueeze(0), 'ux').squeeze(0)
        hr_uy = self._normalize(hr_uy.unsqueeze(0), 'uy').squeeze(0)
        lr_ux = self._normalize(lr_ux.unsqueeze(0), 'ux').squeeze(0)
        lr_uy = self._normalize(lr_uy.unsqueeze(0), 'uy').squeeze(0)

        # Apply augmentations (TODO: implement multi-field augmentations)
        # For now, skip augmentations to keep it simple

        return {
            'lr_omega': lr_omega_stack,  # Input to SR model
            'hr_omega': hr_omega,         # Target for SR
            'hr_ux': hr_ux,              # For physics loss
            'hr_uy': hr_uy,              # For physics loss
            'lr_ux': lr_ux,              # Optional
            'lr_uy': lr_uy,              # Optional
        }


class RandomCropSRDataset(Dataset):
    """
    Dataset that samples random HR patches (e.g., 128×128) from large snapshots
    (e.g., 512×512) and creates LR patches on the fly via average pooling.
    """

    def __init__(
        self,
        omega_seq: np.ndarray,
        indices: List[int],
        patch_size_hr: int = 128,
        scale: int = 4,
        num_crops_per_snapshot: int = 4,
        normalize: str = "global",
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
        stats: Optional[Dict[str, Tuple[float, float]]] = None,
        ux_seq: Optional[np.ndarray] = None,
        uy_seq: Optional[np.ndarray] = None,
        transform_config: Optional[Dict] = None,
    ):
        self.omega = torch.from_numpy(omega_seq).float()
        self.indices = indices
        self.patch = patch_size_hr
        self.scale = scale
        self.num_crops = num_crops_per_snapshot
        self.normalize = normalize
        self.mu = mu
        self.sigma = sigma
        self.stats = stats or {}
        self.ux = torch.from_numpy(ux_seq).float() if ux_seq is not None else None
        self.uy = torch.from_numpy(uy_seq).float() if uy_seq is not None else None
        self.transform_config = transform_config or {}

        if normalize == "global" and (mu is None or sigma is None):
            raise ValueError("Global normalization requires mu and sigma")

    def __len__(self) -> int:
        return len(self.indices) * self.num_crops

    def _rand_coords(self, H: int, W: int) -> Tuple[int, int]:
        step = self.scale
        max_h = (H - self.patch) // step
        max_w = (W - self.patch) // step
        y0 = int(torch.randint(0, max_h + 1, (1,)).item()) * step
        x0 = int(torch.randint(0, max_w + 1, (1,)).item()) * step
        return y0, x0

    def _norm_scalar(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize == "global":
            return (x - self.mu) / (self.sigma + 1e-8)
        mu = x.mean()
        sigma = x.std()
        return (x - mu) / (sigma + 1e-8)

    def _norm_component(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if self.normalize == "global":
            if name not in self.stats:
                raise ValueError(f"Missing stats for component {name}")
            mu, sigma = self.stats[name]
            return (x - mu) / (sigma + 1e-8)
        mu = x.mean()
        sigma = x.std()
        return (x - mu) / (sigma + 1e-8)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        snap_idx = self.indices[idx // self.num_crops]
        field = self.omega[snap_idx]
        H, W = field.shape
        y0, x0 = self._rand_coords(H, W)
        hr_patch = field[y0:y0+self.patch, x0:x0+self.patch]
        lr_patch = F.avg_pool2d(hr_patch.unsqueeze(0).unsqueeze(0), kernel_size=self.scale, stride=self.scale).squeeze(0)

        sample = {
            'lr_omega': self._norm_scalar(lr_patch),
            'hr_omega': self._norm_scalar(hr_patch),
        }

        if self.ux is not None and self.uy is not None:
            ux_patch = self.ux[snap_idx, y0:y0+self.patch, x0:x0+self.patch]
            uy_patch = self.uy[snap_idx, y0:y0+self.patch, x0:x0+self.patch]
            sample['hr_ux'] = self._norm_component(ux_patch, 'ux')
            sample['hr_uy'] = self._norm_component(uy_patch, 'uy')

        return sample


class TiledSRDataset(Dataset):
    """
    Deterministic tiling dataset (e.g., for validation/test) using fixed 128×128 patches.
    """

    def __init__(
        self,
        omega_seq: np.ndarray,
        indices: List[int],
        patch_size_hr: int = 128,
        scale: int = 4,
        normalize: str = "global",
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
        stats: Optional[Dict[str, Tuple[float, float]]] = None,
        ux_seq: Optional[np.ndarray] = None,
        uy_seq: Optional[np.ndarray] = None,
    ):
        self.omega = torch.from_numpy(omega_seq).float()
        self.indices = indices
        self.patch = patch_size_hr
        self.scale = scale
        self.normalize = normalize
        self.mu = mu
        self.sigma = sigma
        self.stats = stats or {}
        self.ux = torch.from_numpy(ux_seq).float() if ux_seq is not None else None
        self.uy = torch.from_numpy(uy_seq).float() if uy_seq is not None else None

        tile_positions = []
        tiles_per_dim = omega_seq.shape[-1] // patch_size_hr
        for i in range(tiles_per_dim):
            for j in range(tiles_per_dim):
                if (i + j) % 2 == 0:
                    y = i * patch_size_hr
                    x = j * patch_size_hr
                    tile_positions.append((y, x))
        self.tile_positions = tile_positions

    def __len__(self) -> int:
        return len(self.indices) * len(self.tile_positions)

    def _norm_scalar(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize == "global":
            return (x - self.mu) / (self.sigma + 1e-8)
        mu = x.mean()
        sigma = x.std()
        return (x - mu) / (sigma + 1e-8)

    def _norm_component(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if self.normalize == "global":
            mu, sigma = self.stats[name]
            return (x - mu) / (sigma + 1e-8)
        mu = x.mean()
        sigma = x.std()
        return (x - mu) / (sigma + 1e-8)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tile_per_snapshot = len(self.tile_positions)
        snap_idx = self.indices[idx // tile_per_snapshot]
        tile_idx = idx % tile_per_snapshot
        y0, x0 = self.tile_positions[tile_idx]
        hr_patch = self.omega[snap_idx, y0:y0+self.patch, x0:x0+self.patch]
        lr_patch = F.avg_pool2d(hr_patch.unsqueeze(0).unsqueeze(0), kernel_size=self.scale, stride=self.scale).squeeze(0)

        sample = {
            'lr_omega': self._norm_scalar(lr_patch),
            'hr_omega': self._norm_scalar(hr_patch),
        }

        if self.ux is not None and self.uy is not None:
            ux_patch = self.ux[snap_idx, y0:y0+self.patch, x0:x0+self.patch]
            uy_patch = self.uy[snap_idx, y0:y0+self.patch, x0:x0+self.patch]
            sample['hr_ux'] = self._norm_component(ux_patch, 'ux')
            sample['hr_uy'] = self._norm_component(uy_patch, 'uy')

        return sample


def compute_multifield_stats(
    ux_data: np.ndarray,
    uy_data: np.ndarray,
    omega_data: np.ndarray,
    train_indices: List[int]
) -> Dict[str, Tuple[float, float]]:
    """
    Compute global normalization stats for all components.

    Args:
        ux_data: x-velocity data (T, H, W)
        uy_data: y-velocity data (T, H, W)
        omega_data: vorticity data (T, H, W)
        train_indices: Indices of training snapshots

    Returns:
        dict mapping component name to (mu, sigma)
    """
    stats = {}

    for name, data in [('ux', ux_data), ('uy', uy_data), ('omega_z', omega_data)]:
        train_data = data[train_indices]
        mu = float(train_data.mean())
        sigma = float(train_data.std())
        stats[name] = (mu, sigma)

    return stats
