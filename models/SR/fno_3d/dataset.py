"""Read existing superresolution-nn residual-correction training pairs.

    <processed_dir>/{train,val,test}/input_tXXXX.npy
    <processed_dir>/{train,val,test}/target_tXXXX.npy

Each source file is (Z,Y,X,3) float data. The target is DNS - input.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def discover_pairs(root: str | Path, split: str):
    directory = Path(root).expanduser() / split
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing split directory: {directory}')
    a = {p.stem[7:]: p for p in directory.glob('input_t*.npy')}
    b = {p.stem[8:]: p for p in directory.glob('target_t*.npy')}
    if not a or a.keys() != b.keys():
        raise ValueError(f'Missing or unmatched input/target pairs in {directory}: '
                         f'{len(a)} inputs vs {len(b)} targets')
    return [(key, a[key], b[key]) for key in sorted(a, key=lambda k: (int(k) if k.isdigit() else k))]


class VelocityCubePairs(Dataset):
    def __init__(self, root: str | Path, split: str, scale: float = 1.0,
                 patch_size: int | None = None, augmentation: bool = False):
        self.pairs = discover_pairs(root, split)
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.scale = float(scale)
        if self.scale <= 0:
            raise ValueError('data.scale must be positive')
        if self.patch_size is not None and self.patch_size < 4:
            raise ValueError('patch_size must be at least 4 or null')

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _load(path):
        # Copy avoids read-only memmaps / non-writeable numpy warnings.
        arr = np.array(np.load(path, mmap_mode='r'), dtype=np.float32, copy=True)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f'Expected (Z,Y,X,3) at {path}, got {arr.shape}')
        if not np.isfinite(arr).all():
            raise ValueError(f'Nonfinite values in {path}')
        return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()

    def __getitem__(self, idx):
        ts, xp, yp = self.pairs[idx]
        x, y = self._load(xp), self._load(yp)
        if x.shape != y.shape:
            raise ValueError(f'Input/target shape mismatch for {ts}: {x.shape}, {y.shape}')
        if self.patch_size is not None:
            s = self.patch_size
            if any(dim < s for dim in x.shape[-3:]):
                raise ValueError(f'patch_size={s} too big for {x.shape[-3:]}')
            start = [int(torch.randint(0, dim-s+1, (1,))) for dim in x.shape[-3:]]
            slices = tuple(slice(p, p+s) for p in start)
            x, y = x[(slice(None), *slices)], y[(slice(None), *slices)]
        if self.augmentation:
            # Reflections of a physical velocity vector require sign changes.
            # Tensor spatial order is Z,Y,X; channel order is u,v,w.
            for dim, channel in [(1, 2), (2, 1), (3, 0)]:
                if torch.rand(()) < 0.5:
                    x, y = torch.flip(x, (dim,)), torch.flip(y, (dim,))
                    x[channel] *= -1
                    y[channel] *= -1
        return x / self.scale, y / self.scale
