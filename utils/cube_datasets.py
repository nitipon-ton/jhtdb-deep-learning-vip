"""Datasets for the 3D velocity cubes used by ``superresolution-nn``.

The parent project writes residual training pairs with::

    data/processed/cnn/<split>/input_tXXXX.npy   # (z, y, x, 3) coarse-upsampled
    data/processed/cnn/<split>/target_tXXXX.npy  # (z, y, x, 3) correction = DNS - input

``VelocityCubeDataset`` reads those files directly, converts them to the
channels-first ``(3, z, y, x)`` layout used by ``Conv3d``, and optionally
normalises or augments them.  The arrays are not modified on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

COMPONENT_NAMES: Tuple[str, str, str] = ("u", "v", "w")


def _timestep(path: Path) -> str:
    """Extract ``XXXX`` from a file such as ``input_t0007.npy``."""
    stem = path.stem
    if "_t" not in stem:
        raise ValueError(f"Unexpected cube filename (missing '_t'): {path.name}")
    return stem.rsplit("_t", 1)[1]


def discover_pairs(
    processed_dir: str | Path,
    split: str,
    *,
    input_prefix: str = "input",
    target_prefix: str = "target",
) -> List[Tuple[str, Path, Path]]:
    """Return sorted ``(timestep, input_path, target_path)`` triplets for a split."""
    split_dir = Path(processed_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    inputs = {_timestep(p): p for p in split_dir.glob(f"{input_prefix}_t*.npy")}
    targets = {
        _timestep(p): p for p in split_dir.glob(f"{target_prefix}_t*.npy")
    }
    if not inputs:
        raise FileNotFoundError(
            f"No {input_prefix}_t*.npy files found in {split_dir}. "
            "Run superresolution.preprocess first."
        )

    missing_targets = sorted(set(inputs) - set(targets))
    missing_inputs = sorted(set(targets) - set(inputs))
    if missing_targets or missing_inputs:
        details = []
        if missing_targets:
            details.append(f"missing targets for {missing_targets[:5]}")
        if missing_inputs:
            details.append(f"missing inputs for {missing_inputs[:5]}")
        raise ValueError(
            f"Unpaired cubes in {split_dir}: " + "; ".join(details)
        )

    return [
        (ts, inputs[ts], targets[ts]) for ts in sorted(inputs, key=int)
    ]


class VelocityCubeDataset(Dataset):
    """Load ``(z, y, x, 3)`` velocity-cube pairs from disk.

    Args:
        processed_dir: Directory containing ``train/``, ``val/``, ``test/``.
        split: Split subdirectory name.
        normalize: ``none``, ``per_sample``, or ``global``.  Global mode expects
            ``stats`` with entries for ``input`` and ``target``, each mapping
            component names to ``(mean, std)``; see :func:`compute_component_stats`.
        stats: Normalisation statistics when ``normalize="global"``.
        augment: Apply random axis flips and periodic rolls.  Flips also negate
            the velocity component that points along the reflected axis.
        component_names: Names used to look up per-channel statistics.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        *,
        normalize: str = "none",
        stats: Dict[str, Dict[str, Tuple[float, float]]] | None = None,
        augment: bool = False,
        component_names: Sequence[str] = COMPONENT_NAMES,
    ) -> None:
        if normalize not in {"none", "per_sample", "global"}:
            raise ValueError(
                "normalize must be one of none, per_sample, global; "
                f"got '{normalize}'."
            )
        if len(component_names) != 3:
            raise ValueError("component_names must contain exactly three names.")

        self.processed_dir = Path(processed_dir)
        self.split = split
        self.normalize = normalize
        self.stats = stats
        self.augment = augment
        self.component_names = tuple(component_names)

        self.pairs = discover_pairs(self.processed_dir, self.split)
        self.timesteps = [ts for ts, _, _ in self.pairs]
        self.input_paths = [inp for _, inp, _ in self.pairs]
        self.target_paths = [tgt for _, _, tgt in self.pairs]

        if normalize == "global" and not stats:
            raise ValueError("normalize='global' requires stats.")

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        array = np.load(path)
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(
                f"Expected a (z, y, x, 3) cube at {path}, got {array.shape}."
            )
        tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
        return tensor.float()

    def _normalise(self, x: torch.Tensor, source: str) -> torch.Tensor:
        if self.normalize == "none":
            return x
        if self.normalize == "per_sample":
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
            return (x - mean) / (std + 1e-8)
        if self.stats is None or source not in self.stats:
            raise ValueError(
                f"normalize='global' requires stats['{source}']."
            )
        source_stats = self.stats[source]
        means = torch.tensor(
            [source_stats[name][0] for name in self.component_names],
            dtype=x.dtype,
        ).view(3, 1, 1, 1)
        stds = torch.tensor(
            [source_stats[name][1] for name in self.component_names],
            dtype=x.dtype,
        ).view(3, 1, 1, 1)
        return (x - means) / (stds + 1e-8)

    def _augment_pair(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reflect spatial axes (dims 1, 2, 3 -> z, y, x).  A velocity vector
        # flips sign along the reflected axis: z->w, y->v, x->u.
        component_for_axis = {1: 2, 2: 1, 3: 0}
        for dim, component in component_for_axis.items():
            if torch.rand(1).item() < 0.5:
                inputs = torch.flip(inputs, dims=(dim,))
                targets = torch.flip(targets, dims=(dim,))
                inputs[component] = -inputs[component]
                targets[component] = -targets[component]

        # Random periodic translation; periodicity makes this label-preserving.
        shifts = [
            int(torch.randint(-inputs.size(dim) // 2, inputs.size(dim) // 2 + 1, (1,)))
            for dim in (1, 2, 3)
        ]
        inputs = torch.roll(inputs, shifts=tuple(shifts), dims=(1, 2, 3))
        targets = torch.roll(targets, shifts=tuple(shifts), dims=(1, 2, 3))
        return inputs.contiguous(), targets.contiguous()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, input_path, target_path = self.pairs[index]
        inputs = self._load(input_path)
        targets = self._load(target_path)
        if inputs.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch for timestep {self.timesteps[index]}: "
                f"{tuple(inputs.shape)} vs {tuple(targets.shape)}."
            )

        inputs = self._normalise(inputs, "input")
        targets = self._normalise(targets, "target")

        if self.augment:
            inputs, targets = self._augment_pair(inputs, targets)
        return inputs, targets


def compute_component_stats(
    processed_dir: str | Path,
    split: str = "train",
    *,
    source: str = "input",
    prefix: str | None = None,
) -> Dict[str, Tuple[float, float]]:
    """Compute per-component mean/std for one side of the training pairs."""
    prefix = prefix or source
    split_dir = Path(processed_dir) / split
    files = sorted(split_dir.glob(f"{prefix}_t*.npy"), key=lambda p: int(_timestep(p)))
    if not files:
        raise FileNotFoundError(f"No {prefix}_t*.npy files in {split_dir}.")

    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for path in files:
        array = np.load(path).astype(np.float64, copy=False)
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(
                f"Expected a (z, y, x, 3) cube at {path}, got {array.shape}."
            )
        flattened = array.reshape(-1, 3)
        total += flattened.sum(axis=0)
        total_sq += np.square(flattened).sum(axis=0)
        count += flattened.shape[0]

    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 0.0)
    std = np.sqrt(variance)
    return {
        name: (float(mean[i]), float(std[i]))
        for i, name in enumerate(COMPONENT_NAMES)
    }
