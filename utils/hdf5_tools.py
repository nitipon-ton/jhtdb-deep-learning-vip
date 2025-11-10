from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import h5py
import numpy as np


def write_sr_h5(
    path: str | Path,
    lr: np.ndarray,
    hr: np.ndarray,
    splits: Mapping[str, float],
    attrs: Dict[str, Any],
) -> None:
    """Persist LR/HR arrays into an HDF5 file with train/val splits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = lr.shape[0]
    order = np.arange(n)
    split_idx = int(n * splits.get("train", 0.8))
    split_map = {
        "train": order[:split_idx],
        "val": order[split_idx:],
    }

    with h5py.File(path, "w") as handle:
        for split, indices in split_map.items():
            grp = handle.create_group(split)
            grp.create_dataset("x", data=lr[indices], compression="gzip")
            grp.create_dataset("y", data=hr[indices], compression="gzip")
            grp.attrs["count"] = len(indices)
        handle.attrs.update({k: serialize_attr(v) for k, v in attrs.items()})


def serialize_attr(value: Any) -> Any:
    """Make sure attributes are JSON serializable before writing to HDF5."""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    return str(value)


def load_split(path: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Convenience loader that returns LR/HR arrays from an HDF5 split."""
    with h5py.File(path, "r") as handle:
        grp = handle[split]
        lr = np.asarray(grp["x"])
        hr = np.asarray(grp["y"])
    return lr, hr

