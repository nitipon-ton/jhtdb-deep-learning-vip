"""
Utility helpers for organising experiment work directories.

Each experiment gets its own folder under ``work/{task}/{name}_{timestamp}``
with sub-directories for checkpoints, predictions, metrics, and plots.  The
manager keeps track of the paths and provides small helper methods to save
artefacts from notebooks without repeating boilerplate.

Features:
- Versioned training history (training_history_v1.json, v2.json, ...)
- Organized plot subdirectories
- History merging for resumed training
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
import json
import shutil
import re

import yaml


class ExperimentManager:
    def __init__(
        self,
        task: str,
        name: str,
        base_dir: Optional[Path] = None,
        run_name: Optional[str] = None,
    ):
        self.task = task
        self.name = name
        self.base_dir = Path(base_dir) if base_dir else Path("work")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if run_name:
            self.run_dir = self.base_dir / task / run_name
        else:
            self.run_dir = self.base_dir / task / f"{name}_{timestamp}"

        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.predictions_dir = self.run_dir / "predictions"
        self.metrics_dir = self.run_dir / "metrics"
        self.plots_dir = self.run_dir / "plots"
        self.history_dir = self.run_dir / "history"
        self.logs_dir = self.run_dir / "logs"

        # Organized plot subdirectories
        self.plots_data_dir = self.plots_dir / "01_data_inspection"
        self.plots_training_dir = self.plots_dir / "02_training"
        self.plots_validation_dir = self.plots_dir / "03_validation"
        self.plots_physics_dir = self.plots_dir / "04_physics"
        self.plots_test_dir = self.run_dir / "plots_test"

    def setup(self) -> None:
        """Create all experiment directories."""
        for path in [
            self.run_dir,
            self.checkpoints_dir,
            self.predictions_dir,
            self.metrics_dir,
            self.plots_dir,
            self.history_dir,
            self.logs_dir,
            self.plots_data_dir,
            self.plots_training_dir,
            self.plots_validation_dir,
            self.plots_physics_dir,
            self.plots_test_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Dict[str, Any], file_name: str = "config.yaml") -> None:
        with open(self.run_dir / file_name, "w") as fp:
            yaml.safe_dump(config, fp, sort_keys=False)

    def save_metrics(self, metrics: Dict[str, Any], split: str = "val") -> Path:
        """Save evaluation metrics to JSON."""
        path = self.metrics_dir / f"{split}_metrics.json"
        with open(path, "w") as fp:
            json.dump(metrics, fp, indent=2)
        return path

    def save_history(
        self,
        history: Dict[str, Any],
        version: Union[str, int] = 'auto',
        legacy: bool = False
    ) -> Path:
        """
        Save training history with automatic versioning.

        Args:
            history: Training history dictionary
            version: Version number (int) or 'auto' for automatic increment
            legacy: If True, save to old location (run_dir/training_history.json)

        Returns:
            Path to saved history file

        Examples:
            manager.save_history(history, version='auto')  # → history/training_history_v1.json
            manager.save_history(history, version=2)        # → history/training_history_v2.json
            manager.save_history(history, legacy=True)      # → training_history.json (old format)
        """
        if legacy:
            # Legacy format for backward compatibility
            path = self.run_dir / "training_history.json"
        else:
            if version == 'auto':
                # Find next available version number
                version = self._get_next_history_version()

            path = self.history_dir / f"training_history_v{version}.json"

        with open(path, "w") as fp:
            json.dump(history, fp, indent=2)

        return path

    def _get_next_history_version(self) -> int:
        """Get next available version number for training history."""
        if not self.history_dir.exists():
            return 1

        existing = list(self.history_dir.glob("training_history_v*.json"))
        if not existing:
            return 1

        versions = []
        for f in existing:
            match = re.search(r'v(\d+)\.json$', f.name)
            if match:
                versions.append(int(match.group(1)))

        return max(versions) + 1 if versions else 1

    def load_history(self, version: Union[str, int, Path, None] = 'latest') -> Dict[str, Any]:
        """
        Load training history from file.

        Args:
            version: Version to load:
                - 'latest': Load most recent version
                - int: Load specific version number
                - Path: Load from specific path
                - None: Return empty history

        Returns:
            Training history dictionary

        Examples:
            history = manager.load_history('latest')
            history = manager.load_history(2)
            history = manager.load_history(Path('work/SR/run1/history/training_history_v1.json'))
        """
        if version is None:
            return self._empty_history()

        if isinstance(version, Path):
            path = version
        elif version == 'latest':
            path = self._get_latest_history_path()
            if path is None:
                return self._empty_history()
        else:
            path = self.history_dir / f"training_history_v{version}.json"

        if not path.exists():
            print(f"Warning: History file not found: {path}")
            return self._empty_history()

        with open(path, 'r') as fp:
            history = json.load(fp)

        return history

    def _get_latest_history_path(self) -> Optional[Path]:
        """Get path to latest training history file."""
        # Check new location first
        if self.history_dir.exists():
            existing = list(self.history_dir.glob("training_history_v*.json"))
            if existing:
                versions = []
                for f in existing:
                    match = re.search(r'v(\d+)\.json$', f.name)
                    if match:
                        versions.append((int(match.group(1)), f))
                if versions:
                    return max(versions, key=lambda x: x[0])[1]

        # Check legacy location
        legacy_path = self.run_dir / "training_history.json"
        if legacy_path.exists():
            return legacy_path

        return None

    def list_history_versions(self) -> List[int]:
        """List all available training history versions."""
        if not self.history_dir.exists():
            return []

        versions = []
        for f in self.history_dir.glob("training_history_v*.json"):
            match = re.search(r'v(\d+)\.json$', f.name)
            if match:
                versions.append(int(match.group(1)))

        return sorted(versions)

    def merge_histories(self, *histories: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple training histories chronologically.

        Args:
            *histories: Variable number of history dictionaries to merge

        Returns:
            Merged history dictionary

        Example:
            hist1 = manager.load_history(1)
            hist2 = manager.load_history(2)
            merged = manager.merge_histories(hist1, hist2)
        """
        if not histories:
            return self._empty_history()

        merged = self._empty_history()

        for hist in histories:
            for key in hist:
                if key in merged and isinstance(hist[key], list):
                    merged[key].extend(hist[key])
                elif isinstance(hist[key], list):
                    merged[key] = hist[key].copy()

        return merged

    def _empty_history(self) -> Dict[str, Any]:
        """Return empty history dictionary with standard keys."""
        return {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'train_loss_pixel': [],
            'train_loss_physics': [],
            'train_loss_advection': [],
            'train_loss_diffusion': []
        }

    def copy_checkpoint(self, src: Path, name: Optional[str] = None) -> Path:
        """Copy checkpoint file to experiment directory."""
        dst = self.checkpoints_dir / (name or src.name)
        shutil.copy2(src, dst)
        return dst

    def summary_path(self) -> Path:
        """Get path for results summary file."""
        return self.run_dir / "results_summary.txt"

    def create_summary(self, lines: str) -> None:
        """Create results summary text file."""
        with open(self.summary_path(), "w") as fp:
            fp.write(lines)
