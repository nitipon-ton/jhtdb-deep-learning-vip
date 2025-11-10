"""
Utility helpers for organising experiment work directories.

Each experiment gets its own folder under ``work/{task}/{name}_{timestamp}``
with sub-directories for checkpoints, predictions, metrics, and plots.  The
manager keeps track of the paths and provides small helper methods to save
artefacts from notebooks without repeating boilerplate.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json
import shutil

import yaml


class ExperimentManager:
    def __init__(self, task: str, name: str, base_dir: Optional[Path] = None):
        self.task = task
        self.name = name
        self.base_dir = Path(base_dir) if base_dir else Path("work")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / task / f"{name}_{timestamp}"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.predictions_dir = self.run_dir / "predictions"
        self.metrics_dir = self.run_dir / "metrics"
        self.plots_dir = self.run_dir / "plots"
        self.logs_dir = self.run_dir / "logs"

    def setup(self) -> None:
        for path in [
            self.run_dir,
            self.checkpoints_dir,
            self.predictions_dir,
            self.metrics_dir,
            self.plots_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Dict[str, Any], file_name: str = "config.yaml") -> None:
        with open(self.run_dir / file_name, "w") as fp:
            yaml.safe_dump(config, fp, sort_keys=False)

    def save_metrics(self, metrics: Dict[str, Any], split: str = "val") -> Path:
        path = self.metrics_dir / f"{split}_metrics.json"
        with open(path, "w") as fp:
            json.dump(metrics, fp, indent=2)
        return path

    def save_history(self, history: Dict[str, Any], file_name: str = "training_history.json") -> Path:
        path = self.run_dir / file_name
        with open(path, "w") as fp:
            json.dump(history, fp, indent=2)
        return path

    def copy_checkpoint(self, src: Path, name: Optional[str] = None) -> Path:
        dst = self.checkpoints_dir / (name or src.name)
        shutil.copy2(src, dst)
        return dst

    def summary_path(self) -> Path:
        return self.run_dir / "results_summary.txt"

    def create_summary(self, lines: str) -> None:
        with open(self.summary_path(), "w") as fp:
            fp.write(lines)
