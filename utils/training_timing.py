"""Persistent wall-clock timing for fresh 2D training runs (stdlib only)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


TIMING_COLUMNS = ("training_total_seconds", "train_only_total_seconds",
                  "train_seconds_per_epoch", "epoch_seconds_mean",
                  "epochs_completed", "best_epoch", "time_to_best_seconds")


def timing_columns(report=None, baseline=False):
    """Stable CSV schema; unknown historical timings stay missing, never zero."""
    if baseline:
        return {**dict.fromkeys(TIMING_COLUMNS, 0), "timing_status": "not_applicable"}
    if not report or not report.get("completed"):
        return {**dict.fromkeys(TIMING_COLUMNS), "timing_status": "missing_or_incomplete"}
    return dict(zip(TIMING_COLUMNS, (
        report["total_training_seconds"], report["total_train_only_seconds"],
        report["mean_train_seconds_per_epoch"], report["mean_epoch_seconds"],
        report["epochs_completed"], report["best_epoch"], report["time_to_best_seconds"],
    )), timing_status="recorded")


def read_training_timing(run_dir, run_name, checkpoint_sha256):
    path = Path(run_dir) / "metrics/training_timing.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("run_name") != run_name:
        raise ValueError(f"Timing run identity does not match {run_name}: {path}")
    if report.get("completed") and report.get("best_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"Timing report refers to different best weights: {path}")
    return report


class TrainingTimer:
    """Synchronize phase boundaries; persist completed epochs after every epoch.

    train_seconds includes batch loading/transfers, forward/backward, optimizer,
    and training progress updates. validation_seconds covers the validation loop.
    epoch_seconds additionally includes logging, checkpoint writes and scheduling.
    total_training_seconds is the entire timed training-loop session, including
    validation and final checkpoint/history writes; excludes initial data/model
    setup and subsequent plots, test evaluation, and notebook idle time.
    """

    def __init__(self, run_dir, run_name, environment, synchronize=lambda: None,
                 clock=time.perf_counter):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "metrics/training_timing.json"
        self.csv_path = self.run_dir / "logs/epoch_timing.csv"
        self.sync, self.clock = synchronize, clock
        self.rows = []
        self.environment = environment
        self.run_name = run_name
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.sync()
        self.started = self.clock()
        self.epoch_started = None
        self._persist(False)

    def start_epoch(self):
        if self.epoch_started is not None:
            raise RuntimeError("Previous timing epoch was not closed.")
        self.sync()
        self.epoch_started = self.clock()
        self.train_ended = self.validation_ended = None

    def end_train(self):
        self.sync()
        self.train_ended = self.clock()

    def end_validation(self):
        self.sync()
        self.validation_ended = self.clock()

    def end_epoch(self, epoch, train_loss, val_loss):
        if self.epoch_started is None or self.train_ended is None or self.validation_ended is None:
            raise RuntimeError("Close train and validation phases before ending the epoch.")
        if not all(math.isfinite(v) for v in (train_loss, val_loss)):
            raise ValueError("Non-finite training/validation loss; no valid completed epoch.")
        self.sync()
        self.rows.append({"epoch": epoch,
                          "train_seconds": self.train_ended - self.epoch_started,
                          "validation_seconds": self.validation_ended - self.train_ended,
                          "epoch_seconds": self.clock() - self.epoch_started,
                          "train_loss": float(train_loss), "val_loss": float(val_loss)})
        self.epoch_started = None
        self._persist(False)
        return self.rows[-1]

    def _persist(self, completed, best_epoch=None, best_checkpoint=None):
        elapsed = self.clock() - self.started
        count = len(self.rows)
        report = {"schema_version": 1, "run_name": self.run_name,
                  "completed": completed, "started_utc": self.started_utc,
                  "environment": self.environment, "epochs_completed": count,
                  "total_training_seconds": elapsed,
                  "total_train_only_seconds": sum(r["train_seconds"] for r in self.rows),
                  "total_validation_seconds": sum(r["validation_seconds"] for r in self.rows),
                  "mean_train_seconds_per_epoch": sum(r["train_seconds"] for r in self.rows) / count if count else None,
                  "mean_epoch_seconds": sum(r["epoch_seconds"] for r in self.rows) / count if count else None,
                  "best_epoch": best_epoch,
                  "best_checkpoint_sha256": file_sha256(best_checkpoint) if best_checkpoint else None,
                  "time_to_best_seconds": sum(r["epoch_seconds"] for r in self.rows[:best_epoch]) if best_epoch else None,
                  "definitions": self.__class__.__doc__, "per_epoch": self.rows}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.path)
        if self.rows:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        return report

    def finish(self, best_epoch, best_checkpoint):
        if self.epoch_started is not None or not 1 <= best_epoch <= len(self.rows):
            raise ValueError("A completed epoch with a valid validation-selected checkpoint is required.")
        self.sync()
        return self._persist(True, best_epoch, best_checkpoint)
