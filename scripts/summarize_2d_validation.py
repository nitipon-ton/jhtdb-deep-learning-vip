"""Summarize completed 2D runs using validation and timing records only.

Run from the project root: python scripts/summarize_2d_validation.py
No checkpoints or test scores are used to rank runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect(base):
    rows = []
    for path in sorted(Path(base).glob("*/metrics/training_timing.json")):
        timing = json.loads(path.read_text(encoding="utf-8"))
        if not timing.get("completed"):
            continue
        epochs = timing["per_epoch"]
        best_epoch = timing["best_epoch"]
        best = next(r for r in epochs if r["epoch"] == best_epoch)
        rows.append({"run_name": timing["run_name"], "best_val_loss": best["val_loss"],
                     "best_epoch": best_epoch, "epochs_completed": timing["epochs_completed"],
                     "training_total_seconds": timing["total_training_seconds"],
                     "train_seconds_per_epoch": timing["mean_train_seconds_per_epoch"],
                     "run_dir": str(path.parents[1]),
                     "training_device_names": "; ".join(timing["environment"].get("device_names", []))})
    return sorted(rows, key=lambda row: row["best_val_loss"])


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=root / "work/SR")
    parser.add_argument("--output", type=Path, default=root / "work/SR/validation_summary.csv")
    args = parser.parse_args()
    rows = collect(args.runs_dir)
    if not rows:
        raise SystemExit("No completed timed runs found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['run_name']}: validation={row['best_val_loss']:.6f}; total={row['training_total_seconds']:.1f}s")
    print("Saved:", args.output)
