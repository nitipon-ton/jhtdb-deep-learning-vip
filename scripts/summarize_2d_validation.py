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
        summary_path = path.parent / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        selection_metric = summary.get("selection_metric", "val_loss")
        selection_value = summary.get("best_selection_value", best["val_loss"])
        rows.append({"run_name": timing["run_name"], "best_val_loss": best["val_loss"],
                     "selection_metric": selection_metric, "selection_value": selection_value,
                     "best_val_mse": summary.get("best_val_mse"),
                     "best_epoch": best_epoch, "epochs_completed": timing["epochs_completed"],
                     "training_total_seconds": timing["total_training_seconds"],
                     "train_seconds_per_epoch": timing["mean_train_seconds_per_epoch"],
                     "run_dir": str(path.parents[1]),
                     "training_device_names": "; ".join(timing["environment"].get("device_names", []))})
    # Different objectives have different scales; do not silently rank them together.
    return sorted(rows, key=lambda row: (row["selection_metric"], row["selection_value"]))


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
        print(f"{row['run_name']}: {row['selection_metric']}={row['selection_value']:.6f}; "
              f"val_loss_at_selected_epoch={row['best_val_loss']:.6f}; total={row['training_total_seconds']:.1f}s")
    print("Saved:", args.output)
