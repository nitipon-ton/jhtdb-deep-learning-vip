#!/usr/bin/env python
"""Thin CLI wrapper around :mod:`utils.training_3d`.

Examples
--------
Train with the default 3D config::

    python scripts/train_3d.py

Override the data location and batch size::

    python scripts/train_3d.py \
        -o data.processed_dir=/scratch/superresolution/data/processed/cnn \
        -o training.batch_size=2

Evaluate the best checkpoint::

    python scripts/train_3d.py --evaluate

Run with a short smoke configuration::

    python scripts/train_3d.py -o training.epochs=1 -o training.batch_size=1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.training_3d import evaluate, load_config, run_training  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs/sr_unet_meteo_3d_velocity128.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the 3D meteorology UNet on velocity cubes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value; repeat for multiple keys.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate a checkpoint instead of training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate (default: best checkpoint for the run).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=None,
        help="Write prediction_tXXXX.npy (predicted correction) files.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.override)

    if args.evaluate:
        metrics = evaluate(
            cfg,
            checkpoint=args.checkpoint,
            split=args.split,
            save_predictions=args.save_predictions,
        )
        print(json.dumps(metrics, indent=2, default=str))
        return 0

    result = run_training(cfg, progress=not args.no_progress)
    print(json.dumps(result.summary(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
