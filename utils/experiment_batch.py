"""Deterministic search preparation and batch bookkeeping (no torch import)."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import yaml

from utils.training_timing import file_sha256, read_training_timing

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def config_identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def prepare_search(plan_path, root=ROOT):
    """Freeze exact candidates once. Existing manifests must match the plan."""
    root = Path(root).resolve()
    plan_path = Path(plan_path)
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    tag = str(plan["search"]["tag"])
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
        raise ValueError("Search tag must be alphanumeric with optional underscores/hyphens.")
    directory = (root / plan["search"]["output_dir"]).resolve()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["plan"] != plan:
            raise ValueError("Search plan changed. Use a new tag and output_dir; existing candidates are frozen.")
        for item in manifest["trials"]:
            cfg = yaml.safe_load((root / item["config_path"]).read_text(encoding="utf-8"))
            if config_identity(cfg) != item["config_sha256"]:
                raise ValueError(f"Frozen candidate was edited: {item['config_path']}")
        return directory, manifest
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output folder has files but no manifest: {directory}")
    trials_per_arch = int(plan["search"]["trials_per_architecture"])
    if trials_per_arch < 1 or not plan["models"]:
        raise ValueError("Declare at least one model and positive trials_per_architecture.")
    # Build everything in memory before writing anything.
    prepared = []
    for arch, specification in plan["models"].items():
        base = yaml.safe_load((root / specification["base_config"]).read_text(encoding="utf-8"))
        if base["model"]["architecture"] != arch:
            raise ValueError(f"Base architecture mismatch: {arch}")
        capacities = specification["capacities"]
        options = specification.get("options", {})
        space = plan["search"]["space"]
        if not capacities or any(not values for values in [*space.values(), *options.values()]):
            raise ValueError("Search dimensions must be nonempty.")
        # Independent stable streams: adding a family does not alter existing trials.
        salt = int(hashlib.sha256(arch.encode()).hexdigest()[:8], 16)
        rng = random.Random(int(plan["search"].get("seed", 2026)) + salt)
        seen = set()
        for index in range(trials_per_arch):
            for attempt in range(10000):
                cfg = copy.deepcopy(base)
                cfg["training"].update(copy.deepcopy(plan["training"]))
                # First trials cover every declared capacity, then random search.
                capacity_index = index % len(capacities) if index < len(capacities) else rng.randrange(len(capacities))
                cfg["model"][arch].update(copy.deepcopy(capacities[capacity_index]))
                for key, values in options.items():
                    cfg["model"][arch][key] = copy.deepcopy(rng.choice(values))
                for key, values in space.items():
                    cfg["training"][key] = copy.deepcopy(rng.choice(values))
                trial_id = config_identity({"model": cfg["model"], "training": cfg["training"]})
                if trial_id not in seen:
                    seen.add(trial_id)
                    break
            else:
                raise ValueError(f"Requested more unique trials than search space supports for {arch}.")
            run_name = f"study2d_{tag}_{arch}_t{index:03d}_s{cfg['training']['seed']}"
            cfg["experiment"].update(name=run_name, run_name=run_name,
                                      description="Frozen validation-only larger-model search; no test evaluation.")
            path = directory / "configs" / f"{run_name}.yaml"
            prepared.append((path, cfg, {"run_name": run_name, "architecture": arch,
                                         "config_path": str(path), "config_sha256": config_identity(cfg)}))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "configs").mkdir(exist_ok=True)
    for path, config, _ in prepared:
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
                "selection_metric": plan["training"].get("selection_metric", "val_loss"),
                "test_evaluated": False, "trials": [item for _, _, item in prepared]}
    write_json(manifest_path, manifest)
    return directory, manifest


def preflight_configs(paths, root=ROOT):
    """Validate a batch before starting workers; never silently compare different losses."""
    trials, seen = [], set()
    reference = None
    metric = None
    for path in paths:
        path = (Path(root) / path).resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = config["experiment"]["run_name"]
        if not name or name in seen or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid/duplicate run name: {name}")
        seen.add(name)
        selected_metric = config["training"].get("selection_metric", "val_loss")
        if selected_metric not in ("val_loss", "val_mse") or (metric is not None and metric != selected_metric):
            raise ValueError("Use a single selection metric per batch: val_loss or val_mse.")
        metric = selected_metric
        signature = {key: config[key] for key in ("data", "domain", "loss")}
        if reference is not None and signature != reference:
            raise ValueError("All batch candidates must share data/domain/loss settings.")
        reference = signature
        trials.append({"run_name": name, "architecture": config["model"]["architecture"],
                       "config_path": str(path), "config_sha256": config_identity(config)})
    if not trials:
        raise ValueError("No candidate configs found.")
    return trials, metric


def completed_result(trial, root=ROOT):
    directory = Path(root) / "work/SR" / trial["run_name"]
    saved_path = directory / "config.yaml"
    if not saved_path.exists():
        return None
    saved = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
    if config_identity(saved) != trial["config_sha256"]:
        raise ValueError(f"Run already exists with different config: {directory}")
    checkpoint = directory / "checkpoints" / saved["training"].get("checkpoints", {}).get("best", "best_model.pth")
    if not checkpoint.is_file():
        return None
    timing = read_training_timing(directory, trial["run_name"], file_sha256(checkpoint))
    if not timing or not timing.get("completed"):
        return None
    metric = saved["training"].get("selection_metric", "val_loss")
    summary_path = directory / "metrics/training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if summary.get("selection_metric"):
        if summary["selection_metric"] != metric or summary.get("test_evaluated") is not False:
            raise ValueError(f"Unexpected selection/test provenance: {summary_path}")
        value = float(summary["best_selection_value"])
        val_loss, val_mse = summary["best_val_loss"], summary["best_val_mse"]
    elif metric == "val_loss":
        # Backward compatible with completed notebook runs from the first sweep.
        row = timing["per_epoch"][timing["best_epoch"] - 1]
        value = val_loss = float(row["val_loss"])
        val_mse = None
    else:
        return None
    if not math.isfinite(value):
        raise ValueError("Non-finite completed selection value.")
    return {**trial, "selection_metric": metric, "selection_value": value,
            "best_val_loss": val_loss, "best_val_mse": val_mse,
            "epochs_completed": timing["epochs_completed"], "best_epoch": timing["best_epoch"],
            "training_total_seconds": timing["total_training_seconds"],
            "train_seconds_per_epoch": timing["mean_train_seconds_per_epoch"],
            "parameter_count": summary.get("parameter_count"),
            "training_device_names": "; ".join(timing["environment"].get("device_names", []))}


def save_batch_reports(directory, trials, state, root=ROOT, completed=None):
    results = list(completed.values()) if completed is not None else []
    if completed is None:
        for trial in trials:
            result = completed_result(trial, root)
            if result:
                results.append(result)
    results.sort(key=lambda r: r["selection_value"])
    path = Path(directory) / "validation_results.csv"
    if results:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    winners = {}
    for result in results:
        winners.setdefault(result["architecture"], result)
    write_json(Path(directory) / "selected_configs.json", {
        "selection_policy": "minimum validation selection metric within each family; no test data",
        "provisional": len(results) != len(trials), "completed": len(results), "planned": len(trials),
        "CONFIG_PATHS": [r["config_path"] for r in winners.values()], "winners": winners,
    })
    write_json(Path(directory) / "batch_status.json", {
        "updated_utc": datetime.now(timezone.utc).isoformat(), "trials": state,
        "completed_count": len(results), "planned_count": len(trials), "test_evaluated": False,
        "completed_training_seconds_sum": sum(r["training_total_seconds"] for r in results),
        "attempt_worker_seconds_sum": sum(attempt.get("worker_wall_seconds", 0.)
                                          for entry in state.values() for attempt in entry.get("attempts", [])),
        "note": "Sum covers completed training sessions only; failure attempts have separate worker wall times/logs.",
    })
    return results
