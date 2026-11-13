"""Training, validation, and evaluation logic for 3D velocity-cube SR.

This module contains no CLI and no notebook code.  The thin entry points live
in ``scripts/train_3d.py`` and
``notebooks/super_resolution/training_sr_3d.ipynb``; both call the functions
below so training behaviour stays identical across interfaces.

The model predicts the residual correction written by
``superresolution.preprocess`` for the ``cnn`` variant:

    u_reconstructed = input + model(input)

Losses operate on that correction.  A 3D gradient term can be enabled in the
config and mirrors the gradient loss used by the 2D meteorology UNet.
"""

from __future__ import annotations

import copy
import csv
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.factory import build_model_from_config
from utils.cube_datasets import VelocityCubeDataset, compute_component_stats
from utils.experiment_manager import ExperimentManager
from utils.losses import GradientLoss3D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ("train", "val", "test")
HISTORY_KEYS = (
    "train_loss",
    "train_loss_pixel",
    "train_loss_gradient",
    "val_loss",
    "val_loss_pixel",
    "val_loss_gradient",
    "lr",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _apply_override(cfg: Dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(
            f"Override '{expression}' must look like key.subkey=value."
        )
    key, raw_value = expression.split("=", 1)
    keys = key.strip().split(".")
    node: Dict[str, Any] = cfg
    for part in keys[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"Unknown config key '{part}' in override '{expression}'.")
        node = node[part]
    leaf = keys[-1]
    if leaf not in node:
        raise KeyError(f"Unknown config key '{key}' in override '{expression}'.")
    node[leaf] = yaml.safe_load(raw_value)


def _resolve_config_paths(cfg: Dict[str, Any]) -> None:
    data = cfg.setdefault("data", {})
    if "processed_dir" not in data:
        raise KeyError("config is missing data.processed_dir")
    data["processed_dir"] = str(_resolve_path(data["processed_dir"]))

    experiment = cfg.setdefault("experiment", {})
    experiment["base_dir"] = str(
        _resolve_path(experiment.get("base_dir", "work"))
    )

    training = cfg.setdefault("training", {})
    if training.get("resume_from"):
        training["resume_from"] = str(_resolve_path(training["resume_from"]))
    evaluation = cfg.setdefault("evaluation", {})
    if evaluation.get("checkpoint"):
        evaluation["checkpoint"] = str(_resolve_path(evaluation["checkpoint"]))


def load_config(
    path: str | Path, overrides: Sequence[str] | None = None
) -> Dict[str, Any]:
    """Load a YAML config and apply ``key.subkey=value`` overrides."""
    config_path = _resolve_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} must contain a mapping.")

    for override in overrides or ():
        _apply_override(cfg, override)

    _resolve_config_paths(cfg)
    cfg.setdefault("hardware", {}).setdefault("device", "cuda")
    cfg["hardware"]["device"] = str(cfg["hardware"]["device"]).lower()
    return cfg


# ---------------------------------------------------------------------------
# Device and seeding
# ---------------------------------------------------------------------------


def resolve_device(cfg: Mapping[str, Any]) -> torch.device:
    requested = str(cfg.get("hardware", {}).get("device", "cuda")).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_enabled(cfg: Mapping[str, Any] | None, device: torch.device) -> bool:
    if cfg is None or device.type != "cuda":
        return False
    return bool(cfg.get("training", {}).get("mixed_precision", False))


def _autocast(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_dataloaders(
    cfg: Mapping[str, Any],
    splits: Sequence[str] = DEFAULT_SPLITS,
    *,
    shuffle: bool | None = None,
) -> Dict[str, DataLoader]:
    """Build one DataLoader per requested split."""
    data_cfg = cfg["data"]
    training_cfg = cfg.get("training", {})
    processed_dir = Path(data_cfg["processed_dir"])
    split_names = data_cfg.get("splits") or {name: name for name in DEFAULT_SPLITS}

    normalize = str(data_cfg.get("normalization", "none")).lower()
    stats = data_cfg.get("stats")
    if normalize == "global" and not stats:
        train_split = split_names.get("train", "train")
        stats = {
            "input": compute_component_stats(
                processed_dir, train_split, source="input"
            ),
            "target": compute_component_stats(
                processed_dir, train_split, source="target"
            ),
        }

    batch_size = int(training_cfg.get("batch_size", 1))
    num_workers = int(training_cfg.get("num_workers", 0))
    if batch_size < 1:
        raise ValueError(f"training.batch_size must be positive, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"training.num_workers must be non-negative, got {num_workers}.")

    augment = bool(data_cfg.get("augmentation", False))
    loaders: Dict[str, DataLoader] = {}
    for name in splits:
        if name not in split_names:
            raise KeyError(f"No split mapping for '{name}'.")
        dataset = VelocityCubeDataset(
            processed_dir,
            split_names[name],
            normalize=normalize,
            stats=stats,
            augment=augment and name == "train",
        )
        do_shuffle = (name == "train") if shuffle is None else shuffle
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=do_shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
    return loaders


# ---------------------------------------------------------------------------
# Optimizer, scheduler, loss
# ---------------------------------------------------------------------------


def build_optimizer(cfg: Mapping[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    training_cfg = cfg.get("training", {})
    name = str(training_cfg.get("optimizer", "adamw")).lower()
    lr = float(training_cfg.get("lr", 1.0e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    betas = tuple(training_cfg.get("betas", (0.9, 0.999)))
    parameters = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(training_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{name}'.")


def build_scheduler(
    cfg: Mapping[str, Any], optimizer: torch.optim.Optimizer
):
    training_cfg = cfg.get("training", {})
    scheduler_cfg = training_cfg.get("lr_scheduler", {})
    if not scheduler_cfg.get("enabled", False):
        return None

    kind = str(scheduler_cfg.get("type", "cosine")).lower()
    epochs = int(training_cfg.get("epochs", 100))
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("t_max", epochs)),
            eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
        )
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 30)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "min")),
            factor=float(scheduler_cfg.get("factor", 0.1)),
            patience=int(scheduler_cfg.get("patience", 10)),
        )
    if kind in {"none", "constant"}:
        return None
    raise ValueError(f"Unsupported lr_scheduler.type '{kind}'.")


class CompositeLoss3D(nn.Module):
    """Weighted pixel MSE plus an optional 3D gradient term."""

    def __init__(
        self,
        pixel_weight: float = 1.0,
        gradient_weight: float = 0.0,
        gradient_periodic: bool = False,
    ) -> None:
        super().__init__()
        if pixel_weight < 0.0 or gradient_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if pixel_weight == 0.0 and gradient_weight == 0.0:
            raise ValueError("At least one loss component must be enabled.")
        self.pixel = nn.MSELoss()
        self.pixel_weight = float(pixel_weight)
        self.gradient_weight = float(gradient_weight)
        self.gradient = (
            GradientLoss3D(periodic=gradient_periodic)
            if gradient_weight > 0.0
            else None
        )

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Compute loss terms in float32 under AMP to avoid half-precision
        # overflow in the squared differences.
        if prediction.dtype in (torch.float16, torch.bfloat16):
            prediction = prediction.float()
        if target.dtype in (torch.float16, torch.bfloat16):
            target = target.float()
        pixel = self.pixel(prediction, target)
        total = self.pixel_weight * pixel
        result: Dict[str, torch.Tensor] = {"total": total, "pixel": pixel}
        if self.gradient is not None:
            gradient = self.gradient(prediction, target)
            result["gradient"] = gradient
            result["total"] = total + self.gradient_weight * gradient
        return result


def build_loss(cfg: Mapping[str, Any]) -> CompositeLoss3D:
    components = cfg.get("loss", {}).get("components", {})
    if components.get("spectral", {}).get("enabled", False):
        raise NotImplementedError(
            "The existing spectral loss is 2D-only and is not supported for 3D cubes."
        )
    if components.get("physics", {}).get("enabled", False):
        raise NotImplementedError(
            "The existing physics losses are 2D-only and are not supported for 3D cubes."
        )
    pixel_cfg = components.get("pixel", {})
    gradient_cfg = components.get("gradient", {})

    pixel_weight = (
        float(pixel_cfg.get("weight", 1.0))
        if pixel_cfg.get("enabled", True)
        else 0.0
    )
    gradient_weight = (
        float(gradient_cfg.get("weight", 0.0))
        if gradient_cfg.get("enabled", False)
        else 0.0
    )
    return CompositeLoss3D(
        pixel_weight=pixel_weight,
        gradient_weight=gradient_weight,
        gradient_periodic=bool(gradient_cfg.get("periodic", False)),
    )


# ---------------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------------


def _unpack_batch(
    batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = batch
    return (
        inputs.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
    )


def _residual_reference(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Input that a ``residual=True`` model adds back to its correction.

    At full resolution this is just ``inputs``; if the model pre-upsamples, the
    reference is the upsampled cube so the subtraction is shape-correct.
    """
    pre_up = getattr(model, "pre_up", None)
    if pre_up is None or isinstance(pre_up, nn.Identity):
        return inputs
    return pre_up(inputs)


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler,
    clip_enabled: bool,
    max_norm: float,
) -> None:
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if scaler is not None:
        if clip_enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        if clip_enabled:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CompositeLoss3D,
    device: torch.device,
    *,
    scaler=None,
    cfg: Mapping[str, Any] | None = None,
    progress: bool = False,
    grad_accum_steps: int | None = None,
    max_batches: int | None = None,
    residual: bool = False,
) -> Dict[str, float]:
    """Run one training epoch and return averaged loss components."""
    training_cfg = (cfg or {}).get("training", {})
    if grad_accum_steps is None:
        grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive.")
    if len(loader) == 0:
        raise RuntimeError("Cannot train on an empty DataLoader.")

    clip_cfg = training_cfg.get("gradient_clipping", {})
    clip_enabled = bool(clip_cfg.get("enabled", False))
    max_norm = float(clip_cfg.get("max_norm", 1.0))
    amp = _amp_enabled(cfg, device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals: Dict[str, float] = {}
    samples = 0
    pending_steps = 0
    iterator = tqdm(loader, desc="train", leave=False, disable=not progress)

    for batch_index, batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, targets = _unpack_batch(batch, device)
        batch_size = inputs.size(0)

        with _autocast(device, amp):
            outputs = model(inputs)
            predictions = (
                outputs - _residual_reference(model, inputs)
                if residual
                else outputs
            )
            components = criterion(predictions, targets)
        loss = components["total"] / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        pending_steps += 1
        if pending_steps == grad_accum_steps:
            _optimizer_step(optimizer, scaler, clip_enabled, max_norm)
            pending_steps = 0

        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        samples += batch_size
        if progress:
            iterator.set_postfix(total=f"{totals['total'] / samples:.4e}")

    if pending_steps > 0:
        _optimizer_step(optimizer, scaler, clip_enabled, max_norm)

    if samples == 0:
        raise RuntimeError("No samples were processed.")
    result = {name: value / samples for name, value in totals.items()}
    result["n_samples"] = float(samples)
    return result


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CompositeLoss3D,
    device: torch.device,
    *,
    cfg: Mapping[str, Any] | None = None,
    progress: bool = False,
    residual: bool = False,
) -> Dict[str, float]:
    """Evaluate loss components without updating the model."""
    if len(loader) == 0:
        raise RuntimeError("Cannot validate on an empty DataLoader.")
    amp = _amp_enabled(cfg, device)
    model.eval()
    totals: Dict[str, float] = {}
    samples = 0
    iterator = tqdm(loader, desc="val", leave=False, disable=not progress)

    with torch.no_grad():
        for batch in iterator:
            inputs, targets = _unpack_batch(batch, device)
            batch_size = inputs.size(0)
            with _autocast(device, amp):
                outputs = model(inputs)
                predictions = (
                    outputs - _residual_reference(model, inputs)
                    if residual
                    else outputs
                )
                components = criterion(predictions, targets)
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            samples += batch_size
            if progress:
                iterator.set_postfix(total=f"{totals['total'] / samples:.4e}")

    if samples == 0:
        raise RuntimeError("No samples were processed.")
    result = {name: value / samples for name, value in totals.items()}
    result["n_samples"] = float(samples)
    return result


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def _model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module.state_dict()
    return model.state_dict()


def _strip_module_prefix(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def _load_torch_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    best_epoch: int,
    patience_counter: int,
    cfg: Mapping[str, Any],
    history: Mapping[str, List[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": int(epoch),
        "model": _model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
        "model_name": cfg.get("model", {}).get("architecture"),
        "input_shape": list(history.get("_input_shape", [])),
        "config": copy.deepcopy(dict(cfg)),
        "history": {key: list(values) for key, values in history.items() if key != "_input_shape"},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _checkpoint_paths(cfg: Mapping[str, Any]) -> Tuple[str, str]:
    checkpoint_cfg = cfg.get("training", {}).get("checkpoints", {})
    return (
        str(checkpoint_cfg.get("best", "best_model.pth")),
        str(checkpoint_cfg.get("last", "last_model.pth")),
    )


def _run_dir(cfg: Mapping[str, Any]) -> Path:
    experiment = cfg["experiment"]
    base_dir = Path(experiment.get("base_dir", PROJECT_ROOT / "work"))
    run_name = experiment.get("run_name") or experiment["name"]
    return base_dir / "SR" / str(run_name)


def _make_manager(
    cfg: Mapping[str, Any], *, save_config: bool = True
) -> ExperimentManager:
    experiment = cfg["experiment"]
    base_dir = Path(experiment.get("base_dir", PROJECT_ROOT / "work"))
    run_name = experiment.get("run_name") or experiment["name"]
    manager = ExperimentManager(
        task="SR",
        name=str(experiment["name"]),
        base_dir=base_dir,
        run_name=str(run_name),
    )
    manager.setup()
    if save_config:
        manager.save_config(copy.deepcopy(dict(cfg)))
    return manager


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    run_dir: Path
    history: Dict[str, List[float]]
    best_val_loss: float
    best_epoch: int
    checkpoint_path: Path
    model: nn.Module = field(repr=False)
    device: torch.device = field(repr=False)
    config: Dict[str, Any] = field(repr=False)

    def summary(self) -> Dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "checkpoint": str(self.checkpoint_path),
            "epochs_completed": len(self.history.get("val_loss", [])),
        }


def run_training(
    cfg: Dict[str, Any],
    *,
    progress: bool = False,
    callbacks: Sequence[Callable[[int, Dict[str, float]], None]] | None = None,
) -> TrainingResult:
    """Train the model described by ``cfg`` and return the run artefacts."""
    cfg = copy.deepcopy(cfg)
    training_cfg = cfg.setdefault("training", {})
    epochs = int(training_cfg.get("epochs", 100))
    if epochs < 1:
        raise ValueError(f"training.epochs must be positive, got {epochs}.")
    seed = int(training_cfg.get("seed", 42))
    seed_everything(seed)

    device = resolve_device(cfg)
    loaders = build_dataloaders(cfg, splits=("train", "val"))
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    train_dataset = train_loader.dataset
    sample_input, sample_target = train_dataset[0]
    input_shape = tuple(sample_input.shape)
    target_shape = tuple(sample_target.shape)

    model = build_model_from_config(
        cfg["model"],
        patch_size_lr=input_shape[-1],
        patch_size_hr=target_shape[-1],
    ).to(device)
    residual = bool(getattr(model, "residual", False))
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    criterion = build_loss(cfg).to(device)
    scaler = _make_grad_scaler(_amp_enabled(cfg, device))

    manager = _make_manager(cfg)
    best_name, last_name = _checkpoint_paths(cfg)
    checkpoint_dir = manager.checkpoints_dir
    best_path = checkpoint_dir / best_name
    last_path = checkpoint_dir / last_name

    history: Dict[str, List[float]] = {key: [] for key in HISTORY_KEYS}
    history["_input_shape"] = list(input_shape)  # type: ignore[assignment]

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    last_completed_epoch = 0

    resume_path = training_cfg.get("resume_from")
    if resume_path:
        state = _load_torch_checkpoint(resume_path, device)
        model.load_state_dict(
            _strip_module_prefix(state.get("model", state))
        )
        if "optimizer" in state and state["optimizer"] is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        best_epoch = int(state.get("best_epoch", 0))
        patience_counter = int(state.get("patience_counter", 0))
        saved_history = state.get("history") or {}
        for key in HISTORY_KEYS:
            if key in saved_history:
                history[key] = list(saved_history[key])
        print(f"Resumed from {resume_path} at epoch {start_epoch}.")

    early_cfg = training_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_metric = str(early_cfg.get("metric", "val_loss"))
    early_mode = str(early_cfg.get("mode", "min"))
    if early_metric != "val_loss" or early_mode != "min":
        raise NotImplementedError(
            "3D early stopping supports only metric='val_loss' with mode='min'; "
            f"got metric='{early_metric}', mode='{early_mode}'."
        )
    patience = int(early_cfg.get("patience", epochs))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    checkpoint_every = int(training_cfg.get("checkpoint_every", 0))

    log_path = manager.logs_dir / "training_log.csv"
    log_exists = resume_path and log_path.exists() and log_path.stat().st_size > 0
    log_mode = "a" if log_exists else "w"
    csv_header = [
        "epoch",
        "train_loss",
        "train_loss_pixel",
        "train_loss_gradient",
        "val_loss",
        "val_loss_pixel",
        "val_loss_gradient",
        "lr",
        "epoch_time_s",
    ]

    print(f"Device: {device}")
    print(f"Run directory: {manager.run_dir}")
    print(f"Input shape: {input_shape}; target shape: {target_shape}")
    print(f"Train batches: {len(train_loader)}; val batches: {len(val_loader)}")

    with log_path.open(log_mode, newline="") as handle:
        writer = csv.writer(handle)
        if log_mode == "w":
            writer.writerow(csv_header)

        for epoch in range(start_epoch, epochs + 1):
            started = time.time()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler=scaler,
                cfg=cfg,
                progress=progress,
                residual=residual,
            )
            val_metrics = validate(
                model,
                val_loader,
                criterion,
                device,
                cfg=cfg,
                progress=progress,
                residual=residual,
            )
            elapsed = time.time() - started
            lr = float(optimizer.param_groups[0]["lr"])
            last_completed_epoch = epoch

            row = {
                "epoch": epoch,
                "train_loss": train_metrics.get("total", 0.0),
                "train_loss_pixel": train_metrics.get("pixel", 0.0),
                "train_loss_gradient": train_metrics.get("gradient", 0.0),
                "val_loss": val_metrics.get("total", 0.0),
                "val_loss_pixel": val_metrics.get("pixel", 0.0),
                "val_loss_gradient": val_metrics.get("gradient", 0.0),
                "lr": lr,
                "epoch_time_s": elapsed,
            }
            for key in HISTORY_KEYS:
                history[key].append(float(row[key]))

            writer.writerow([row[key] for key in csv_header])
            handle.flush()

            print(
                f"epoch {epoch}/{epochs} "
                f"train={row['train_loss']:.4e} val={row['val_loss']:.4e} "
                f"pixel={row['val_loss_pixel']:.4e} "
                f"grad={row['val_loss_gradient']:.4e} lr={lr:.2e} "
                f"time={elapsed:.1f}s"
            )

            improved = row["val_loss"] < best_val_loss - min_delta
            if improved:
                best_val_loss = row["val_loss"]
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            _save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                patience_counter=patience_counter,
                cfg=cfg,
                history=history,
            )
            if improved:
                _save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    patience_counter=patience_counter,
                    cfg=cfg,
                    history=history,
                )
            if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                _save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    patience_counter=patience_counter,
                    cfg=cfg,
                    history=history,
                )

            manager.save_history(history, version=1)

            if callbacks:
                callback_payload = dict(row)
                for callback in callbacks:
                    callback(epoch, callback_payload)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(row["val_loss"])
                else:
                    scheduler.step()

            if early_enabled and patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {patience_counter} epochs)."
                )
                break

    model.eval()
    if not best_path.is_file() and last_path.is_file():
        # Validation never improved (for example, NaN losses); fall back to the
        # latest weights so evaluation still has a usable checkpoint.
        shutil.copy2(last_path, best_path)
        if best_epoch == 0:
            best_epoch = last_completed_epoch
    if best_path.is_file():
        # Return the validation-selected weights, not the last epoch's.
        best_state = _load_torch_checkpoint(best_path, device)
        model.load_state_dict(
            _strip_module_prefix(best_state.get("model", best_state))
        )
        model.eval()
    return TrainingResult(
        run_dir=manager.run_dir,
        history=history,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        checkpoint_path=best_path,
        model=model,
        device=device,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _default_checkpoint(cfg: Mapping[str, Any]) -> Path:
    best_name, _ = _checkpoint_paths(cfg)
    evaluation_cfg = cfg.get("evaluation", {})
    configured = evaluation_cfg.get("checkpoint")
    if configured:
        return Path(configured)
    return _run_dir(cfg) / "checkpoints" / best_name


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def evaluate(
    cfg: Dict[str, Any],
    checkpoint: str | Path | None = None,
    *,
    split: str = "test",
    save_predictions: bool | None = None,
) -> Dict[str, Any]:
    """Evaluate a checkpoint and write aggregate and per-timestep metrics."""
    cfg = copy.deepcopy(cfg)
    device = resolve_device(cfg)
    loaders = build_dataloaders(cfg, splits=(split,), shuffle=False)
    loader = loaders[split]
    dataset = loader.dataset
    sample_input, sample_target = dataset[0]

    checkpoint_path = Path(checkpoint) if checkpoint else _default_checkpoint(cfg)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = _load_torch_checkpoint(checkpoint_path, device)
    saved_config = state.get("config") or {}
    model_cfg = saved_config.get("model") or cfg["model"]

    model = build_model_from_config(
        model_cfg,
        patch_size_lr=sample_input.shape[-1],
        patch_size_hr=sample_target.shape[-1],
    ).to(device)
    model.load_state_dict(_strip_module_prefix(state.get("model", state)))
    model.eval()
    residual = bool(getattr(model, "residual", False))

    # Prefer the loss that was used during training when the checkpoint
    # carries its config, so reported losses stay comparable across runs.
    criterion = build_loss(saved_config if saved_config else cfg).to(device)
    loss_metrics = validate(
        model,
        loader,
        criterion,
        device,
        cfg=cfg,
        progress=False,
        residual=residual,
    )

    evaluation_cfg = cfg.get("evaluation", {})
    if save_predictions is None:
        save_predictions = bool(evaluation_cfg.get("save_predictions", False))

    manager = _make_manager(cfg, save_config=False)
    prediction_dir = manager.predictions_dir
    amp = _amp_enabled(cfg, device)
    eps = 1e-12

    samples = 0
    correction_mse_sum = 0.0
    correction_rel_sum = 0.0
    reconstruction_rel_sum = 0.0
    component_mse_sum = torch.zeros(3, dtype=torch.float64)
    rows: List[Dict[str, Any]] = []
    start = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with _autocast(device, amp):
                predictions = model(inputs)
            if predictions.dtype != inputs.dtype:
                predictions = predictions.float()

            if residual:
                correction_pred = predictions - _residual_reference(model, inputs)
                full_pred = predictions
            else:
                correction_pred = predictions
                full_pred = inputs + predictions
            full_target = inputs + targets
            correction_error = correction_pred - targets

            batch_size = inputs.size(0)
            flat_error = correction_error.reshape(batch_size, -1)
            flat_target = targets.reshape(batch_size, -1)
            flat_full_error = (full_pred - full_target).reshape(batch_size, -1)
            flat_full_target = full_target.reshape(batch_size, -1)

            per_correction_mse = flat_error.pow(2).mean(dim=1)
            per_correction_rel = torch.sqrt(
                flat_error.pow(2).sum(dim=1)
                / (flat_target.pow(2).sum(dim=1) + eps)
            )
            per_reconstruction_rel = torch.sqrt(
                flat_full_error.pow(2).sum(dim=1)
                / (flat_full_target.pow(2).sum(dim=1) + eps)
            )
            per_component_mse = correction_error.pow(2).mean(dim=(0, 2, 3, 4))

            for index in range(batch_size):
                timestep = dataset.timesteps[start + index]
                row = {
                    "timestep": timestep,
                    "correction_mse": float(per_correction_mse[index]),
                    "correction_rel_l2": float(per_correction_rel[index]),
                    "reconstruction_rel_l2": float(per_reconstruction_rel[index]),
                }
                rows.append(row)
                correction_mse_sum += row["correction_mse"]
                correction_rel_sum += row["correction_rel_l2"]
                reconstruction_rel_sum += row["reconstruction_rel_l2"]

                if save_predictions:
                    # Match superresolution-nn's inference output: the model's
                    # predicted correction, saved as (z, y, x, 3).
                    array = (
                        correction_pred[index]
                        .detach()
                        .cpu()
                        .numpy()
                        .transpose(1, 2, 3, 0)
                        .astype(np.float32)
                    )
                    _atomic_save_npy(
                        prediction_dir / f"prediction_t{timestep}.npy", array
                    )

            component_mse_sum += (
                per_component_mse.detach().cpu().double() * batch_size
            )
            samples += batch_size
            start += batch_size

    if samples == 0:
        raise RuntimeError(f"No samples evaluated for split '{split}'.")

    metrics: Dict[str, Any] = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "num_samples": samples,
        "loss": loss_metrics.get("total"),
        "pixel_loss": loss_metrics.get("pixel"),
        "gradient_loss": loss_metrics.get("gradient"),
        "correction_mse": correction_mse_sum / samples,
        "correction_rel_l2": correction_rel_sum / samples,
        "reconstruction_rel_l2": reconstruction_rel_sum / samples,
        "per_component_mse": {
            name: float(component_mse_sum[i] / samples)
            for i, name in enumerate(("u", "v", "w"))
        },
    }

    manager.save_metrics(metrics, split=split)
    per_timestep_path = manager.metrics_dir / f"{split}_per_timestep.csv"
    with per_timestep_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestep",
                "correction_mse",
                "correction_rel_l2",
                "reconstruction_rel_l2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{split}: correction MSE={metrics['correction_mse']:.4e} "
        f"correction rel-L2={metrics['correction_rel_l2']:.4e} "
        f"reconstruction rel-L2={metrics['reconstruction_rel_l2']:.4e}"
    )
    return metrics
