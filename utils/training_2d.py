"""Headless controlled 2D SR training. Reads train/validation fields only.

Uses the same model factory, MultiFieldDataset normalization and corrected
physics operators as training_sr.ipynb. Never evaluates the test partition.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import random
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from models.factory import build_model_from_config
from utils.losses import GradientLoss
from utils.physics_losses import PhysicsInformedLoss
from utils.sr_datasets import MultiFieldDataset, compute_multifield_stats
from utils.training_timing import TrainingTimer

ROOT = Path(__file__).resolve().parents[1]


def load_train_validation(config, root=ROOT):
    """Keep the training notebook's sorted-key order and train-only stats.

    Test keys may be counted to verify the declared split, but their arrays are
    never indexed or read. Flat (T,H,W) datasets are supported too.
    """
    data = config["data"]
    if (data["component"] != "omega_z" or data["normalization"]["method"] != "global"
            or data["lr_maker"] != {"method": "avgpool", "scale": 4}
            or data.get("temporal", {}).get("enabled", False)
            or data.get("augmentation", {}).get("enabled", False)):
        raise ValueError("Batch trainer requires global omega_z, avgpool x4, no temporal stacking/augmentation.")
    split = data["split"]
    nt, nv, ntest = (int(split[f"{s}_snapshots"]) for s in ("train", "val", "test"))
    if min(nt, nv, ntest) <= 0:
        raise ValueError("All split sizes must be positive.")
    stop = nt + nv
    path = (Path(root) / data["h5_path"]).resolve()
    with h5py.File(path, "r") as h5:
        if "snapshots" in h5:
            keys = sorted(h5["snapshots"].keys())
            if len(keys) < stop + ntest:
                raise ValueError("Insufficient snapshots for the declared split.")
            fields = {name: np.stack([np.asarray(h5["snapshots"][key][name])
                                     for key in keys[:stop]]) for name in ("ux", "uy", "omega_z")}
        else:
            if any(h5[name].shape[0] < stop + ntest for name in ("ux", "uy", "omega_z")):
                raise ValueError("Insufficient snapshots for the declared split.")
            keys = [str(i) for i in range(stop)]
            fields = {name: h5[name][:stop] for name in ("ux", "uy", "omega_z")}
    shape = fields["omega_z"].shape
    if shape[1:] != (128, 128) or any(a.shape != shape or not np.isfinite(a).all() for a in fields.values()):
        raise ValueError("Expected finite matching (T,128,128) fields.")
    stats = compute_multifield_stats(fields["ux"], fields["uy"], fields["omega_z"], list(range(nt)))
    if any(not np.isfinite(s).all() or s[1] <= 0 for s in stats.values()):
        raise ValueError("Invalid train-only normalization stats.")
    datasets = {}
    for name, indices in (("train", slice(0, nt)), ("val", slice(nt, stop))):
        datasets[name] = MultiFieldDataset(
            fields["ux"][indices], fields["uy"][indices], fields["omega_z"][indices],
            N=1, scale=4, normalize="global", stats=stats, transform_config={"enabled": False})
    return datasets, {"h5_path": str(path), "normalization_stats": stats,
                      "h5_size_bytes": path.stat().st_size, "h5_mtime_ns": path.stat().st_mtime_ns,
                      "train_keys": keys[:nt], "validation_keys": keys[nt:stop],
                      "test_arrays_read": False, "split": split}


class CorrectedLoss:
    def __init__(self, config, device):
        c = config["loss"]["components"]
        if (not c["pixel"].get("enabled", True) or c["pixel"].get("type") != "l2"
                or c.get("spectral", {}).get("enabled", False)):
            raise ValueError("Headless controlled study supports pixel L2, gradient, corrected physics only.")
        p = c["physics"]
        if not p.get("enabled") or p.get("lambda_grad", 0) != 0 or p.get("warmup", {}).get("enabled", False):
            raise ValueError("Use corrected physics with lambda_grad=0 and no warmup.")
        self.pixel_weight = float(c["pixel"]["weight"])
        self.gradient_weight = float(c["gradient"]["weight"]) if c["gradient"]["enabled"] else 0.
        self.gradient = GradientLoss().to(device)
        self.physics = PhysicsInformedLoss(lambda_adv=float(p["lambda_adv"]),
                                          lambda_diff=float(p["lambda_diff"]), lambda_grad=0.).to(device)

    def __call__(self, pred, target, u, v):
        pred = pred.float()
        pixel = F.mse_loss(pred, target)
        gradient = self.gradient(pred, target)
        physics = self.physics(u, v, pred, u, v, target, 1., 1.)
        total = self.pixel_weight * pixel + self.gradient_weight * gradient + physics["total"]
        return {"loss": total, "pixel": pixel, "gradient": gradient,
                "advection": physics["advection"], "diffusion": physics["diffusion"]}


def _atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _save_weights(path, model):
    temporary = path.with_suffix(".pth.tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def _optimizer_step(model, optimizer, scaler, samples, max_norm):
    # Backpropagated sample sums -> exact mean over this accumulation group,
    # including partial last batches/groups. Unscale before normalizing/clipping.
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(samples)
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def run_epoch(model, loader, criterion, device, *, optimizer=None, scaler=None,
              amp=False, accumulation=1, max_norm=1.):
    training = optimizer is not None
    model.train(training)
    totals, samples = {}, 0
    pending_samples = pending_batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad() if training else torch.inference_mode():
        for index, batch in enumerate(loader):
            inputs = batch["lr_omega"].to(device, non_blocking=True)
            target = batch["hr_omega"].unsqueeze(1).to(device, non_blocking=True)
            u = batch["hr_ux"].to(device, non_blocking=True)
            v = batch["hr_uy"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext():
                pred = model(inputs)
            if pred.shape != target.shape or not torch.isfinite(pred).all():
                raise FloatingPointError("Invalid shape or non-finite prediction.")
            components = criterion(pred, target, u, v)
            if not all(torch.isfinite(value) for value in components.values()):
                raise FloatingPointError("Non-finite loss.")
            count = inputs.size(0)
            if training:
                scaler.scale(components["loss"] * count).backward()
                pending_samples += count
                pending_batches += 1
                if pending_batches == accumulation or index + 1 == len(loader):
                    _optimizer_step(model, optimizer, scaler, pending_samples, max_norm)
                    pending_samples = pending_batches = 0
            for key, value in components.items():
                totals[key] = totals.get(key, 0.) + value.detach().item() * count
            samples += count
    if not samples:
        raise ValueError("Empty training/validation loader.")
    return {**{key: value / samples for key, value in totals.items()}, "samples": samples}


def train(config, root=ROOT):
    root = Path(root).resolve()
    t = config["training"]
    epochs, batch_size = int(t["epochs"]), int(t["batch_size"])
    accumulation = int(t.get("grad_accum_steps", 1))
    workers = int(t.get("num_workers", 0))
    metric = t.get("selection_metric", "val_loss")
    if min(epochs, batch_size, accumulation) < 1 or workers < 0 or metric not in ("val_loss", "val_mse"):
        raise ValueError("Invalid training limits or selection_metric.")
    if t.get("gan", {}).get("enabled", False):
        raise ValueError("GAN training is outside the controlled objective.")
    name = config["experiment"]["run_name"]
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("run_name must be a simple folder name.")
    directory = root / "work/SR" / name
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Run folder is nonempty: {directory}")
    device = torch.device(config["hardware"]["device"])
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError(f"Requested device unavailable/unsupported: {device}")
    seed = int(t.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(t.get("torch_num_threads", 4)))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    datasets, data_metadata = load_train_validation(config, root)
    generator = torch.Generator().manual_seed(seed)
    loaders = {key: DataLoader(ds, batch_size=batch_size, shuffle=key == "train",
                               generator=generator if key == "train" else None,
                               num_workers=workers, pin_memory=device.type == "cuda",
                               persistent_workers=workers > 0, drop_last=False)
               for key, ds in datasets.items()}
    model = build_model_from_config(config["model"], 32, 128).to(device)
    criterion = CorrectedLoss(config, device)
    opt_class = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}.get(t["optimizer"].lower())
    if opt_class is None:
        raise ValueError("Use Adam or AdamW.")
    optimizer = opt_class(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]),
                          betas=tuple(t["betas"]))
    sched = t.get("lr_scheduler", {})
    scheduler = None
    if sched.get("enabled", False):
        if sched["type"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=float(sched.get("factor", .5)),
                patience=int(sched.get("patience", 8)), threshold=float(sched.get("threshold", 1e-5)),
                threshold_mode="abs", min_lr=float(sched.get("min_lr", 1e-6)))
        elif sched["type"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(sched.get("t_max", epochs)), eta_min=float(sched.get("eta_min", 1e-6)))
        else:
            raise ValueError(f"Unsupported scheduler: {sched['type']}")
    amp = bool(t.get("mixed_precision", False) and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    clipping = t.get("gradient_clipping", {})
    max_norm = float(clipping["max_norm"]) if clipping.get("enabled", False) else None
    for subdir in ("metrics", "logs", "checkpoints", "history", "plots"):
        (directory / subdir).mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _atomic_json(directory / "metrics/data_provenance.json", data_metadata)
    names = t.get("checkpoints", {})
    best_path = directory / "checkpoints" / names.get("best", "best_model.pth")
    last_path = directory / "checkpoints" / names.get("last", "last_model.pth")
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    timer = TrainingTimer(directory, name, {
        "python": platform.python_version(), "torch": str(torch.__version__), "device": str(device),
        "device_names": [torch.cuda.get_device_name(device)] if device.type == "cuda" else [platform.processor()],
        "gpu_count": 1 if device.type == "cuda" else 0, "batch_size": batch_size,
        "effective_batch_size": batch_size * accumulation, "grad_accum_steps": accumulation,
        "seed": seed, "mixed_precision": amp, "trainer": "headless_2d",
        "selection_metric": metric,
    }, synchronize=synchronize)
    sigma = data_metadata["normalization_stats"]["omega_z"][1] + 1e-8
    best, best_epoch, stall, stop_best = math.inf, 0, 0, math.inf
    early = t.get("early_stopping", {})
    patience = int(early.get("patience", 50))
    min_epochs = int(early.get("min_epochs", 1))
    if not 1 <= min_epochs <= epochs or patience < 1:
        raise ValueError("early_stopping requires 1 <= min_epochs <= epochs and positive patience.")
    rows = []
    print(f"Run {name}; parameters={sum(p.numel() for p in model.parameters()):,}; device={device}; "
          f"train={len(datasets['train'])}, val={len(datasets['val'])}; select={metric}", flush=True)
    with (directory / "logs/training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for epoch in range(1, epochs + 1):
            timer.start_epoch()
            tr = run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer,
                           scaler=scaler, amp=amp, accumulation=accumulation, max_norm=max_norm)
            timer.end_train()
            va = run_epoch(model, loaders["val"], criterion, device, amp=amp)
            timer.end_validation()
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                   **{f"val_{k}": v for k, v in va.items()}, "val_mse": va["pixel"] * sigma**2,
                   "lr": optimizer.param_groups[0]["lr"]}
            score = row[metric]
            if score < best:
                best, best_epoch = score, epoch
                _save_weights(best_path, model)
            if score < stop_best - float(early.get("min_delta", 0.)):
                stop_best, stall = score, 0
            else:
                stall += 1
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(score)
                else:
                    scheduler.step()
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            # Persist curves even if the allocation ends later.
            _atomic_json(directory / "history/training_history_v1.json",
                         {key: [r[key] for r in rows] for key in row})
            phase = timer.end_epoch(epoch, tr["loss"], va["loss"])
            print(f"epoch={epoch}/{epochs} train={tr['loss']:.6f} val_loss={va['loss']:.6f} "
                  f"val_mse={row['val_mse']:.6f} lr={row['lr']:.2g} "
                  f"epoch_s={phase['epoch_seconds']:.2f} best_epoch={best_epoch}", flush=True)
            if early.get("enabled", True) and epoch >= min_epochs and stall >= patience:
                print(f"Early stopping: {stall} epochs without selection-metric improvement.", flush=True)
                break
    _save_weights(last_path, model)
    timing = timer.finish(best_epoch, best_path)
    selected = rows[best_epoch - 1]
    summary = {"run_name": name, "architecture": config["model"]["architecture"],
               "selection_metric": metric, "best_selection_value": best, "best_epoch": best_epoch,
               "best_val_loss": selected["val_loss"], "best_val_mse": selected["val_mse"],
               "epochs_completed": len(rows), "parameter_count": sum(p.numel() for p in model.parameters()),
               "timing": timing, "test_evaluated": False,
               "artifacts": {"best_checkpoint": str(best_path), "last_checkpoint": str(last_path)}}
    _atomic_json(directory / "metrics/training_summary.json", summary)
    # Figures run after the timed session, and are not required for completion.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ("train_loss", "val_loss"):
        axes[0].plot([r["epoch"] for r in rows], [r[key] for r in rows], label=key)
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].plot([r["epoch"] for r in rows], [r["val_mse"] for r in rows])
    axes[1].set_title("Validation MSE (physical omega_z units)")
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(directory / "plots/training_curves.png", dpi=150)
    plt.close(fig)
    return summary
