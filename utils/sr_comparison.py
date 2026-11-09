"""Held-out evaluation for the controlled 32 -> 128 omega_z study.

No training, checkpoint search, epoch selection, or test-based ranking occurs here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from models.factory import build_model_from_config
from utils.metrics import ssim
from utils.physics_losses import AdvectionLoss, DiffusionLoss
from utils.sr_datasets import MultiFieldDataset, compute_multifield_stats, create_snapshot_splits

METRICS = ("mse", "mae", "psnr", "ssim", "advection_error", "diffusion_error",
           "physical_advection_mse", "physical_diffusion_mse", "inference_ms")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_runs(root, config_paths, run_dirs=None):
    """Resolve exact runs; fail before evaluation if any comparison is incompatible."""
    root = Path(root).resolve()
    runs = []
    reference = None
    run_dirs = run_dirs or {}
    if not config_paths:
        raise ValueError("Select at least one config.")
    for config_path in config_paths:
        path = (root / config_path).resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        data = config["data"]
        if (data["component"] != "omega_z"
                or data["normalization"]["method"] != "global"
                or data.get("mode", "standard") != "standard"
                or data.get("temporal", {}).get("enabled", False)
                or data["lr_maker"] != {"method": "avgpool", "scale": 4}
                or config["model"]["scale_factor"] != 4):
            raise ValueError(f"{path}: requires single-frame, global-normalized omega_z, avgpool x4.")
        if data["split"] != {"train_snapshots": 3000, "val_snapshots": 600, "test_snapshots": 400}:
            raise ValueError(f"{path}: expected the fixed 3000/600/400 split.")
        signature = {"data": data, "domain": config.get("domain"), "loss": config["loss"]}
        if reference is not None and signature != reference:
            raise ValueError("Configs must share data, domain, and loss settings for this controlled study.")
        reference = signature
        name = config["experiment"].get("run_name")
        if not name:
            raise ValueError("Set experiment.run_name to the exact trained run; no latest-run search is used.")
        if name == "bicubic" or any(r["name"] == name for r in runs):
            raise ValueError(f"Duplicate or reserved run name: {name}")
        directory = (root / run_dirs.get(name, Path("work/SR") / name)).resolve()
        checkpoint = directory / "checkpoints/best_model.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing validation-selected checkpoint: {checkpoint}. No fallback to last_model.pth.")
        saved_config = directory / "config.yaml"
        if saved_config.exists():
            saved = yaml.safe_load(saved_config.read_text(encoding="utf-8"))
            for section in ("data", "model", "domain", "loss"):
                if saved.get(section) != config.get(section):
                    raise ValueError(f"{path}: {section} differs from trained run config {saved_config}")
        runs.append({"name": name, "config_path": str(path), "config": config,
                     "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                     "saved_config_verified": saved_config.exists()})
    return runs


def load_data(root, config):
    """Match training's lexicographic snapshot ordering and train-only float32 stats."""
    path = (Path(root) / config["data"]["h5_path"]).resolve()
    with h5py.File(path, "r") as h5:
        if "snapshots" in h5:
            keys = sorted(h5["snapshots"].keys())
            fields = {name: np.stack([np.asarray(h5["snapshots"][k][name]) for k in keys])
                      for name in ("ux", "uy", "omega_z")}
        else:
            fields = {name: np.asarray(h5[name]) for name in ("ux", "uy", "omega_z")}
            keys = [str(i) for i in range(len(fields["omega_z"]))]
    shape = fields["omega_z"].shape
    if len(shape) != 3 or shape[1:] != (128, 128):
        raise ValueError(f"Expected (T,128,128) fields, got {shape}")
    if any(a.shape != shape or not np.isfinite(a).all() for a in fields.values()):
        raise ValueError("Fields must have matching shapes and finite values.")
    train, val, test = create_snapshot_splits(shape[0], **config["data"]["split"])
    stats = compute_multifield_stats(fields["ux"], fields["uy"], fields["omega_z"], train)
    if any(not np.isfinite(s).all() or s[1] <= 0 for s in stats.values()):
        raise ValueError("Invalid training normalization statistics.")
    dataset = MultiFieldDataset(fields["ux"][test], fields["uy"][test], fields["omega_z"][test],
                                N=1, scale=4, normalize="global", stats=stats,
                                transform_config={"enabled": False})
    train_omega = fields["omega_z"][train]
    data_range = float(train_omega.max() - train_omega.min())
    grid = config["domain"]["dns_info"]["effective_resolution"]
    dx, dy = float(grid["dx"]), float(grid["dy"])
    if min(dx, dy, data_range) <= 0:
        raise ValueError("Grid spacings and training data range must be positive.")
    return dataset, {"h5_path": str(path), "h5_size_bytes": path.stat().st_size,
                     "h5_mtime_ns": path.stat().st_mtime_ns,
                     "train_indices": train, "validation_indices": val, "test_indices": test,
                     "test_snapshot_keys": [keys[i] for i in test],
                     "normalization_stats": stats, "data_range": data_range, "dx": dx, "dy": dy}


def physical_terms(omega, u, v, dx, dy):
    """Unclipped central stencils on the 126x126 interior, last axis x (as in training)."""
    gx = (omega[..., 1:-1, 2:] - omega[..., 1:-1, :-2]) / (2 * dx)
    gy = (omega[..., 2:, 1:-1] - omega[..., :-2, 1:-1]) / (2 * dy)
    center = omega[..., 1:-1, 1:-1]
    lap = ((omega[..., 1:-1, 2:] - 2 * center + omega[..., 1:-1, :-2]) / dx**2
           + (omega[..., 2:, 1:-1] - 2 * center + omega[..., :-2, 1:-1]) / dy**2)
    return u[..., 1:-1, 1:-1] * gx + v[..., 1:-1, 1:-1] * gy, lap


def sample_metrics(pred, target, u, v, metadata):
    """One sample; physical pixel metrics plus training-compatible physics errors."""
    if pred.shape != target.shape or not torch.isfinite(pred).all():
        raise ValueError(f"Invalid prediction: shape={pred.shape}, expected={target.shape}")
    stats = metadata["normalization_stats"]
    mu, sigma = stats["omega_z"]
    # MultiFieldDataset divides by sigma + 1e-8; invert that exact transform.
    p, t = pred.double() * (sigma + 1e-8) + mu, target.double() * (sigma + 1e-8) + mu
    error = p - t
    mse = error.square().mean().item()
    dynamic_range = metadata["data_range"]
    # Identical reconstruction convention: cap at float64 tiny to retain valid JSON.
    psnr = 10 * np.log10(dynamic_range**2 / max(mse, np.finfo(np.float64).tiny))
    pu = u.double() * (stats["ux"][1] + 1e-8) + stats["ux"][0]
    pv = v.double() * (stats["uy"][1] + 1e-8) + stats["uy"][0]
    pa, pd = physical_terms(p, pu, pv, metadata["dx"], metadata["dy"])
    ta, td = physical_terms(t, pu, pv, metadata["dx"], metadata["dy"])
    return {"mse": mse, "mae": error.abs().mean().item(), "psnr": float(psnr),
            "ssim": ssim(p, t, data_range=dynamic_range).item(),
            "advection_error": AdvectionLoss()(u, v, pred, target, 1., 1.).item(),
            "diffusion_error": DiffusionLoss()(pred, target, 1., 1.).item(),
            "physical_advection_mse": (pa - ta).square().mean().item(),
            "physical_diffusion_mse": (pd - td).square().mean().item()}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(runs, dataset, metadata, device="cuda", warmup=20, example_indices=(0, 199, 399)):
    """FP32 batch-one forward timing excludes transfer, metrics, and disk IO."""
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda for synchronized timing.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Select a PACE GPU kernel or explicitly use cpu.")
    if warmup < 1 or len(dataset) != 400:
        raise ValueError("Require warmup >= 1 and all 400 test samples.")
    if any(i < 0 or i >= 400 for i in example_indices):
        raise ValueError("Example indices must be fixed test offsets in [0,399].")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    rows, summary, examples = [], [], {}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    # Input-only warmup; no test targets or scores are used to make decisions.
    warm_input = torch.zeros(1, 1, 32, 32, device=device)
    for run in [{"name": "bicubic"}] + runs:
        name = run["name"]
        model = None
        if name == "bicubic":
            forward = lambda x: F.interpolate(x, size=(128, 128), mode="bicubic", align_corners=False)
            parameter_count = 0
        else:
            model = build_model_from_config(run["config"]["model"], 32, 128)
            state = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
            state = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            model = model.float().to(device).eval()
            parameter_count = sum(p.numel() for p in model.parameters())
            forward = model
        for _ in range(warmup):
            forward(warm_input)
        _sync(device)
        model_rows = []
        for offset, batch in enumerate(loader):
            lr = batch["lr_omega"].to(device)
            _sync(device)
            start = time.perf_counter()
            pred = forward(lr)
            _sync(device)
            elapsed_ms = (time.perf_counter() - start) * 1000
            pred = pred.cpu()
            target = batch["hr_omega"].unsqueeze(1)
            row = {"model": name, "test_offset": offset,
                   "snapshot_index": metadata["test_indices"][offset],
                   "snapshot_key": metadata["test_snapshot_keys"][offset],
                   **sample_metrics(pred, target, batch["hr_ux"], batch["hr_uy"], metadata),
                   "inference_ms": elapsed_ms}
            if not all(np.isfinite(row[k]) for k in METRICS):
                raise ValueError(f"Non-finite metrics for {name}, test offset {offset}")
            model_rows.append(row)
            if offset in example_indices:
                mu, sigma = metadata["normalization_stats"]["omega_z"]
                payload = examples.setdefault(offset, {"hr": target.squeeze().numpy() * (sigma + 1e-8) + mu})
                payload[name] = pred.squeeze().numpy() * (sigma + 1e-8) + mu
        if len(model_rows) != 400:
            raise RuntimeError(f"{name}: did not evaluate all 400 samples")
        aggregate = {"model": name, "num_test_samples": 400, "parameter_count": parameter_count}
        for key in METRICS:
            values = np.asarray([r[key] for r in model_rows])
            aggregate[key] = float(values.mean())
            aggregate[key + "_std"] = float(values.std())
        aggregate["inference_ms_median"] = float(np.median([r["inference_ms"] for r in model_rows]))
        summary.append(aggregate)
        rows.extend(model_rows)
        print(f"{name}: 400/400 test samples; MSE={aggregate['mse']:.6g}, "
              f"PSNR={aggregate['psnr']:.3f} dB, forward={aggregate['inference_ms']:.3f} ms/sample")
        del forward, model
    return rows, summary, examples


def save_results(output_dir, runs, metadata, rows, summary, examples, device, warmup):
    """Save a provenance manifest, tidy per-sample table, summary, and scalable plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "per_sample_metrics.csv", rows)
    _write_csv(output_dir / "summary.csv", summary)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "git_revision": revision,
              "selection_policy": "Configs fixed before testing; best_model.pth selected by validation loss only. Test is reporting only.",
              "metric_definitions": {
                  "mse_mae": "Denormalized omega_z; mean over all 128x128 pixels per sample.",
                  "psnr": "10 log10(training omega_z range squared / per-sample MSE); perfect match uses float64 tiny floor.",
                  "ssim": "Repository SSIM: Gaussian 11x11 sigma=1.5, zero padding; fixed training omega_z max-min range.",
                  "advection_error": "Unweighted training AdvectionLoss MSE: normalized DNS velocities, normalized omega, dx=dy=1, interior central stencil; terms clipped at +/-1000 as in training.",
                  "diffusion_error": "Unweighted training DiffusionLoss MSE: normalized omega, dx=dy=1, interior 5-point stencil; terms clipped at +/-1000 as in training.",
                  "physical_advection_mse": "Denormalized DNS u,v and omega; unclipped interior central stencil, configured dx,dy; last axis x, as in training.",
                  "physical_diffusion_mse": "Denormalized omega Laplacian MSE; configured dx,dy; no viscosity factor, no clipping.",
                  "aggregation": "Arithmetic mean and population std across the same 400 samples. PSNR is mean per-image PSNR.",
                  "inference_ms": "FP32 eval/inference_mode, batch=1, synchronized forward wall time; excludes H2D/D2H transfers, IO and metrics; includes CUDA synchronization overhead.",
                  "parameter_count": "All model parameters including frozen parameters; bicubic=0."},
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "device": str(device),
                              "device_name": torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == "cuda" else platform.processor(),
                              "cuda": torch.version.cuda, "warmup_forwards": warmup, "precision": "float32", "batch_size": 1},
              "data": metadata, "runs": runs, "summary": summary}
    (output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    names = [s["model"] for s in summary]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, key in zip(axes.flat, ("mse", "mae", "psnr", "ssim", "advection_error", "diffusion_error", "parameter_count", "inference_ms")):
        ax.bar(range(len(names)), [s[key] for s in summary])
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
        ax.set_title(key)
    fig.tight_layout()
    fig.savefig(output_dir / "metric_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, key in zip(axes.flat, METRICS[:6]):
        for name in names:
            subset = [r for r in rows if r["model"] == name]
            ax.plot([r["snapshot_index"] for r in subset], [r[key] for r in subset], label=name, alpha=.7)
        ax.set_title(key)
        ax.set_xlabel("Held-out snapshot index")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "per_sample_comparison.png", dpi=160)
    plt.close(fig)
    for offset, payload in examples.items():
        fig, axes = plt.subplots(2, len(names) + 1, figsize=(4 * (len(names) + 1), 7), squeeze=False)
        vmax = max(float(np.abs(a).max()) for a in payload.values()) or 1.
        err_max = max(float(np.abs(payload[n] - payload["hr"]).max()) for n in names) or 1.
        for j, name in enumerate(["hr"] + names):
            im = axes[0, j].imshow(payload[name], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            axes[0, j].set_title(name, fontsize=9)
            fig.colorbar(im, ax=axes[0, j], shrink=.7)
            im = axes[1, j].imshow(np.abs(payload[name] - payload["hr"]), cmap="magma", vmin=0, vmax=err_max, interpolation="nearest")
            axes[1, j].set_title("Absolute error")
            fig.colorbar(im, ax=axes[1, j], shrink=.7)
        fig.suptitle(f"Preselected test offset {offset}; snapshot {metadata['test_indices'][offset]}")
        fig.tight_layout()
        fig.savefig(output_dir / f"fields_test_{offset:03d}.png", dpi=160)
        plt.close(fig)
    return output_dir
