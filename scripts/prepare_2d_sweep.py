"""Write validation-only candidate configs; never train or inspect test scores.

Default: six trials for each non-CNN family (two capacities x three learning
rates); keep configs/study_2d/residual_cnn.yaml as the fixed CNN reference.
Use --include-cnn for six trials per family including CNN.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("unet_meteo", "unet", "edsr", "hdsc_ms", "swinir", "esrgan", "mlp")
LEARNING_RATES = (5e-5, 1e-4, 3e-4)


def capacity_models(config):
    """Return two explicitly declared capacities; do not call either 'best'."""
    small, large = copy.deepcopy(config["model"]), copy.deepcopy(config["model"])
    arch = small["architecture"]
    if arch == "unet_meteo":
        small[arch]["feature_channels"] = [16, 32, 64, 128, 128]
        large[arch]["feature_channels"] = [32, 64, 128, 256, 256]
    elif arch == "unet":
        small[arch]["base_ch"], large[arch]["base_ch"] = 16, 32
    elif arch == "edsr":
        small[arch].update(n_feats=32, n_resblocks=8)
        large[arch].update(n_feats=64, n_resblocks=16)
    elif arch == "hdsc_ms":
        small[arch].update(dsc_width=16, ms_mid=8, ms_small=4)
        large[arch].update(dsc_width=32, ms_mid=16, ms_small=8)
    elif arch == "swinir":
        small[arch].update(embed_dim=32, depths=4)
        large[arch].update(embed_dim=64, depths=6)
    elif arch == "esrgan":
        small[arch].update(nf=32, gc=16, num_blocks=4)
        large[arch].update(nf=64, gc=32, num_blocks=8)
    elif arch == "mlp":
        small[arch]["hidden_dims"] = [256, 512, 256]
        large[arch]["hidden_dims"] = [512, 1024, 512]
    elif arch == "residual_cnn":
        small[arch]["hidden_channels"], large[arch]["hidden_channels"] = 32, 64
    else:
        raise ValueError(arch)
    return small, large


def generate(output_dir, architectures=ARCHITECTURES, seed=42, tag="tune01"):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite sweep configs: {output_dir}")
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
        raise ValueError("Use a nonempty alphanumeric run tag with optional underscores/hyphens.")
    candidates = []
    for arch in architectures:
        path = ROOT / "configs/study_2d" / f"{arch}.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        for capacity, model in enumerate(capacity_models(cfg)):
            for lr in LEARNING_RATES:
                trial = copy.deepcopy(cfg)
                trial["model"] = model
                trial["training"].update(seed=seed, lr=lr)
                run_name = f"study2d_{tag}_{arch}_c{capacity}_lr{lr:.0e}_s{seed}"
                trial["experiment"].update(name=run_name, run_name=run_name)
                trial["experiment"]["description"] = "Validation-only tuning candidate; test data must not choose the winner."
                candidates.append((output_dir / f"{run_name}.yaml", trial))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, cfg in candidates:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest = {"selection_metric": "minimum validation composite loss", "seed": seed,
                "tag": tag, "trials_per_architecture": 6,
                "note": "Equal trial count is not equal GPU time; report both. These are candidates, not optimized results.",
                "configs": [str(path) for path, _ in candidates]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs/study_2d/sweep")
    parser.add_argument("--include-cnn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="tune01", help="Change for a new sweep to keep run names unique.")
    args = parser.parse_args()
    architectures = ARCHITECTURES + (("residual_cnn",) if args.include_cnn else ())
    manifest = generate(args.output_dir, architectures, args.seed, args.tag)
    print(f"Wrote {len(manifest['configs'])} candidate configs to {args.output_dir}")
