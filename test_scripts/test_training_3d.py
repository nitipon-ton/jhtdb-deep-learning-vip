"""Smoke and unit tests for the 3D velocity-cube training stack.

Run in the project's ``jhtdb-ml`` environment::

    python -m unittest test_scripts.test_training_3d

The tests build tiny synthetic cubes in a temporary directory, so no DNS data
or trained weights are required.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.SR.unet_meteo_3d import MeteorologyUNetSR3D
from models.factory import available_models
from utils.cube_datasets import (
    VelocityCubeDataset,
    compute_component_stats,
    discover_pairs,
)
from utils.losses import GradientLoss3D
from utils.training_3d import (
    CompositeLoss3D,
    build_dataloaders,
    build_loss,
    build_optimizer,
    evaluate,
    load_config,
    run_training,
    train_one_epoch,
    validate,
)

CUBE = 32


def _write_split(root: Path, split: str, count: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        inputs = rng.normal(size=(CUBE, CUBE, CUBE, 3)).astype(np.float32)
        targets = rng.normal(scale=0.1, size=(CUBE, CUBE, CUBE, 3)).astype(
            np.float32
        )
        np.save(split_dir / f"input_t{index:04d}.npy", inputs)
        np.save(split_dir / f"target_t{index:04d}.npy", targets)


def _tiny_model() -> MeteorologyUNetSR3D:
    return MeteorologyUNetSR3D(
        in_channels=3,
        out_channels=3,
        scale_factor=1,
        feature_channels=[4, 8, 8, 8, 8],
        n_blocks_per_stage=1,
        num_latent_layers=1,
        normalization="group",
        group_norm_groups=4,
        residual=False,
    )


def _config(root: Path) -> dict:
    return {
        "experiment": {
            "name": "test_3d",
            "run_name": "test_3d",
            "base_dir": str(root / "work"),
        },
        "data": {
            "processed_dir": str(root / "processed"),
            "splits": {"train": "train", "val": "val", "test": "test"},
            "normalization": "none",
            "stats": None,
            "augmentation": False,
        },
        "model": {
            "architecture": "unet_meteo_3d",
            "scale_factor": 1,
            "unet_meteo_3d": {
                "in_ch": 3,
                "out_ch": 3,
                "feature_channels": [4, 8, 8, 8, 8],
                "n_blocks_per_stage": 1,
                "num_latent_layers": 1,
                "norm": "group",
                "group_norm_groups": 4,
                "residual": False,
            },
        },
        "training": {
            "seed": 0,
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "optimizer": "adamw",
            "lr": 1.0e-3,
            "mixed_precision": False,
            "gradient_clipping": {"enabled": True, "max_norm": 1.0},
        },
        "loss": {
            "components": {
                "pixel": {"enabled": True, "weight": 1.0},
                "gradient": {"enabled": True, "weight": 0.1, "periodic": True},
            }
        },
        "evaluation": {},
        "hardware": {"device": "cpu"},
    }


class GradientLoss3DTests(unittest.TestCase):
    def test_zero_for_identical_fields(self):
        loss = GradientLoss3D(periodic=True)
        field = torch.randn(2, 3, 8, 8, 8)
        self.assertAlmostEqual(float(loss(field, field)), 0.0, places=6)

    def test_periodic_and_forward_differences_match_manual_definition(self):
        pred = torch.randn(2, 3, 8, 8, 8)
        target = torch.randn(2, 3, 8, 8, 8)

        manual = pred.new_zeros(())
        for dim in (-3, -2, -1):
            pred_diff = torch.roll(pred, shifts=-1, dims=dim) - pred
            target_diff = torch.roll(target, shifts=-1, dims=dim) - target
            manual = manual + torch.mean((pred_diff - target_diff) ** 2)
        self.assertAlmostEqual(
            float(GradientLoss3D(periodic=True)(pred, target)),
            float(manual),
            places=5,
        )

        manual = pred.new_zeros(())
        for axis in (2, 3, 4):
            pred_diff = pred.narrow(axis, 1, pred.size(axis) - 1) - pred.narrow(
                axis, 0, pred.size(axis) - 1
            )
            target_diff = target.narrow(
                axis, 1, target.size(axis) - 1
            ) - target.narrow(axis, 0, target.size(axis) - 1)
            manual = manual + torch.mean((pred_diff - target_diff) ** 2)
        self.assertAlmostEqual(
            float(GradientLoss3D(periodic=False)(pred, target)),
            float(manual),
            places=5,
        )

        # A single shifted x-plane makes the wrap contribution explicit:
        # non-periodic forward differences see one jump, periodic ones see two.
        pred = torch.zeros(1, 1, 4, 4, 8)
        target = torch.zeros(1, 1, 4, 4, 8)
        pred[..., 0] = 1.0
        self.assertAlmostEqual(
            float(GradientLoss3D(periodic=False)(pred, target)),
            1.0 / 7.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(GradientLoss3D(periodic=True)(pred, target)),
            2.0 / 8.0,
            places=6,
        )

    def test_rejects_non_5d_input(self):
        with self.assertRaises(ValueError):
            GradientLoss3D()(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8))


class CubeDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _write_split(self.root, "train", 3, seed=1)
        _write_split(self.root, "val", 2, seed=2)
        _write_split(self.root, "test", 1, seed=3)

    def tearDown(self):
        self.temp.cleanup()

    def test_discover_pairs_sorted(self):
        pairs = discover_pairs(self.root, "train")
        self.assertEqual([ts for ts, _, _ in pairs], ["0000", "0001", "0002"])
        for _, input_path, target_path in pairs:
            self.assertTrue(input_path.is_file())
            self.assertTrue(target_path.is_file())

    def test_missing_target_is_reported(self):
        (self.root / "train" / "input_t0003.npy").write_bytes(b"")
        with self.assertRaises(ValueError):
            discover_pairs(self.root, "train")

    def test_dataset_shape_and_dtype(self):
        dataset = VelocityCubeDataset(self.root, "train")
        inputs, targets = dataset[0]
        self.assertEqual(tuple(inputs.shape), (3, CUBE, CUBE, CUBE))
        self.assertEqual(tuple(targets.shape), (3, CUBE, CUBE, CUBE))
        self.assertEqual(inputs.dtype, torch.float32)

    def test_augmentation_preserves_shapes(self):
        torch.manual_seed(0)
        dataset = VelocityCubeDataset(self.root, "train", augment=True)
        inputs, targets = dataset[0]
        self.assertEqual(tuple(inputs.shape), (3, CUBE, CUBE, CUBE))
        self.assertEqual(tuple(targets.shape), (3, CUBE, CUBE, CUBE))

    def test_global_normalization_uses_stats(self):
        stats = {
            "input": {"u": (0.0, 1.0), "v": (0.0, 2.0), "w": (0.0, 4.0)},
            "target": {"u": (0.0, 1.0), "v": (0.0, 1.0), "w": (0.0, 1.0)},
        }
        dataset = VelocityCubeDataset(
            self.root, "train", normalize="global", stats=stats
        )
        inputs, _ = dataset[0]
        self.assertTrue(torch.isfinite(inputs).all())

    def test_component_stats(self):
        stats = compute_component_stats(self.root, "train", source="target")
        self.assertEqual(set(stats), {"u", "v", "w"})
        for mean, std in stats.values():
            self.assertGreaterEqual(std, 0.0)
            self.assertTrue(np.isfinite(mean))


class ModelTests(unittest.TestCase):
    def test_registered_in_factory(self):
        self.assertIn("unet_meteo_3d", available_models())

    def test_forward_shape_full_resolution(self):
        model = _tiny_model()
        x = torch.randn(2, 3, CUBE, CUBE, CUBE)
        self.assertEqual(tuple(model(x).shape), tuple(x.shape))

    def test_forward_shape_upsampling(self):
        model = MeteorologyUNetSR3D(
            in_channels=3,
            out_channels=3,
            scale_factor=2,
            feature_channels=[4, 8, 8, 8, 8],
            n_blocks_per_stage=1,
            num_latent_layers=0,
            normalization="group",
            group_norm_groups=4,
        )
        x = torch.randn(1, 3, 32, 32, 32)
        self.assertEqual(tuple(model(x).shape), (1, 3, 64, 64, 64))

    def test_residual_adds_input(self):
        model = _tiny_model()
        model.residual = True
        x = torch.randn(1, 3, CUBE, CUBE, CUBE)
        with torch.no_grad():
            self.assertEqual(tuple(model(x).shape), tuple(x.shape))


class LossTests(unittest.TestCase):
    def test_composite_total(self):
        criterion = CompositeLoss3D(pixel_weight=1.0, gradient_weight=0.1)
        prediction = torch.randn(1, 3, 8, 8, 8)
        target = torch.randn(1, 3, 8, 8, 8)
        components = criterion(prediction, target)
        self.assertEqual(set(components), {"total", "pixel", "gradient"})
        expected = components["pixel"] + 0.1 * components["gradient"]
        self.assertAlmostEqual(
            float(components["total"]), float(expected), places=5
        )

    def test_build_loss_reads_config(self):
        cfg = {"loss": {"components": {
            "pixel": {"enabled": True, "weight": 2.0},
            "gradient": {"enabled": True, "weight": 0.25, "periodic": True},
        }}}
        criterion = build_loss(cfg)
        self.assertEqual(criterion.pixel_weight, 2.0)
        self.assertEqual(criterion.gradient_weight, 0.25)
        self.assertTrue(criterion.gradient.periodic)


class TrainingLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _write_split(self.root / "processed", "train", 2, seed=10)
        _write_split(self.root / "processed", "val", 1, seed=11)
        _write_split(self.root / "processed", "test", 1, seed=12)
        self.cfg = _config(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_train_and_validate(self):
        loaders = build_dataloaders(
            self.cfg, splits=("train", "val"), shuffle=False
        )
        model = _tiny_model()
        optimizer = build_optimizer(self.cfg, model)
        criterion = build_loss(self.cfg)
        device = torch.device("cpu")

        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            cfg=self.cfg,
        )
        val_metrics = validate(
            model, loaders["val"], criterion, device, cfg=self.cfg
        )
        for metrics in (train_metrics, val_metrics):
            self.assertIn("total", metrics)
            self.assertIn("pixel", metrics)
            self.assertIn("gradient", metrics)
            self.assertGreater(metrics["n_samples"], 0.0)
            self.assertTrue(np.isfinite(metrics["total"]))

    def test_run_training_and_evaluate(self):
        result = run_training(self.cfg, progress=False)
        self.assertTrue(result.checkpoint_path.is_file())
        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(len(result.history["val_loss"]), 1)

        metrics = evaluate(self.cfg, split="test", save_predictions=True)
        self.assertEqual(metrics["num_samples"], 1)
        self.assertIn("correction_mse", metrics)
        self.assertIn("reconstruction_rel_l2", metrics)
        self.assertEqual(set(metrics["per_component_mse"]), {"u", "v", "w"})
        prediction = (
            result.run_dir / "predictions" / "prediction_t0000.npy"
        )
        self.assertTrue(prediction.is_file())


class ConfigTests(unittest.TestCase):
    def test_override_existing_key(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(
            root / "configs/sr_unet_meteo_3d_velocity128.yaml",
            ["training.epochs=2", "training.batch_size=2"],
        )
        self.assertEqual(cfg["training"]["epochs"], 2)
        self.assertEqual(cfg["training"]["batch_size"], 2)
        self.assertTrue(Path(cfg["data"]["processed_dir"]).is_absolute())

    def test_unknown_override_key_fails(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(KeyError):
            load_config(
                root / "configs/sr_unet_meteo_3d_velocity128.yaml",
                ["training.not_a_key=1"],
            )


if __name__ == "__main__":
    unittest.main()
