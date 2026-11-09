"""Regression checks: python -m unittest test_scripts.test_sr_comparison.

Run in the project's jhtdb-ml environment (no DNS files or trained weights needed).
"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from utils.sr_comparison import evaluate, physical_terms, prepare_runs, sample_metrics


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.config = yaml.safe_load((self.root / "configs/vip_unet_corrected_physics_128.yaml").read_text())
        self.metadata = {"normalization_stats": {"omega_z": (2., 3.), "ux": (0., 1.), "uy": (0., 1.)},
                         "data_range": 20., "dx": .1, "dy": .1}

    def test_best_checkpoint_required_even_when_last_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.yaml").write_text(yaml.safe_dump(self.config))
            checkpoint_dir = root / "work/SR" / self.config["experiment"]["run_name"] / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "last_model.pth").write_bytes(b"last")
            with self.assertRaisesRegex(FileNotFoundError, "No fallback"):
                prepare_runs(root, ["config.yaml"])
            (checkpoint_dir / "best_model.pth").write_bytes(b"best")
            runs = prepare_runs(root, ["config.yaml"])
            self.assertEqual(Path(runs[0]["checkpoint"]).name, "best_model.pth")
            altered = copy.deepcopy(self.config)
            altered["data"]["split"]["test_snapshots"] = 399
            (root / "altered.yaml").write_text(yaml.safe_dump(altered))
            with self.assertRaisesRegex(ValueError, "3000/600/400"):
                prepare_runs(root, ["altered.yaml"])

    def test_identity_and_prediction_independent_range(self):
        torch.manual_seed(7)
        target = torch.randn(1, 1, 128, 128)
        u = torch.ones(1, 128, 128)
        identical = sample_metrics(target, target, u, u, self.metadata)
        for key in ("mse", "mae", "advection_error", "diffusion_error",
                    "physical_advection_mse", "physical_diffusion_mse"):
            self.assertEqual(identical[key], 0.)
        self.assertAlmostEqual(identical["ssim"], 1., places=6)
        noisy = sample_metrics(target + 1., target, u, u, self.metadata)
        self.assertAlmostEqual(noisy["mse"], 9., places=5)
        self.assertAlmostEqual(noisy["psnr"], 10 * np.log10(400 / 9), places=5)

    def test_physical_stencils_and_axis_convention(self):
        y, x = torch.meshgrid(torch.arange(8, dtype=torch.float64),
                              torch.arange(8, dtype=torch.float64), indexing="ij")
        omega = (x.square() + 3 * y.square())[None, None]
        u, v = torch.ones_like(omega), torch.zeros_like(omega)
        adv, lap = physical_terms(omega, u, v, 1., 1.)
        torch.testing.assert_close(adv[0, 0], 2 * x[1:-1, 1:-1])
        torch.testing.assert_close(lap, torch.full_like(lap, 8.))

    def test_all_400_samples_in_order_for_baseline(self):
        # Exercise the real evaluator without 400 expensive SSIM convolutions.
        sample = {"lr_omega": torch.zeros(1, 32, 32), "hr_omega": torch.zeros(128, 128),
                  "hr_ux": torch.zeros(128, 128), "hr_uy": torch.zeros(128, 128)}
        metadata = {**self.metadata, "test_indices": list(range(3600, 4000)),
                    "test_snapshot_keys": [f"snapshot_{i}" for i in range(3600, 4000)]}
        scores = {k: 0. for k in ("mse", "mae", "psnr", "ssim", "advection_error",
                                  "diffusion_error", "physical_advection_mse", "physical_diffusion_mse")}
        with patch("utils.sr_comparison.sample_metrics", return_value=scores) as metric:
            rows, summary, examples = evaluate([], [sample] * 400, metadata, device="cpu", warmup=1)
        self.assertEqual(metric.call_count, 400)
        self.assertEqual([r["snapshot_index"] for r in rows], list(range(3600, 4000)))
        self.assertEqual(summary[0]["parameter_count"], 0)
        self.assertEqual(summary[0]["num_test_samples"], 400)
        self.assertEqual(set(examples), {0, 199, 399})


if __name__ == "__main__":
    unittest.main()
