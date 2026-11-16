"""Synthetic-data and restart tests: python -m unittest test_scripts.test_batch_2d."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
import yaml

from utils.experiment_batch import (completed_result, preflight_configs, prepare_search,
                                    save_batch_reports)
from utils.training_2d import CorrectedLoss, load_train_validation, run_epoch, train
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    cfg = yaml.safe_load((ROOT / "configs/study_2d/mlp.yaml").read_text())
    cfg["experiment"].update(name="synthetic_batch", run_name="synthetic_batch")
    cfg["data"]["h5_path"] = "dns.h5"
    cfg["data"]["split"] = {"train_snapshots": 3, "val_snapshots": 2, "test_snapshots": 1}
    cfg["model"]["mlp"].update(hidden_dims=[8], dropout_p=0.)
    cfg["training"].update(epochs=2, batch_size=2, grad_accum_steps=2, mixed_precision=False,
                           torch_num_threads=1, selection_metric="val_mse")
    cfg["training"]["lr_scheduler"] = {"enabled": True, "type": "plateau", "patience": 1}
    cfg["training"]["early_stopping"] = {"enabled": True, "patience": 2, "min_epochs": 1}
    cfg["hardware"]["device"] = "cpu"
    rng = np.random.default_rng(9)
    with h5py.File(root / "dns.h5", "w") as h5:
        snapshots = h5.create_group("snapshots")
        for index in range(5):
            group = snapshots.create_group(f"snapshot_{index:04d}")
            for name in ("ux", "uy", "omega_z"):
                group[name] = rng.normal(size=(128, 128)).astype("float32")
        # This test sample intentionally has no fields. Any attempt to read
        # test arrays during training will fail immediately.
        snapshots.create_group("snapshot_0005").attrs["never_read"] = True
    return cfg


class PlanningTests(unittest.TestCase):
    def test_large_search_freezes_96_unique_configs_and_detects_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = yaml.safe_load((ROOT / "configs/study_2d/large_search.yaml").read_text())
            plan["search"]["output_dir"] = temp
            path = Path(temp).parent / (Path(temp).name + "_plan.yaml")
            self.addCleanup(path.unlink, missing_ok=True)
            path.write_text(yaml.safe_dump(plan))
            directory, manifest = prepare_search(path)
            self.assertEqual(len(manifest["trials"]), 96)
            self.assertEqual(len({r["run_name"] for r in manifest["trials"]}), 96)
            trials, metric = preflight_configs([t["config_path"] for t in manifest["trials"]])
            self.assertEqual(metric, "val_mse")
            self.assertEqual(len(trials), 96)
            self.assertFalse(manifest["test_evaluated"])
            self.assertEqual(prepare_search(path)[1], manifest)
            edited = Path(trials[0]["config_path"])
            cfg = yaml.safe_load(edited.read_text())
            cfg["training"]["lr"] = .123
            edited.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, "edited"):
                prepare_search(path)

    def test_mixed_selection_metrics_fail_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for i, metric in enumerate(("val_loss", "val_mse")):
                cfg = yaml.safe_load((ROOT / "configs/study_2d/edsr.yaml").read_text())
                cfg["training"]["selection_metric"] = metric
                cfg["experiment"]["run_name"] = f"test_{i}"
                path = Path(temp) / f"{i}.yaml"
                path.write_text(yaml.safe_dump(cfg))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "single selection metric"):
                preflight_configs(paths)

    def test_largest_declared_models_have_finite_sr_outputs(self):
        from models.factory import build_model_from_config
        plan = yaml.safe_load((ROOT / "configs/study_2d/large_search.yaml").read_text())
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for arch, specification in plan["models"].items():
                base = yaml.safe_load((ROOT / specification["base_config"]).read_text())
                base_model = build_model_from_config(base["model"], 32, 128)
                baseline_count = sum(p.numel() for p in base_model.parameters())
                del base_model
                counts = []
                for capacity in specification["capacities"]:
                    cfg = copy.deepcopy(base["model"])
                    cfg[arch].update(capacity)
                    model = build_model_from_config(cfg, 32, 128)
                    count = sum(p.numel() for p in model.parameters())
                    self.assertGreater(count, baseline_count)
                    counts.append((count, cfg))
                    del model
                count, cfg = max(counts, key=lambda item: item[0])
                with self.subTest(architecture=arch):
                    model = build_model_from_config(cfg, 32, 128).eval()
                    with torch.inference_mode():
                        out = model(torch.randn(1, 1, 32, 32))
                    self.assertEqual(tuple(out.shape), (1, 1, 128, 128))
                    self.assertTrue(torch.isfinite(out).all())
                    print(f"Large model smoke: {arch}: {count:,} parameters", flush=True)
                    del model, out
        finally:
            torch.set_num_threads(previous_threads)


class TrainingTests(unittest.TestCase):
    def test_actual_training_never_reads_test_and_completed_runs_are_verifiable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = fixture(root)
            path = root / "candidate.yaml"
            path.write_text(yaml.safe_dump(cfg))
            summary = train(cfg, root)
            self.assertFalse(summary["test_evaluated"])
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertTrue(np.isfinite(summary["best_val_mse"]))
            directory = root / "work/SR/synthetic_batch"
            self.assertTrue((directory / "checkpoints/best_model.pth").exists())
            self.assertTrue((directory / "plots/training_curves.png").exists())
            provenance = json.loads((directory / "metrics/data_provenance.json").read_text())
            self.assertFalse(provenance["test_arrays_read"])
            trials, _ = preflight_configs([path], root)
            result = completed_result(trials[0], root)
            self.assertEqual(result["selection_value"], summary["best_val_mse"])
            report = root / "report"
            report.mkdir()
            save_batch_reports(report, trials, {}, root)
            selected = json.loads((report / "selected_configs.json").read_text())
            self.assertFalse(selected["provisional"])
            self.assertEqual(selected["CONFIG_PATHS"], [str(path)])
            with self.assertRaises(FileExistsError):
                train(cfg, root)

    def test_checkpoint_selection_uses_mse_when_composite_loss_disagrees(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = fixture(root)
            def metrics(loss, pixel, samples):
                return {"loss": loss, "pixel": pixel, "gradient": 0.,
                        "advection": 0., "diffusion": 0., "samples": samples}
            phases = [metrics(.3, .3, 3), metrics(.1, 1., 2),
                      metrics(.2, .2, 3), metrics(.2, .9, 2)]
            with patch("utils.training_2d.run_epoch", side_effect=phases):
                summary = train(cfg, root)
            self.assertEqual(summary["best_epoch"], 2)
            self.assertEqual(summary["best_val_loss"], .2)

    def test_accumulation_matches_equivalent_batches_with_partial_last_group(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture(Path(temp))
            datasets, _ = load_train_validation(cfg, Path(temp))
            # Five samples split as (2+2),1 versus 4,1. No batch-dependent layers.
            samples = [datasets["train"][i % 3] for i in range(5)]
            torch.manual_seed(1)
            model = torch.nn.Sequential(torch.nn.Upsample(scale_factor=4, mode="nearest"),
                                        torch.nn.Conv2d(1, 1, 1))
            other = copy.deepcopy(model)
            loss = CorrectedLoss(cfg, torch.device("cpu"))
            for net, batch, accumulation in ((model, 2, 2), (other, 4, 1)):
                optimizer = torch.optim.SGD(net.parameters(), lr=.01)
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                run_epoch(net, DataLoader(samples, batch_size=batch), loss, torch.device("cpu"),
                          optimizer=optimizer, scaler=scaler, accumulation=accumulation, max_norm=None)
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


class ControllerTests(unittest.TestCase):
    def test_real_isolated_worker_process(self):
        from scripts.run_2d_experiments import run_worker
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = fixture(root)
            cfg["data"]["h5_path"] = str(root / "dns.h5")
            # Worker output goes to the actual workspace's ignored work folder;
            # its unique name cannot collide with real trained runs.
            name = "test_isolated_" + root.name
            cfg["experiment"].update(name=name, run_name=name)
            cfg["training"]["epochs"] = 1
            path = root / "worker.yaml"
            path.write_text(yaml.safe_dump(cfg))
            # Preserve the importing environment, including isolated test deps.
            env_path = os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p)
            with patch.dict(os.environ, {"PYTHONPATH": env_path}):
                code = run_worker(path, root / "worker.log", False)
            self.assertEqual(code, 0, (root / "worker.log").read_text())
            summary_path = ROOT / "work/SR" / name / "metrics/training_summary.json"
            summary = json.loads(summary_path.read_text())
            self.assertFalse(summary["test_evaluated"])
            self.assertEqual(summary["epochs_completed"], 1)

    def test_failure_continue_skip_complete_and_archive_retry(self):
        from scripts import run_2d_experiments as controller
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = fixture(root)
            cfg["data"]["h5_path"] = str(root / "dns.h5")
            cfg["training"]["epochs"] = 1
            candidates = root / "configs"
            candidates.mkdir()
            for index in range(2):
                candidate = copy.deepcopy(cfg)
                candidate["experiment"].update(name=f"trial_{index}", run_name=f"trial_{index}")
                (candidates / f"{index}.yaml").write_text(yaml.safe_dump(candidate))
            report = root / "report"
            failed_once = False
            calls = []
            def worker(path, log, verbose):
                nonlocal failed_once
                current = yaml.safe_load(Path(path).read_text())
                name = current["experiment"]["run_name"]
                calls.append(name)
                if name == "trial_0" and not failed_once:
                    failed_once = True
                    partial = root / "work/SR" / name
                    partial.mkdir(parents=True)
                    (partial / "config.yaml").write_text(yaml.safe_dump(current))
                    (partial / "keep_me.txt").write_text("failed attempt is preserved")
                    return 1
                train(current, root)
                return 0
            args = ["--configs-dir", str(candidates), "--report-dir", str(report)]
            with patch.object(controller, "ROOT", root), patch.object(controller, "run_worker", side_effect=worker):
                self.assertEqual(controller.main(args), 1)
                self.assertEqual(calls, ["trial_0", "trial_1"])
                self.assertEqual(controller.main(args), 1)
                self.assertEqual(calls, ["trial_0", "trial_1"])  # success skipped; failed folder preserved
                self.assertEqual(controller.main(args + ["--retry-failed"]), 0)
                self.assertEqual(calls, ["trial_0", "trial_1", "trial_0"])
                self.assertEqual(controller.main(args), 0)
                self.assertEqual(len(calls), 3)
            archives = list((report / "failed_attempts").glob("trial_0_*"))
            self.assertEqual(len(archives), 1)
            self.assertTrue((archives[0] / "keep_me.txt").is_file())
            selected = json.loads((report / "selected_configs.json").read_text())
            self.assertEqual(selected["completed"], 2)
            self.assertFalse(selected["provisional"])
            status = json.loads((report / "batch_status.json").read_text())
            self.assertEqual(len(status["trials"]["trial_0"]["attempts"]), 2)
            self.assertFalse((report / ".batch.lock").exists())


if __name__ == "__main__":
    unittest.main()
