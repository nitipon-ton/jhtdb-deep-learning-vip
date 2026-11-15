"""Run: python -m unittest test_scripts.test_2d_study.

Timer tests need only stdlib. Config/sweep checks need PyYAML. Model smoke
checks require the full project environment and run without DNS files or GPU.
"""
import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_2d_validation import collect
from utils.training_timing import TrainingTimer, file_sha256, read_training_timing, timing_columns

HAS_YAML = importlib.util.find_spec("yaml") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value


class TimingTests(unittest.TestCase):
    def test_timings_survive_interruption_then_bind_to_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            clock = Clock()
            synchronization_calls = []
            timer = TrainingTimer(run, "run", {"device_names": ["test"]},
                                  synchronize=lambda: synchronization_calls.append(True), clock=clock)
            timer.start_epoch()
            clock.value = 10
            timer.end_train()
            clock.value = 13
            timer.end_validation()
            clock.value = 15
            timer.end_epoch(1, 1., .5)
            partial = json.loads(timer.path.read_text())
            self.assertFalse(partial["completed"])
            self.assertEqual(partial["per_epoch"][0]["train_seconds"], 10)
            self.assertEqual(partial["per_epoch"][0]["validation_seconds"], 3)
            self.assertEqual(partial["per_epoch"][0]["epoch_seconds"], 15)
            self.assertEqual(collect(Path(directory)), [])
            checkpoint = run / "best_model.pth"
            checkpoint.write_bytes(b"best-validation-weights")
            clock.value = 17
            report = timer.finish(1, checkpoint)
            self.assertTrue(report["completed"])
            self.assertEqual(report["total_training_seconds"], 17)
            self.assertEqual(report["time_to_best_seconds"], 15)
            self.assertEqual(len(collect(Path(directory))), 1)
            with timer.csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["epoch"], "1")
            loaded = read_training_timing(run, "run", file_sha256(checkpoint))
            self.assertEqual(timing_columns(loaded)["train_seconds_per_epoch"], 10)
            checkpoint.write_bytes(b"different-weights")
            with self.assertRaisesRegex(ValueError, "different best weights"):
                read_training_timing(run, "run", file_sha256(checkpoint))
            self.assertGreaterEqual(len(synchronization_calls), 5)

    def test_missing_times_are_not_zero_and_bicubic_is_not_trained(self):
        missing = timing_columns()
        self.assertIsNone(missing["training_total_seconds"])
        baseline = timing_columns(baseline=True)
        self.assertEqual(baseline["training_total_seconds"], 0)
        self.assertEqual(baseline["timing_status"], "not_applicable")

    def test_nonfinite_losses_do_not_complete_the_report(self):
        with tempfile.TemporaryDirectory() as directory:
            timer = TrainingTimer(directory, "test", {}, clock=Clock())
            timer.start_epoch()
            timer.end_train()
            timer.end_validation()
            with self.assertRaises(ValueError):
                timer.end_epoch(1, 1., float("nan"))
            self.assertFalse(json.loads(timer.path.read_text())["completed"])


@unittest.skipUnless(HAS_YAML, "PyYAML required for study config checks")
class StudyConfigTests(unittest.TestCase):
    def test_configs_cover_2d_registry_and_keep_objective_fixed(self):
        import ast
        import yaml
        registry = ast.parse((ROOT / "models/factory.py").read_text())
        registered = set()
        for node in ast.walk(registry):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register_model":
                registered.add(node.args[0].value)
        configs = [yaml.safe_load(p.read_text()) for p in (ROOT / "configs/study_2d").glob("*.yaml")]
        self.assertEqual({c["model"]["architecture"] for c in configs}, registered - {"unet_meteo_3d"})
        reference = yaml.safe_load((ROOT / "configs/vip_unet_corrected_physics_128.yaml").read_text())
        names = []
        for cfg in configs:
            for key in ("data", "domain", "loss", "evaluation", "hardware"):
                self.assertEqual(cfg[key], reference[key])
            self.assertTrue(cfg["training"]["reject_existing_run"])
            self.assertFalse(cfg["training"]["gan"]["enabled"])
            self.assertEqual(cfg["training"]["lr_scheduler"]["t_max"], cfg["training"]["epochs"])
            names.append(cfg["experiment"]["run_name"])
        self.assertEqual(len(names), len(set(names)))

    def test_sweep_declares_six_unique_candidates_per_family(self):
        import yaml
        from scripts.prepare_2d_sweep import generate, ARCHITECTURES
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sweep"
            manifest = generate(output)
            self.assertEqual(len(manifest["configs"]), 42)
            configs = [yaml.safe_load(Path(p).read_text()) for p in manifest["configs"]]
            names = [c["experiment"]["run_name"] for c in configs]
            self.assertEqual(len(names), len(set(names)))
            for arch in ARCHITECTURES:
                self.assertEqual(sum(c["model"]["architecture"] == arch for c in configs), 6)
            with self.assertRaises(FileExistsError):
                generate(output)
            all_manifest = generate(Path(directory) / "all", ARCHITECTURES + ("residual_cnn",), tag="all01")
            self.assertEqual(len(all_manifest["configs"]), 48)


@unittest.skipUnless(HAS_YAML and HAS_TORCH, "Full project environment required for forward/backward checks")
class ModelSmokeTests(unittest.TestCase):
    def test_every_candidate_capacity_accepts_32_and_outputs_128_with_gradients(self):
        import torch
        import yaml
        from models.factory import build_model_from_config
        from scripts.prepare_2d_sweep import capacity_models
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for path in sorted((ROOT / "configs/study_2d").glob("*.yaml")):
                cfg = yaml.safe_load(path.read_text())
                for model_cfg in capacity_models(cfg):
                    with self.subTest(architecture=model_cfg["architecture"], model=model_cfg):
                        model = build_model_from_config(copy.deepcopy(model_cfg), 32, 128).train()
                        out = model(torch.randn(1, 1, 32, 32))
                        self.assertEqual(tuple(out.shape), (1, 1, 128, 128))
                        self.assertTrue(torch.isfinite(out).all())
                        out.square().mean().backward()
                        grads = [p.grad for p in model.parameters() if p.grad is not None]
                        self.assertTrue(grads)
                        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                        del model, out, grads
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
