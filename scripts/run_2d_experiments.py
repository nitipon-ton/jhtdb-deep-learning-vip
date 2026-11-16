"""One-command sequential experiment runner; no test evaluation.

python scripts/run_2d_experiments.py --plan configs/study_2d/large_search.yaml
python scripts/run_2d_experiments.py --configs-dir configs/study_2d/sweep_all

Each training run is a separate process, so GPU memory is released between
trials. Re-running skips verified completed runs. Incomplete runs are preserved;
use --retry-failed to archive them and restart those trials from epoch one.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from utils.experiment_batch import (completed_result, preflight_configs, prepare_search,
                                    save_batch_reports, write_json)


def archive_incomplete(run_dir, archive_dir):
    """Move a failed run intact; never delete weights or overwrite an archive."""
    run_dir, archive_dir = Path(run_dir).resolve(), Path(archive_dir).resolve()
    expected_parent = (ROOT / "work/SR").resolve()
    if run_dir.parent != expected_parent or run_dir == archive_dir or run_dir in archive_dir.parents:
        raise ValueError(f"Unsafe archive paths: {run_dir}, {archive_dir}")
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    if archive_dir.exists():
        raise FileExistsError(archive_dir)
    shutil.move(str(run_dir), str(archive_dir))


def run_worker(config_path, log_path, verbose=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(config_path)]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if verbose:
                    print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--plan", type=Path)
    source.add_argument("--configs-dir", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--report-dir", type=Path, help="For --configs-dir; defaults to work/SR/searches/<folder-name>.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare/validate configs and print the run list without training.")
    parser.add_argument("--retry-failed", action="store_true", help="Archive incomplete runs, then retrain from scratch.")
    parser.add_argument("--limit", type=int, help="Run at most N pending trials; useful for a pilot.")
    parser.add_argument("--verbose", action="store_true", help="Print worker epoch logs live (always written to disk).")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        # Heavy dependencies are only imported inside workers, not in planning.
        from utils.training_2d import train
        config = yaml.safe_load(args.worker.read_text(encoding="utf-8"))
        summary = train(config)
        print(json.dumps({k: v for k, v in summary.items() if k != "timing"}, indent=2), flush=True)
        return 0
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.configs_dir:
        paths = [p for p in sorted((ROOT / args.configs_dir).glob("*.yaml"))
                 if "model" in yaml.safe_load(p.read_text(encoding="utf-8"))]
        report_dir = (ROOT / (args.report_dir or Path("work/SR/searches") / args.configs_dir.name)).resolve()
    else:
        if args.report_dir:
            parser.error("--report-dir applies to --configs-dir; use the plan's output_dir with --plan")
        plan_path = ROOT / (args.plan or Path("configs/study_2d/large_search.yaml"))
        report_dir, manifest = prepare_search(plan_path, root=ROOT)
        paths = [item["config_path"] for item in manifest["trials"]]
    trials, metric = preflight_configs(paths, root=ROOT)
    report_dir.mkdir(parents=True, exist_ok=True)
    # Lock atomically; stale locks are preserved for explicit operator inspection.
    lock = report_dir / ".batch.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "host": os.environ.get("HOSTNAME", os.environ.get("COMPUTERNAME")),
                                     "started_utc": datetime.now(timezone.utc).isoformat()}))
    except FileExistsError:
        raise RuntimeError(f"Batch lock exists: {lock}. Check whether that process is running before removing a stale lock.")
    try:
        state_path = report_dir / "batch_status.json"
        state = json.loads(state_path.read_text(encoding="utf-8")).get("trials", {}) if state_path.exists() else {}
        completed = {}
        for trial in trials:
            result = completed_result(trial, root=ROOT)
            if result:
                completed[trial["run_name"]] = result
                state.setdefault(trial["run_name"], {})["status"] = "completed"
        print(f"Candidates={len(trials)}, completed={len(completed)}, selection={metric}", flush=True)
        print("Reports:", report_dir, flush=True)
        if args.dry_run:
            for trial in trials:
                print("SKIP" if trial["run_name"] in completed else "PLAN", trial["run_name"])
            return 0
        if len(completed) == len(trials):
            save_batch_reports(report_dir, trials, state, root=ROOT, completed=completed)
            print("All candidates already complete; refreshed reports without requiring a GPU.", flush=True)
            return 0
        # Check imports and requested accelerators before launching a long queue.
        import torch
        import h5py  # noqa: F401
        import matplotlib  # noqa: F401
        from models.factory import available_models
        for trial in trials:
            if trial["run_name"] in completed:
                continue
            cfg = yaml.safe_load(Path(trial["config_path"]).read_text(encoding="utf-8"))
            if cfg["model"]["architecture"] not in available_models():
                raise ValueError(f"Unregistered architecture: {trial['architecture']}")
            if str(cfg["hardware"]["device"]).startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable. Run inside a GPU allocation with the project environment.")
            data_path = ROOT / cfg["data"]["h5_path"]
            if not data_path.is_file():
                raise FileNotFoundError(data_path)
        launched = failures = 0
        for index, trial in enumerate(trials, 1):
            name = trial["run_name"]
            if name in completed:
                print(f"[{index}/{len(trials)}] SKIP verified complete: {name}", flush=True)
                continue
            if args.limit is not None and launched >= args.limit:
                break
            directory = ROOT / "work/SR" / name
            if directory.exists() and any(directory.iterdir()):
                if not args.retry_failed:
                    state.setdefault(name, {}).update(
                        status="incomplete_preserved", message="Use --retry-failed to archive/restart.")
                    print(f"[{index}/{len(trials)}] PRESERVE incomplete: {name}; use --retry-failed", flush=True)
                    failures += 1
                    if args.fail_fast:
                        break
                    continue
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                archive_incomplete(directory, report_dir / "failed_attempts" / f"{name}_{stamp}")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            log = report_dir / "logs" / f"{name}_{stamp}.log"
            print(f"[{index}/{len(trials)}] TRAIN {name}; log={log}", flush=True)
            started = time.perf_counter()
            attempts = state.get(name, {}).get("attempts", [])
            attempt = {"status": "running", "log": str(log), "started_utc": stamp}
            attempts.append(attempt)
            state[name] = {"status": "running", "attempts": attempts}
            write_json(state_path, {"trials": state, "test_evaluated": False})
            try:
                code = run_worker(trial["config_path"], log, args.verbose)
            except KeyboardInterrupt:
                attempt.update(status="interrupted", worker_wall_seconds=time.perf_counter() - started)
                state[name]["status"] = "interrupted"
                save_batch_reports(report_dir, trials, state, root=ROOT, completed=completed)
                print("Interrupted. Completed runs are retained; no epoch-level resume is claimed.", flush=True)
                return 130
            except Exception as error:
                code = 1
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(f"\nWorker launch/controller error: {type(error).__name__}: {error}\n")
            launched += 1
            attempt.update(returncode=code, worker_wall_seconds=time.perf_counter() - started)
            result = completed_result(trial, root=ROOT) if code == 0 else None
            if result:
                completed[name] = result
                attempt["status"] = state[name]["status"] = "completed"
                print(f"  DONE {metric}={result['selection_value']:.6f}, "
                      f"epochs={result['epochs_completed']}, train_s={result['training_total_seconds']:.1f}", flush=True)
            else:
                failures += 1
                attempt["status"] = state[name]["status"] = "failed"
                print(f"  FAILED exit={code}; inspect {log}. Continuing without changing batch size/settings.", flush=True)
            save_batch_reports(report_dir, trials, state, root=ROOT, completed=completed)
            if failures and args.fail_fast:
                break
        results = save_batch_reports(report_dir, trials, state, root=ROOT, completed=completed)
        print(f"Finished: {len(results)}/{len(trials)} complete; failures/preserved={failures}. No test evaluation performed.", flush=True)
        print("Validation results:", report_dir / "validation_results.csv", flush=True)
        print("Selected config paths:", report_dir / "selected_configs.json", flush=True)
        return 1 if failures else 0
    finally:
        lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
