#!/usr/bin/env python
"""Run the frozen six-fold R006V0F diagnostic without exposing partial metrics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--planner-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--nested-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    if len(args.source_commit) != 40:
        raise ValueError("source commit must be a full SHA")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    execution = config.get("execution", {})
    gpu_roster = [int(value) for value in execution.get("gpu_roster", [])]
    if gpu_roster != [0, 1, 6, 7]:
        raise ValueError("formal R006V0F GPU roster changed")
    if execution.get("max_concurrent_fold_chains") != 4:
        raise ValueError("formal concurrency changed")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError("output root must be absent or empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.json"
    lock = threading.Lock()
    state: dict[str, Any] = {
        "status": "RUNNING",
        "phase": "primary_replay_audit",
        "completed_fold_count": 0,
        "total_fold_count": 6,
        "running_fold_count": 0,
        "failed_fold_count": 0,
        "gpu_roster": gpu_roster,
        "development_calibration_test_navdream_target_accessed": False,
        "scientific_metrics_emitted": False,
    }
    _write_json(progress_path, state)

    def update(**values: Any) -> None:
        with lock:
            state.update(values)
            _write_json(progress_path, state)

    def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=args.source_root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"command failed with exit {completed.returncode}: {log_path}"
            )

    def run_fold(fold: int, gpu: int) -> tuple[int, str]:
        fold_root = args.output_root / f"fold_{fold}"
        fold_root.mkdir()
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(args.source_root / "src")
        with lock:
            state["running_fold_count"] += 1
            _write_json(progress_path, state)
        try:
            common = [
                str(args.python),
                str(args.source_root / "scripts/run_r006v0f_fold.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--planner-root",
                str(args.planner_root),
                "--cache-dir",
                str(args.cache_dir),
                "--nested-manifest",
                str(args.nested_manifest),
                "--config",
                str(args.config),
                "--outer-fold",
                str(fold),
                "--source-commit",
                args.source_commit,
                "--device",
                "cuda:0",
            ]
            primary = fold_root / "primary"
            replay = fold_root / "replay"
            run_command(
                [*common, "--output-dir", str(primary)],
                fold_root / "primary.log",
                env,
            )
            run_command(
                [*common, "--output-dir", str(replay)],
                fold_root / "replay.log",
                env,
            )
            audit_command = [
                str(args.python),
                str(args.source_root / "scripts/audit_r006v0f_fold.py"),
                "--result-dir",
                str(primary),
                "--replay-dir",
                str(replay),
                "--nested-manifest",
                str(args.nested_manifest),
                "--config",
                str(args.config),
                "--checkpoint",
                str(args.checkpoint),
                "--cache-dir",
                str(args.cache_dir),
                "--source-root",
                str(args.source_root),
                "--expected-source-commit",
                args.source_commit,
                "--output",
                str(fold_root / "audit.json"),
            ]
            run_command(audit_command, fold_root / "audit.log", env)
            return fold, "PASS"
        except Exception as error:
            (fold_root / "failure.txt").write_text(
                f"{type(error).__name__}: {error}\n", encoding="utf-8"
            )
            return fold, "FAIL"
        finally:
            with lock:
                state["running_fold_count"] -= 1
                _write_json(progress_path, state)

    assignments: dict[int, list[int]] = {gpu: [] for gpu in gpu_roster}
    for fold in range(6):
        assignments[gpu_roster[fold % len(gpu_roster)]].append(fold)

    def run_gpu_queue(gpu: int) -> list[tuple[int, str]]:
        return [run_fold(fold, gpu) for fold in assignments[gpu]]

    outcomes: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_gpu_queue, gpu) for gpu in gpu_roster]
        for future in as_completed(futures):
            rows = future.result()
            outcomes.extend(rows)
            update(
                completed_fold_count=sum(status == "PASS" for _, status in outcomes),
                failed_fold_count=sum(status == "FAIL" for _, status in outcomes),
            )

    outcomes.sort()
    passed = all(status == "PASS" for _, status in outcomes) and len(outcomes) == 6
    update(
        status="COMPLETE" if passed else "FAILED",
        phase="complete" if passed else "failed",
        completed_fold_count=sum(status == "PASS" for _, status in outcomes),
        failed_fold_count=sum(status == "FAIL" for _, status in outcomes),
        scientific_metrics_emitted=passed,
        outcomes=[{"outer_fold": fold, "status": status} for fold, status in outcomes],
    )
    (args.output_root / "exit_code.txt").write_text(
        "0\n" if passed else "2\n", encoding="utf-8"
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
