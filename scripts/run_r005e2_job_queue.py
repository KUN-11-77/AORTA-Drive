"""Four-GPU, fail-closed scheduler for all frozen R005E2 jobs and replay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARMS = ("V0_removed", "V1_mismatch", "V2_correct")
SEEDS = (20261025, 20261026, 20261027)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _jobs() -> list[dict[str, Any]]:
    return [
        {
            "outer_fold": fold,
            "arm": arm,
            "seed": seed,
            "name": f"fold{fold}_{arm}_seed{seed}",
        }
        for fold in range(6)
        for arm in ARMS
        for seed in SEEDS
    ]


def _gpu_memory() -> dict[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        index, used = (part.strip() for part in line.split(","))
        result[int(index)] = int(used)
    return result


def _complete(path: Path) -> bool:
    result_path = path / "result.json"
    if not result_path.is_file():
        return False
    result = json.loads(result_path.read_text("utf-8"))
    return result.get("status") == "COMPLETED" and (path / "predictions.npz").is_file()


def _command(args: argparse.Namespace, job: dict[str, Any], output: Path) -> list[str]:
    return [
        sys.executable,
        str(args.runner),
        "--bundle-root",
        str(args.bundle_root),
        "--visual-root",
        str(args.visual_root),
        "--crossfit-manifest",
        str(args.crossfit_manifest),
        "--mismatch-manifest",
        str(args.mismatch_manifest),
        "--outer-fold",
        str(job["outer_fold"]),
        "--arm",
        str(job["arm"]),
        "--seed",
        str(job["seed"]),
        "--workers",
        str(args.workers),
        "--code-commit",
        args.code_commit,
        "--physical-gpu",
        str(job["physical_gpu"]),
        "--output-directory",
        str(output),
    ]


def _run_phase(
    *,
    args: argparse.Namespace,
    phase: str,
    assignment: dict[str, int],
) -> None:
    phase_root = args.output_root / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    jobs = _jobs()
    pending = [job for job in jobs if not _complete(phase_root / str(job["name"]))]
    for job in pending:
        path = phase_root / str(job["name"])
        if path.exists():
            raise RuntimeError(f"partial job requires manual adjudication: {path}")
    running: dict[int, tuple[subprocess.Popen[bytes], dict[str, Any], Any]] = {}
    completed_count = len(jobs) - len(pending)
    while pending or running:
        failed: list[tuple[str, int]] = []
        for gpu, (process, job, log_handle) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del running[gpu]
            if return_code != 0 or not _complete(phase_root / str(job["name"])):
                failed.append((str(job["name"]), int(return_code or -1)))
            else:
                completed_count += 1
        if failed:
            _write_json(
                args.output_root / "progress.json",
                {
                    "status": "FAILED",
                    "phase": phase,
                    "completed_job_count": completed_count,
                    "total_job_count": 54,
                    "failures": failed,
                    "development_calibration_test_navdream_target_accessed": False,
                    "scientific_metrics_emitted": False,
                },
            )
            raise RuntimeError(f"R005E2 job failure: {failed}")
        memory = _gpu_memory()
        for gpu in args.gpus:
            if gpu in running or memory.get(gpu, 10**9) > args.maximum_idle_memory_mib:
                continue
            selected_index = next(
                (
                    index
                    for index, job in enumerate(pending)
                    if phase == "primary" or assignment[str(job["name"])] == gpu
                ),
                None,
            )
            if selected_index is None:
                continue
            job = pending.pop(selected_index)
            job["physical_gpu"] = gpu
            if phase == "primary":
                assignment[str(job["name"])] = gpu
                _write_json(args.output_root / "gpu_assignment.json", assignment)
            output = phase_root / str(job["name"])
            log_path = phase_root / f"{job['name']}.log"
            log_handle = log_path.open("wb")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            process = subprocess.Popen(
                _command(args, job, output),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            running[gpu] = (process, job, log_handle)
        _write_json(
            args.output_root / "progress.json",
            {
                "status": "IN_PROGRESS",
                "phase": phase,
                "completed_job_count": completed_count,
                "total_job_count": 54,
                "running_job_count": len(running),
                "development_calibration_test_navdream_target_accessed": False,
                "scientific_metrics_emitted": False,
            },
        )
        if pending or running:
            time.sleep(args.poll_seconds)


def run(args: argparse.Namespace) -> None:
    if sorted(set(args.gpus)) != sorted(args.gpus) or not args.gpus:
        raise ValueError("GPU roster must be nonempty and unique")
    if args.output_root.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    initial_memory = _gpu_memory()
    blocked = {
        gpu: initial_memory.get(gpu, 10**9)
        for gpu in args.gpus
        if initial_memory.get(gpu, 10**9) > args.maximum_idle_memory_mib
    }
    if blocked:
        raise RuntimeError(f"registered GPUs are not idle at launch: {blocked}")
    contract = {
        "schema": "aorta.r005e2.job_queue.v1",
        "job_count_per_phase": 54,
        "phase_order": ["primary", "replay"],
        "registered_gpus": args.gpus,
        "one_job_per_gpu": True,
        "same_job_gpu_for_replay": True,
        "workers_per_job": args.workers,
        "code_commit": args.code_commit,
        "maximum_idle_memory_mib": args.maximum_idle_memory_mib,
        "development_calibration_test_navdream_target_accessed": False,
        "partial_scientific_metric_reading_prohibited": True,
    }
    contract_path = args.output_root / "run_contract.json"
    assignment_path = args.output_root / "gpu_assignment.json"
    if args.resume:
        existing = json.loads(contract_path.read_text("utf-8"))
        comparable = dict(existing)
        comparable.pop("started_at_utc", None)
        if comparable != contract:
            raise ValueError("resume contract differs from the frozen run")
        assignment = (
            {
                str(key): int(value)
                for key, value in json.loads(assignment_path.read_text("utf-8")).items()
            }
            if assignment_path.is_file()
            else {}
        )
    else:
        assignment = {}
        _write_json(
            contract_path,
            {"started_at_utc": datetime.now(timezone.utc).isoformat(), **contract},
        )
    _run_phase(args=args, phase="primary", assignment=assignment)
    _run_phase(args=args, phase="replay", assignment=assignment)
    _write_json(
        args.output_root / "progress.json",
        {
            "status": "PRIMARY_REPLAY_COMPLETED",
            "phase": "audit_pending",
            "completed_job_count": 108,
            "total_job_count": 108,
            "development_calibration_test_navdream_target_accessed": False,
            "scientific_metrics_emitted": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--mismatch-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum-idle-memory-mib", type=int, default=256)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
