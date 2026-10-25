"""Sequential bundle, 54-job primary, replay, and audit chain for R005E2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _run(command: list[str], log: Path) -> None:
    with log.open("wb") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(
            f"subprocess failed with exit {completed.returncode}: {command[1]}"
        )


def _bundle_command(args: argparse.Namespace, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(args.bundle_builder),
        "--config",
        str(args.config),
        "--panel-roster",
        str(args.panel_roster),
        "--target-result",
        str(args.target_result),
        "--previous-root",
        str(args.previous_root),
        "--previous-manifest",
        str(args.previous_manifest),
        "--visual-root",
        str(args.visual_root),
        "--visual-manifest",
        str(args.visual_manifest),
        "--crossfit-manifest",
        str(args.crossfit_manifest),
        "--output-directory",
        str(output),
    ]
    for root in args.panel_root:
        command.extend(("--panel-root", str(root)))
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(root: Path) -> None:
    destination = root / "SHA256SUMS.txt"
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != destination and not path.name.endswith(".log")
    )
    destination.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
        ),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    _write_json(
        args.output_root / "progress.json",
        {
            "status": "BUNDLE_PRIMARY",
            "completed_job_count": 0,
            "total_job_count": 108,
            "development_calibration_test_navdream_target_accessed": False,
            "scientific_metrics_emitted": False,
        },
    )
    primary_bundle = args.output_root / "bundle_primary"
    replay_bundle = args.output_root / "bundle_replay"
    try:
        _run(
            _bundle_command(args, primary_bundle),
            args.output_root / "bundle_primary.log",
        )
        _write_json(
            args.output_root / "progress.json",
            {
                "status": "BUNDLE_REPLAY",
                "completed_job_count": 0,
                "total_job_count": 108,
                "development_calibration_test_navdream_target_accessed": False,
                "scientific_metrics_emitted": False,
            },
        )
        _run(
            _bundle_command(args, replay_bundle),
            args.output_root / "bundle_replay.log",
        )
        _run(
            [
                sys.executable,
                str(args.bundle_auditor),
                "--primary",
                str(primary_bundle),
                "--replay",
                str(replay_bundle),
                "--output",
                str(args.output_root / "bundle_audit.json"),
            ],
            args.output_root / "bundle_audit.log",
        )
        science_root = args.output_root / "scientific_run"
        _write_json(
            args.output_root / "progress.json",
            {
                "status": "SCIENTIFIC_PRIMARY_REPLAY",
                "completed_job_count": 0,
                "total_job_count": 108,
                "nested_progress": str(science_root / "progress.json"),
                "development_calibration_test_navdream_target_accessed": False,
                "scientific_metrics_emitted": False,
            },
        )
        _run(
            [
                sys.executable,
                str(args.queue_runner),
                "--runner",
                str(args.fold_runner),
                "--bundle-root",
                str(primary_bundle),
                "--visual-root",
                str(args.visual_root),
                "--crossfit-manifest",
                str(args.crossfit_manifest),
                "--mismatch-manifest",
                str(args.mismatch_manifest),
                "--output-root",
                str(science_root),
                "--code-commit",
                args.code_commit,
                "--gpus",
                *(str(gpu) for gpu in args.gpus),
                "--workers",
                str(args.workers),
            ],
            args.output_root / "scientific_queue.log",
        )
        _write_json(
            args.output_root / "progress.json",
            {
                "status": "MECHANICAL_AUDIT",
                "completed_job_count": 108,
                "total_job_count": 108,
                "development_calibration_test_navdream_target_accessed": False,
                "scientific_metrics_emitted": False,
            },
        )
        _run(
            [
                sys.executable,
                str(args.scientific_auditor),
                "--run-root",
                str(science_root),
                "--bundle-root",
                str(primary_bundle),
                "--crossfit-manifest",
                str(args.crossfit_manifest),
                "--mismatch-manifest",
                str(args.mismatch_manifest),
                "--config",
                str(args.config),
                "--output",
                str(args.output_root / "audit.json"),
            ],
            args.output_root / "audit.log",
        )
        _write_checksums(args.output_root)
        (args.output_root / "exit_code.txt").write_text("0\n", encoding="ascii")
        audit = json.loads((args.output_root / "audit.json").read_text("utf-8"))
        _write_json(
            args.output_root / "progress.json",
            {
                "status": "COMPLETED",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_job_count": 108,
                "total_job_count": 108,
                "development_calibration_test_navdream_target_accessed": False,
                "scientific_metrics_emitted": True,
                "support": audit["support"],
                "decision": audit["decision"],
            },
        )
    except Exception as error:
        (args.output_root / "exit_code.txt").write_text("1\n", encoding="ascii")
        _write_json(
            args.output_root / "failure.json",
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "development_calibration_test_navdream_target_accessed": False,
                "scientific_metrics_emitted": False,
                "automatic_third_run_started": False,
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "bundle-builder",
        "bundle-auditor",
        "queue-runner",
        "fold-runner",
        "scientific-auditor",
        "config",
        "panel-roster",
        "target-result",
        "previous-root",
        "previous-manifest",
        "visual-root",
        "visual-manifest",
        "crossfit-manifest",
        "mismatch-manifest",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, action="append", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
