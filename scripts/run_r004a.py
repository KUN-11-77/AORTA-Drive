"""Run the formal R004A source-protocol replay audit on a clean revision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from aorta_drive.certification.calibration import minimum_calibration_sessions
from freeze_r004_protocol import freeze_protocol

ROOT = Path(__file__).parents[1]
COMMITTED_MANIFEST_ROOT = ROOT / "configs" / "manifests"
PROTOCOL_CONFIG = ROOT / "configs" / "experiment" / "r004a_source_session_protocol.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _maximum_relative_balance_error(
    rows: list[dict[str, str]], metric: str, capacities: dict[str, int]
) -> float:
    total = sum(int(row[metric]) for row in rows)
    group_total = sum(capacities.values())
    errors: list[float] = []
    for split, capacity in capacities.items():
        actual = sum(int(row[metric]) for row in rows if row["split"] == split)
        expected = total * capacity / group_total
        errors.append(abs(actual - expected) / expected)
    return max(errors)


def run(args: argparse.Namespace) -> dict[str, object]:
    """Replay both manifest construction and repository-only protocol checks."""

    dirty = _git("status", "--porcelain")
    if dirty:
        raise RuntimeError("formal R004A run requires a clean implementation revision")
    revision = _git("rev-parse", "HEAD")

    with tempfile.TemporaryDirectory(prefix="aorta-r004a-first-") as first_name:
        with tempfile.TemporaryDirectory(prefix="aorta-r004a-second-") as second_name:
            first = Path(first_name)
            second = Path(second_name)
            summary_first = freeze_protocol(
                args.source_roster,
                args.token_index,
                args.assignment,
                first,
                anchors_per_session=16,
            )
            summary_second = freeze_protocol(
                args.source_roster,
                args.token_index,
                args.assignment,
                second,
                anchors_per_session=16,
            )
            names = sorted(path.name for path in first.iterdir())
            if names != sorted(path.name for path in second.iterdir()):
                raise AssertionError("replay outputs differ in file names")
            replay_hashes = {name: _sha256(first / name) for name in names}
            if replay_hashes != {name: _sha256(second / name) for name in names}:
                raise AssertionError("two independent manifest replays are not byte-identical")
            committed_hashes = {name: _sha256(COMMITTED_MANIFEST_ROOT / name) for name in names}
            if replay_hashes != committed_hashes:
                raise AssertionError("replayed manifests do not match the committed freeze")
            if summary_first != summary_second:
                raise AssertionError("replayed protocol summaries differ")

    groups = _read_csv(COMMITTED_MANIFEST_ROOT / "r004_recording_group_split.csv")
    logs = _read_csv(COMMITTED_MANIFEST_ROOT / "r004_source_log_manifest.csv")
    anchors = _read_csv(COMMITTED_MANIFEST_ROOT / "r004_session_time_set.csv")
    capacities = {"train": 66, "development": 24, "calibration": 40, "test": 32}
    if Counter(row["split"] for row in groups) != capacities:
        raise AssertionError("frozen group capacities changed")
    if len(groups) != 162 or len(logs) != 1192 or len(anchors) != 2592:
        raise AssertionError("frozen manifest cardinality changed")
    if len({row["scene_token"] for row in anchors}) != len(anchors):
        raise AssertionError("time set contains duplicate scene tokens")
    calibration_sessions = capacities["calibration"]
    alpha_eligibility = {
        "0.10": calibration_sessions >= minimum_calibration_sessions(0.10),
        "0.05": calibration_sessions >= minimum_calibration_sessions(0.05),
    }
    if not all(alpha_eligibility.values()):
        raise AssertionError("calibration split is ineligible for a registered alpha")

    result: dict[str, object] = {
        "experiment_id": "R004A-source-session-protocol",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "clean_implementation_revision": revision,
        "protocol_config_sha256": _sha256(PROTOCOL_CONFIG),
        "source_inputs": {
            "source_roster_sha256": _sha256(args.source_roster),
            "source_token_index_sha256": _sha256(args.token_index),
            "split_assignment_sha256": _sha256(args.assignment),
        },
        "split": summary_first["split_audit"],
        "anchors_per_recording_group": 16,
        "total_anchors": len(anchors),
        "calibration_independent_sessions": calibration_sessions,
        "calibration_anchor_count": sum(row["split"] == "calibration" for row in anchors),
        "alpha_eligibility": alpha_eligibility,
        "maximum_relative_scene_imbalance": _maximum_relative_balance_error(
            groups, "scene_count", capacities
        ),
        "maximum_relative_log_imbalance": _maximum_relative_balance_error(
            groups, "log_count", capacities
        ),
        "byte_identical_replays": 2,
        "committed_manifest_hashes": committed_hashes,
        "claim_boundary": (
            "Closes source split independence and finite T(Z) only; no predictor, "
            "coverage, performance, OOD, or Gate B success claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-roster", type=Path, required=True)
    parser.add_argument("--token-index", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
