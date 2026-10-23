"""Repository-level audit of the frozen R004A source protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from aorta_drive.certification.calibration import minimum_calibration_sessions
from aorta_drive.protocol.source_split import midpoint_anchor_positions

ROOT = Path(__file__).parents[1]
MANIFEST_ROOT = ROOT / "configs" / "manifests"


def _rows(name: str) -> list[dict[str, str]]:
    with (MANIFEST_ROOT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_manifests_match_the_committed_fingerprint() -> None:
    fingerprint = json.loads(
        (MANIFEST_ROOT / "r004_protocol_fingerprint.json").read_text(encoding="utf-8")
    )
    for name, digest in fingerprint["manifests"].items():
        assert _sha256(MANIFEST_ROOT / name) == digest
    assert fingerprint["logical_sha256"] == (
        "090acf0c3621362b3d7c50e88a92a59d823a80d900e6e285fe25012d9dacab2b"
    )


def test_recording_groups_are_disjoint_and_all_logs_inherit_one_split() -> None:
    groups = _rows("r004_recording_group_split.csv")
    logs = _rows("r004_source_log_manifest.csv")
    group_split = {row["recording_group"]: row["split"] for row in groups}
    assert len(groups) == len(group_split) == 162
    assert Counter(group_split.values()) == {
        "train": 66,
        "development": 24,
        "calibration": 40,
        "test": 32,
    }
    assert len(logs) == len({row["log_name"] for row in logs}) == 1192
    assert sum(int(row["scene_count"]) for row in logs) == 103288
    for row in logs:
        assert row["split"] == group_split[row["recording_group"]]
        assert row["sensor_dir_exists"] == "true"


def test_time_set_uses_exact_midpoint_positions_for_every_session() -> None:
    groups = _rows("r004_recording_group_split.csv")
    anchors = _rows("r004_session_time_set.csv")
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in anchors:
        by_group[row["recording_group"]].append(row)
    assert len(anchors) == len({row["scene_token"] for row in anchors}) == 2592
    for group in groups:
        rows = sorted(by_group[group["recording_group"]], key=lambda row: int(row["anchor_index"]))
        assert len(rows) == 16
        positions = tuple(int(row["session_position"]) for row in rows)
        assert positions == midpoint_anchor_positions(int(group["scene_count"]), 16)
        assert {row["split"] for row in rows} == {group["split"]}


def test_calibration_count_supports_both_registered_alphas() -> None:
    calibration_sessions = 40
    assert calibration_sessions >= minimum_calibration_sessions(0.10)
    assert calibration_sessions >= minimum_calibration_sessions(0.05)
