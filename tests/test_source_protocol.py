"""Unit tests for the Gate B source-session protocol."""

from __future__ import annotations

import pytest

from aorta_drive.protocol.source_split import (
    SourceLogRecord,
    build_split_audit,
    choose_anchor_tokens,
    midpoint_anchor_positions,
    recording_group_for_log,
)


def _record(log_name: str, scene_count: int = 17) -> SourceLogRecord:
    return SourceLogRecord(
        log_name=log_name,
        recording_group=recording_group_for_log(log_name),
        scene_count=scene_count,
        token_sha256="a" * 64,
        log_sha256="b" * 64,
        sensor_dir_exists=True,
    )


def test_recording_group_strips_only_the_fragment_interval() -> None:
    name = "2021.06.14.18.33.41_veh-35_04768_04894"
    assert recording_group_for_log(name) == "2021.06.14.18.33.41_veh-35"
    with pytest.raises(ValueError, match="end precedes start"):
        recording_group_for_log("2021.06.14.18.33.41_veh-35_04894_04768")


def test_midpoint_anchors_are_exact_unique_and_cover_the_session() -> None:
    assert midpoint_anchor_positions(17, 16) == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    )
    assert midpoint_anchor_positions(100, 4) == (12, 37, 62, 87)
    with pytest.raises(ValueError, match="cannot exceed"):
        midpoint_anchor_positions(3, 4)


def test_anchor_selection_rejects_duplicate_tokens() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        choose_anchor_tokens(("a", "b", "a"), 2)


def test_split_audit_rejects_fragment_leakage_by_construction() -> None:
    rows = (
        _record("2021.06.14.18.33.41_veh-35_00001_00030"),
        _record("2021.06.14.18.33.41_veh-35_00031_00060"),
        _record("2021.07.16.18.06.21_veh-38_00001_00030"),
        _record("2021.08.17.13.15.12_veh-45_00001_00030"),
        _record("2021.09.14.19.46.05_veh-45_00001_00030"),
    )
    assignments = {
        "2021.06.14.18.33.41_veh-35": "train",
        "2021.07.16.18.06.21_veh-38": "development",
        "2021.08.17.13.15.12_veh-45": "calibration",
        "2021.09.14.19.46.05_veh-45": "test",
    }
    audit = build_split_audit(
        rows,
        assignments,
        {"train": 1, "development": 1, "calibration": 1, "test": 1},
    )
    assert audit.group_counts == {
        "train": 1,
        "development": 1,
        "calibration": 1,
        "test": 1,
    }
    assert audit.log_counts["train"] == 2
    assert audit.total_groups == 4
    assert audit.total_logs == 5


def test_split_audit_rejects_missing_group_and_bad_sensor_contract() -> None:
    row = _record("2021.06.14.18.33.41_veh-35_00001_00030")
    with pytest.raises(ValueError, match="assignment group mismatch"):
        build_split_audit(
            (row,),
            {},
            {"train": 0, "development": 0, "calibration": 0, "test": 0},
        )

    missing_sensor = SourceLogRecord(**{**row.__dict__, "sensor_dir_exists": False})
    with pytest.raises(ValueError, match="missing sensor"):
        build_split_audit(
            (missing_sensor,),
            {row.recording_group: "train"},
            {"train": 1, "development": 0, "calibration": 0, "test": 0},
        )
