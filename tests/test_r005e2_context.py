from __future__ import annotations

import numpy as np
import pytest

from aorta_drive.certification.geometry import Pose2D
from aorta_drive.training.r005e2_context import (
    align_previous_simulated_context,
    array_sha256,
)
from build_r005e2_scientific_bundle import _records_for_keys


def test_previous_context_removes_expired_prefix_and_uses_current_frame() -> None:
    states = np.zeros((8, 3), dtype=np.float64)
    states[:, 0] = np.arange(8, dtype=np.float64)
    states[:, 2] = 0.25
    context = align_previous_simulated_context(
        states,
        previous_time_s=10.0,
        current_time_s=10.2,
        current_pose=Pose2D(2.0, 0.0, 0.25),
        simulation_interval_s=0.1,
        x_index=0,
        y_index=1,
        heading_index=2,
    )
    assert context.overlap_start == 2
    assert context.poses.shape == (6, 3)
    assert context.poses[0] == pytest.approx(np.zeros(3), abs=1.0e-12)
    assert context.poses[-1, 0] == pytest.approx(5.0 * np.cos(0.25))
    assert context.poses[-1, 1] == pytest.approx(-5.0 * np.sin(0.25))


def test_previous_context_rejects_noncausal_or_empty_overlap() -> None:
    states = np.zeros((4, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="observation gap"):
        align_previous_simulated_context(
            states,
            previous_time_s=1.0,
            current_time_s=1.0,
            current_pose=Pose2D(0.0, 0.0, 0.0),
            simulation_interval_s=0.1,
            x_index=0,
            y_index=1,
            heading_index=2,
        )
    with pytest.raises(ValueError, match="no valid temporal overlap"):
        align_previous_simulated_context(
            states,
            previous_time_s=1.0,
            current_time_s=1.4,
            current_pose=Pose2D(0.0, 0.0, 0.0),
            simulation_interval_s=0.1,
            x_index=0,
            y_index=1,
            heading_index=2,
        )


def test_r005e1b_array_hash_binds_shape_and_values() -> None:
    value = np.arange(12, dtype=np.float64).reshape(4, 3)
    assert array_sha256(value) == array_sha256(value.copy())
    assert array_sha256(value) != array_sha256(value.reshape(3, 4))
    changed = value.copy()
    changed[0, 0] = 1.0
    assert array_sha256(value) != array_sha256(changed)


def test_metric_manifest_may_be_a_superset_but_join_is_exact() -> None:
    payload = {
        "records": [
            {"recording_group": "g0", "anchor_index": 0},
            {"recording_group": "g1", "anchor_index": 0},
            {"recording_group": "unused", "anchor_index": 0},
        ]
    }
    assert set(_records_for_keys(payload, {"g0::00", "g1::00"})) == {
        "g0::00",
        "g1::00",
    }


def test_metric_manifest_rejects_duplicate_or_missing_required_identity() -> None:
    duplicate = {
        "records": [
            {"recording_group": "g0", "anchor_index": 0},
            {"recording_group": "g0", "anchor_index": 0},
        ]
    }
    with pytest.raises(ValueError, match="duplicate"):
        _records_for_keys(duplicate, {"g0::00"})
    with pytest.raises(ValueError, match="uniquely cover"):
        _records_for_keys({"records": []}, {"g0::00"})
