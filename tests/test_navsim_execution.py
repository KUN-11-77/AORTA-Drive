"""Dependency-free tests for the NAVSIM execution bridge arithmetic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import Pose2D
from aorta_drive.evaluation.navsim_execution import _states_to_local_poses


ROOT = Path(__file__).resolve().parents[1]


def test_registered_official_execution_config_parses() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/experiment/r004h4p4e_official_execution_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["secondary_diagnostics"]["decision_use"] == "prohibited"
    assert len(config["secondary_diagnostics"]["metrics"]) == 5
    assert config["interface_gates"]["expected_batch_size"] == 7


def test_state_batch_inverse_transform_is_shape_preserving_and_exact_at_origin() -> None:
    states = np.zeros((2, 3, 5), dtype=np.float64)
    states[:, :, 0] = np.array([[10.0, 11.0, 12.0], [10.0, 9.0, 8.0]])
    states[:, :, 1] = np.array([[20.0, 20.0, 20.0], [20.0, 20.0, 20.0]])
    states[:, :, 2] = 0.25
    poses, error = _states_to_local_poses(
        states,
        current_pose=Pose2D(10.0, 20.0, 0.0),
        x_index=0,
        y_index=1,
        heading_index=2,
    )
    assert poses.shape == (2, 3, 3)
    assert np.array_equal(poses[:, 0, :2], np.zeros((2, 2)))
    assert np.array_equal(poses[0, :, 0], np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(poses[1, :, 0], np.array([0.0, -1.0, -2.0]))
    assert error >= 0.0


def test_state_batch_inverse_transform_rejects_nonfinite_values() -> None:
    states = np.zeros((1, 2, 3), dtype=np.float64)
    states[0, 1, 0] = np.nan
    try:
        _states_to_local_poses(
            states,
            current_pose=Pose2D(0.0, 0.0, 0.0),
            x_index=0,
            y_index=1,
            heading_index=2,
        )
    except ValueError:
        return
    raise AssertionError("non-finite official states must fail closed")
