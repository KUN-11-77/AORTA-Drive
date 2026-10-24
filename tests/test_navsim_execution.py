"""Dependency-free tests for the NAVSIM execution bridge arithmetic."""

from __future__ import annotations

import numpy as np

from aorta_drive.certification.geometry import Pose2D
from aorta_drive.evaluation.navsim_execution import _states_to_local_poses


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
