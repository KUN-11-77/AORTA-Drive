"""Tests for the physical-node to DINO patch-token projection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aorta_drive.vision.projection import (
    project_points_to_patch_grid,
    transform_current_to_history,
    yaw_from_wxyz,
)


def test_temporal_transform_matches_absolute_pose_composition() -> None:
    points = np.array([[1.0, 0.0], [0.0, 1.0]])
    transformed = transform_current_to_history(
        points,
        current_translation_xy=[10.0, 2.0],
        current_yaw=math.pi / 2,
        history_translation_xy=[9.0, 2.0],
        history_yaw=0.0,
    )
    assert np.allclose(transformed, [[1.0, 1.0], [0.0, 0.0]], atol=1e-14)


def test_quaternion_yaw_is_scale_invariant() -> None:
    angle = 0.7
    quaternion = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
    assert yaw_from_wxyz(quaternion * 9.0) == pytest.approx(angle)


def test_projection_uses_half_pixel_resize_and_patch_center_coordinates() -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [-50.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    intrinsic = np.array([[100.0, 0.0, 799.5], [0.0, 100.0, 449.5], [0.0, 0.0, 1.0]])
    result = project_points_to_patch_grid(
        points,
        np.eye(3),
        np.zeros(3),
        intrinsic,
        np.zeros(5),
        raw_shape=(900, 1600),
        resized_shape=(288, 512),
        patch_size=16,
    )
    assert np.allclose(result.resized_uv[0], [255.5, 143.5])
    assert np.allclose(result.patch_xy[0], [15.5, 8.5])
    assert result.valid_for_bilinear_patch_sampling.tolist() == [True, False, False]


def test_projection_rejects_unsupported_preprocessing_contracts() -> None:
    common = (
        np.array([[0.0, 0.0, 1.0]]),
        np.eye(3),
        np.zeros(3),
        np.eye(3),
    )
    with pytest.raises(ValueError, match="distortion"):
        project_points_to_patch_grid(
            *common,
            np.array([1e-3, 0.0]),
            raw_shape=(900, 1600),
            resized_shape=(288, 512),
            patch_size=16,
        )
    with pytest.raises(ValueError, match="divisible"):
        project_points_to_patch_grid(
            *common,
            np.zeros(5),
            raw_shape=(900, 1600),
            resized_shape=(287, 512),
            patch_size=16,
        )
