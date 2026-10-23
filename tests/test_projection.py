"""Tests for the physical-node to DINO patch-token projection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aorta_drive.vision.projection import (
    distort_normalized_points_opencv,
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


def test_brown_conrady_matches_analytic_forward_equations() -> None:
    normalized = np.array([[0.2, -0.1], [-0.3, 0.4]])
    distortion = np.array([-0.2, 0.03, 0.004, -0.005, 0.01])
    result = distort_normalized_points_opencv(normalized, distortion)

    expected = []
    k1, k2, p1, p2, k3 = distortion
    for x, y in normalized:
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        expected.append(
            [
                x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
                y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
            ]
        )
    assert np.allclose(result, expected, rtol=0.0, atol=1e-15)


def test_nonzero_distortion_changes_raw_projection() -> None:
    common = (
        np.array([[0.4, -0.2, 1.0]]),
        np.eye(3),
        np.zeros(3),
        np.array([[100.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]]),
    )
    pinhole = project_points_to_patch_grid(
        *common,
        np.zeros(5),
        raw_shape=(100, 120),
        resized_shape=(80, 96),
        patch_size=16,
    )
    distorted = project_points_to_patch_grid(
        *common,
        np.array([-0.2, 0.03, 0.004, -0.005, 0.01]),
        raw_shape=(100, 120),
        resized_shape=(80, 96),
        patch_size=16,
    )
    assert np.allclose(pinhole.raw_uv, [[90.0, 16.0]], rtol=0.0, atol=1e-14)
    assert not np.allclose(distorted.raw_uv, pinhole.raw_uv)


def test_projection_rejects_polynomial_foldback_outside_calibrated_ray_domain() -> None:
    intrinsic = np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
    distortion = np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231])
    result = project_points_to_patch_grid(
        np.array([[1.8, 0.0, 1.0]]),
        np.eye(3),
        np.zeros(3),
        intrinsic,
        distortion,
        raw_shape=(1080, 1920),
        resized_shape=(288, 512),
        patch_size=16,
    )
    assert 0.0 <= result.raw_uv[0, 0] <= 1919.0
    assert not result.inside_calibrated_ray_domain[0]
    assert not result.inside_raw_image[0]
    assert not result.valid_for_bilinear_patch_sampling[0]


def test_projection_rejects_unsupported_preprocessing_contracts() -> None:
    common = (
        np.array([[0.0, 0.0, 1.0]]),
        np.eye(3),
        np.zeros(3),
        np.eye(3),
    )
    with pytest.raises(ValueError, match="exactly"):
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
