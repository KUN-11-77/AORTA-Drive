"""R002C: analytic continuous-sweep outer-cover qualification."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aorta_drive.certification.geometry import (
    ContinuousCorrection,
    Pose2D,
    RectangularGrid,
    VehicleRectangle,
    certify_native_interior_nodes,
    certify_swept_outer_cover,
)


def _world_body_point(body_point: np.ndarray, pose: Pose2D) -> np.ndarray:
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return np.array([pose.x, pose.y]) + rotation @ body_point


def _interpolate(start: Pose2D, end: Pose2D, amount: float) -> Pose2D:
    return Pose2D(
        x=(1.0 - amount) * start.x + amount * end.x,
        y=(1.0 - amount) * start.y + amount * end.y,
        yaw=(1.0 - amount) * start.yaw + amount * end.yaw,
    )


def _cell_index(point: np.ndarray, grid: RectangularGrid) -> int:
    column = math.floor((float(point[0]) - grid.x_min) / grid.resolution)
    row = math.floor((float(point[1]) - grid.y_min) / grid.resolution)
    assert 0 <= column < grid.columns
    assert 0 <= row < grid.rows
    return row * grid.columns + column


def _rectangle_infimum(
    x_min: float, x_max: float, y_min: float, y_max: float, gradient: np.ndarray
) -> float:
    x = x_min if gradient[0] >= 0 else x_max
    y = y_min if gradient[1] >= 0 else y_max
    return float(gradient @ np.array([x, y]))


def test_outer_cover_contains_straight_rotating_and_mixed_sweeps() -> None:
    body = VehicleRectangle(rear_overhang=1.0, front_extent=2.0, half_width=0.75)
    grid = RectangularGrid(x_min=-10.0, y_min=-10.0, columns=40, rows=40, resolution=0.5)
    paths = (
        (Pose2D(-2.0, 0.0, 0.0), Pose2D(2.0, 0.0, 0.0)),
        (Pose2D(0.0, 0.0, -0.6), Pose2D(0.0, 0.0, 0.8)),
        (
            Pose2D(-2.0, -1.0, -0.4),
            Pose2D(0.0, 1.0, 0.3),
            Pose2D(2.0, 0.0, 1.0),
        ),
    )
    body_points = np.array(
        [
            [-body.rear_overhang, -body.half_width],
            [-body.rear_overhang, body.half_width],
            [body.front_extent, -body.half_width],
            [body.front_extent, body.half_width],
            [0.0, 0.0],
            [body.front_extent, 0.0],
        ]
    )

    for poses in paths:
        certificate = certify_swept_outer_cover(body, poses, grid)
        selected = set(certificate.node_indices)
        assert certificate.h == grid.cell_radius
        for start, end in zip(poses, poses[1:], strict=False):
            for amount in np.linspace(0.0, 1.0, 201):
                pose = _interpolate(start, end, float(amount))
                for body_point in body_points:
                    world_point = _world_body_point(body_point, pose)
                    assert _cell_index(world_point, grid) in selected


def test_segment_inflation_matches_the_analytic_motion_bound() -> None:
    body = VehicleRectangle(rear_overhang=1.0, front_extent=3.0, half_width=1.0)
    grid = RectangularGrid(x_min=-20.0, y_min=-20.0, columns=80, rows=80, resolution=0.5)
    start = Pose2D(-1.0, -2.0, -0.4)
    end = Pose2D(3.0, 1.0, 0.8)
    pose_error = 0.03
    certificate = certify_swept_outer_cover(body, (start, end), grid, pose_error=pose_error)
    cover = certificate.segment_covers[0]
    expected_translation = 0.5 * 5.0
    expected_rotation = 2.0 * body.radius * math.sin(abs(end.yaw - start.yaw) / 4.0)
    assert np.isclose(cover.translation_term, expected_translation)
    assert np.isclose(cover.rotation_term, expected_rotation)
    assert np.isclose(cover.inflation, expected_translation + expected_rotation + pose_error)


def test_outer_cover_lifting_is_sound_for_linear_one_lipschitz_teachers() -> None:
    rng = np.random.default_rng(20261026)
    body = VehicleRectangle(rear_overhang=1.0, front_extent=2.0, half_width=0.5)
    grid = RectangularGrid(x_min=-8.0, y_min=-4.0, columns=32, rows=16, resolution=0.5)
    challenger_poses = (Pose2D(0.0, 0.0, 0.0), Pose2D(2.0, 0.0, 0.0))
    native_poses = (Pose2D(-2.0, 0.0, 0.0),)
    challenger = certify_swept_outer_cover(body, challenger_poses, grid)
    native = certify_native_interior_nodes(body, native_poses, grid)
    centers = grid.centers()
    correction = ContinuousCorrection(challenger.h, 0.0, 0.0, 0.0)

    for _ in range(500):
        gradient = rng.normal(size=2)
        gradient /= max(1.0, np.linalg.norm(gradient))
        labels = centers @ gradient
        discrete = float(
            np.min(labels[list(challenger.node_indices)])
            - np.min(labels[list(native.node_indices)])
        )
        certified = correction.apply(discrete)
        true_challenger = _rectangle_infimum(-1.0, 4.0, -0.5, 0.5, gradient)
        true_native = _rectangle_infimum(-3.0, 0.0, -0.5, 0.5, gradient)
        assert certified <= true_challenger - true_native + 1e-12


def test_checker_rejects_roi_clipping_wrapped_yaw_and_empty_native_set() -> None:
    body = VehicleRectangle(rear_overhang=1.0, front_extent=2.0, half_width=0.75)
    small_grid = RectangularGrid(x_min=-1.0, y_min=-1.0, columns=4, rows=4, resolution=0.5)
    with pytest.raises(ValueError, match="leaves the certified grid ROI"):
        certify_swept_outer_cover(body, (Pose2D(0.0, 0.0, 0.0),), small_grid)

    large_grid = RectangularGrid(x_min=-10.0, y_min=-10.0, columns=40, rows=40, resolution=0.5)
    with pytest.raises(ValueError, match="unwrapped"):
        certify_swept_outer_cover(
            body,
            (Pose2D(0.0, 0.0, 3.0), Pose2D(0.0, 0.0, -3.0)),
            large_grid,
        )

    tiny_body = VehicleRectangle(rear_overhang=0.01, front_extent=0.01, half_width=0.01)
    with pytest.raises(ValueError, match="no fixed-grid center"):
        certify_native_interior_nodes(tiny_body, (Pose2D(0.0, 0.0, 0.0),), large_grid)


def test_every_native_node_is_inside_a_real_discrete_footprint() -> None:
    body = VehicleRectangle(rear_overhang=1.0, front_extent=2.0, half_width=0.75)
    grid = RectangularGrid(x_min=-10.0, y_min=-10.0, columns=40, rows=40, resolution=0.5)
    poses = (Pose2D(-1.0, 0.0, -0.3), Pose2D(1.0, 1.0, 0.6))
    certificate = certify_native_interior_nodes(body, poses, grid)
    centers = grid.centers()

    for index in certificate.node_indices:
        point = centers[index]
        inside_any = False
        for pose in poses:
            relative = point - np.array([pose.x, pose.y])
            cosine = math.cos(pose.yaw)
            sine = math.sin(pose.yaw)
            local_x = cosine * relative[0] + sine * relative[1]
            local_y = -sine * relative[0] + cosine * relative[1]
            inside_any |= (
                -body.rear_overhang <= local_x <= body.front_extent
                and abs(local_y) <= body.half_width
            )
        assert inside_any
