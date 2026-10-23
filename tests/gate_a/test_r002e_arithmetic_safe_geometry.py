"""R002E: arithmetic-safe rotation-free production geometry fallback."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from aorta_drive.certification.geometry import (
    Pose2D,
    RectangularGrid,
    VehicleRectangle,
    certify_native_inner_disk_nodes,
    certify_swept_disk_cover,
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


def test_cell_radius_is_outward_safe_for_stored_binary64_centers() -> None:
    grid = RectangularGrid(
        x_min=-7.3,
        y_min=-4.1,
        columns=31,
        rows=23,
        resolution=0.3,
    )
    half = Fraction.from_float(grid.resolution) / 2
    ideal_lower_bound = half * half + half * half
    assert Fraction.from_float(grid.cell_radius) ** 2 >= ideal_lower_bound


def test_disk_cover_contains_motion_with_arbitrary_yaw_interpolation() -> None:
    body = VehicleRectangle(rear_overhang=1.127, front_extent=4.049, half_width=1.1485)
    grid = RectangularGrid(x_min=-20.0, y_min=-20.0, columns=80, rows=80, resolution=0.5)
    poses = (
        Pose2D(-4.0, -1.0, 3.0),
        Pose2D(0.0, 2.0, -2.8),
        Pose2D(5.0, 0.0, 7.0),
    )
    certificate = certify_swept_disk_cover(body, poses, grid)
    selected = set(certificate.node_indices)
    body_points = np.array(
        [
            [-body.rear_overhang, -body.half_width],
            [-body.rear_overhang, body.half_width],
            [body.front_extent, -body.half_width],
            [body.front_extent, body.half_width],
            [0.0, 0.0],
        ]
    )
    for start, end in zip(poses, poses[1:], strict=False):
        for amount in np.linspace(0.0, 1.0, 201):
            pose = _interpolate(start, end, float(amount))
            for body_point in body_points:
                world = _world_body_point(body_point, pose)
                assert _cell_index(world, grid) in selected


def test_native_inner_disk_nodes_stay_inside_under_pose_error_and_any_yaw() -> None:
    body = VehicleRectangle(rear_overhang=1.127, front_extent=4.049, half_width=1.1485)
    grid = RectangularGrid(x_min=-10.0, y_min=-10.0, columns=40, rows=40, resolution=0.5)
    nominal = Pose2D(0.0, 0.0, 0.0)
    pose_error = 0.1
    certificate = certify_native_inner_disk_nodes(body, (nominal,), grid, pose_error=pose_error)
    centers = grid.centers()
    rng = np.random.default_rng(20261029)

    for index in certificate.node_indices:
        point = centers[index]
        for _ in range(50):
            angle = float(rng.uniform(-math.pi, math.pi))
            error_angle = float(rng.uniform(-math.pi, math.pi))
            error_radius = float(rng.uniform(0.0, pose_error))
            actual_center = np.array(
                [
                    error_radius * math.cos(error_angle),
                    error_radius * math.sin(error_angle),
                ]
            )
            relative = point - actual_center
            cosine = math.cos(angle)
            sine = math.sin(angle)
            local_x = cosine * relative[0] + sine * relative[1]
            local_y = -sine * relative[0] + cosine * relative[1]
            assert -body.rear_overhang <= local_x <= body.front_extent
            assert abs(local_y) <= body.half_width


def test_disk_route_rejects_roi_clipping_empty_sets_and_excess_pose_error() -> None:
    body = VehicleRectangle(rear_overhang=1.127, front_extent=4.049, half_width=1.1485)
    small_grid = RectangularGrid(x_min=-2.0, y_min=-2.0, columns=8, rows=8, resolution=0.5)
    with pytest.raises(ValueError, match="leaves the certified grid ROI"):
        certify_swept_disk_cover(body, (Pose2D(0.0, 0.0, 0.0),), small_grid)

    large_grid = RectangularGrid(x_min=-10.0, y_min=-10.0, columns=40, rows=40, resolution=0.5)
    with pytest.raises(ValueError, match="no non-empty"):
        certify_native_inner_disk_nodes(
            body,
            (Pose2D(0.0, 0.0, 0.0),),
            large_grid,
            pose_error=body.inner_radius,
        )

    offset_grid = RectangularGrid(x_min=10.0, y_min=10.0, columns=20, rows=20, resolution=0.5)
    with pytest.raises(ValueError, match="no fixed-grid center"):
        certify_native_inner_disk_nodes(
            body,
            (Pose2D(0.0, 0.0, 0.0),),
            offset_grid,
        )
