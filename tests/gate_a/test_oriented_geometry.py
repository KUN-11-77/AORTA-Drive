"""Synthetic qualification tests for orientation-aware sweep certificates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aorta_drive.certification.geometry import Pose2D, RectangularGrid, VehicleRectangle
from aorta_drive.certification.oriented_geometry import (
    certify_oriented_bidirectional_noninclusion,
    certify_oriented_inner_cells,
    certify_oriented_swept_outer_cover,
)


BODY = VehicleRectangle(rear_overhang=1.127, front_extent=4.049, half_width=1.1485)
GRID = RectangularGrid(x_min=-16.0, y_min=-16.0, columns=32, rows=32, resolution=1.0)
FINE_GEOMETRY_GRID = RectangularGrid(
    x_min=-4.0, y_min=-3.0, columns=40, rows=24, resolution=0.25
)


def test_identical_oriented_sweeps_have_zero_witness_area() -> None:
    poses = (Pose2D(-4.0, 0.0, 0.0), Pose2D(4.0, 0.0, 0.0))
    result = certify_oriented_bidirectional_noninclusion(BODY, poses, poses, GRID)
    assert result.challenger_minus_native_area_lower_m2 == 0.0
    assert result.native_minus_challenger_area_lower_m2 == 0.0


def test_one_meter_cells_can_be_vacuous_for_one_point_five_meter_shift() -> None:
    native = (Pose2D(0.0, -0.75, 0.0),)
    challenger = (Pose2D(0.0, 0.75, 0.0),)
    result = certify_oriented_bidirectional_noninclusion(
        BODY, challenger, native, GRID
    )
    assert result.challenger_minus_native_area_lower_m2 == 0.0
    assert result.native_minus_challenger_area_lower_m2 == 0.0


def test_fine_geometry_cells_resolve_one_point_five_meter_shift() -> None:
    native = (Pose2D(0.0, -0.75, 0.0),)
    challenger = (Pose2D(0.0, 0.75, 0.0),)
    result = certify_oriented_bidirectional_noninclusion(
        BODY, challenger, native, FINE_GEOMETRY_GRID
    )
    assert result.challenger_minus_native_area_lower_m2 >= 1.0
    assert result.native_minus_challenger_area_lower_m2 >= 1.0


def test_pose_error_cannot_create_oriented_witness_area() -> None:
    native = (Pose2D(-4.0, -1.0, 0.0), Pose2D(4.0, -1.0, 0.0))
    challenger = (Pose2D(-4.0, 1.0, 0.0), Pose2D(4.0, 1.0, 0.0))
    exact = certify_oriented_bidirectional_noninclusion(BODY, challenger, native, GRID)
    uncertain = certify_oriented_bidirectional_noninclusion(
        BODY, challenger, native, GRID, pose_error=0.2
    )
    assert (
        uncertain.challenger_minus_native_area_lower_m2
        <= exact.challenger_minus_native_area_lower_m2
    )
    assert (
        uncertain.native_minus_challenger_area_lower_m2
        <= exact.native_minus_challenger_area_lower_m2
    )


def test_single_last_pose_can_supply_inner_cells() -> None:
    cells = certify_oriented_inner_cells(BODY, (Pose2D(0.0, 0.0, 0.0),), GRID)
    assert cells.cell_indices


def test_roi_exit_is_rejected_instead_of_clipped() -> None:
    poses = (Pose2D(14.0, 0.0, 0.0), Pose2D(15.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="leaves the certified grid ROI"):
        certify_oriented_swept_outer_cover(BODY, poses, GRID)


def test_shifted_witness_grid_cannot_expand_the_certified_physical_roi() -> None:
    shifted_partition = RectangularGrid(
        x_min=-16.5,
        y_min=-16.5,
        columns=33,
        rows=33,
        resolution=1.0,
    )
    poses = (Pose2D(14.0, 0.0, 0.0), Pose2D(15.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="leaves the certified grid ROI"):
        certify_oriented_swept_outer_cover(
            BODY,
            poses,
            shifted_partition,
            certification_support=GRID,
        )


def test_shifted_witness_partition_keeps_same_support_contract() -> None:
    shifted_partition = RectangularGrid(
        x_min=-4.125,
        y_min=-3.125,
        columns=33,
        rows=25,
        resolution=0.25,
    )
    native = (Pose2D(0.0, -0.75, 0.0),)
    challenger = (Pose2D(0.0, 0.75, 0.0),)
    result = certify_oriented_bidirectional_noninclusion(
        BODY,
        challenger,
        native,
        shifted_partition,
        certification_support=FINE_GEOMETRY_GRID,
    )
    assert result.challenger_minus_native_area_lower_m2 > 0.0
    assert result.native_minus_challenger_area_lower_m2 > 0.0


def test_unwrapped_yaw_jump_above_pi_is_rejected() -> None:
    poses = (Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, math.pi + 0.01))
    with pytest.raises(ValueError, match="unwrapped"):
        certify_oriented_swept_outer_cover(BODY, poses, GRID)


def test_rotation_increases_segment_cover_without_losing_finiteness() -> None:
    straight = certify_oriented_swept_outer_cover(
        BODY, (Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, 0.0)), GRID
    )
    turning = certify_oriented_swept_outer_cover(
        BODY, (Pose2D(0.0, 0.0, -0.2), Pose2D(0.0, 0.0, 0.2)), GRID
    )
    assert turning.segment_covers[0].rotation_term_m > 0.0
    assert (
        turning.segment_covers[0].inflation_m
        > straight.segment_covers[0].inflation_m
    )


def test_dense_interpolated_corners_are_covered_by_selected_nodes() -> None:
    start = Pose2D(-1.0, -0.5, -0.3)
    end = Pose2D(1.0, 0.5, 0.4)
    certificate = certify_oriented_swept_outer_cover(BODY, (start, end), GRID)
    centers = GRID.centers()[list(certificate.node_indices)]
    local_corners = (
        (-BODY.rear_overhang, -BODY.half_width),
        (-BODY.rear_overhang, BODY.half_width),
        (BODY.front_extent, -BODY.half_width),
        (BODY.front_extent, BODY.half_width),
    )
    for fraction in np.linspace(0.0, 1.0, 101):
        x = start.x + fraction * (end.x - start.x)
        y = start.y + fraction * (end.y - start.y)
        yaw = start.yaw + fraction * (end.yaw - start.yaw)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        for local_x, local_y in local_corners:
            point = np.array(
                [
                    x + cosine * local_x - sine * local_y,
                    y + sine * local_x + cosine * local_y,
                ]
            )
            nearest = float(np.min(np.linalg.norm(centers - point, axis=1)))
            assert nearest <= certificate.h + 1e-12


def test_certified_inner_cell_corners_are_inside_rotated_body() -> None:
    pose = Pose2D(0.0, 0.0, 0.37)
    certificate = certify_oriented_inner_cells(BODY, (pose,), FINE_GEOMETRY_GRID)
    assert certificate.cell_indices
    for index in certificate.cell_indices:
        row, column = divmod(index, FINE_GEOMETRY_GRID.columns)
        x0 = FINE_GEOMETRY_GRID.x_min + column * FINE_GEOMETRY_GRID.resolution
        y0 = FINE_GEOMETRY_GRID.y_min + row * FINE_GEOMETRY_GRID.resolution
        for x in (x0, x0 + FINE_GEOMETRY_GRID.resolution):
            for y in (y0, y0 + FINE_GEOMETRY_GRID.resolution):
                dx, dy = x - pose.x, y - pose.y
                local_x = math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy
                local_y = -math.sin(pose.yaw) * dx + math.cos(pose.yaw) * dy
                assert -BODY.rear_overhang - 1e-12 <= local_x <= BODY.front_extent + 1e-12
                assert -BODY.half_width - 1e-12 <= local_y <= BODY.half_width + 1e-12
