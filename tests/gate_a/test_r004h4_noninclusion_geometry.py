"""R004H4 preflight: proof-carrying continuous non-inclusion geometry."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from aorta_drive.certification.geometry import (
    Pose2D,
    RectangularGrid,
    VehicleRectangle,
    certify_bidirectional_disk_noninclusion,
)
from aorta_drive.certification.transform import certify_global_to_ego


def test_global_to_ego_zero_heading_is_exact() -> None:
    certificate = certify_global_to_ego((13.0, 18.0), Pose2D(10.0, 20.0, 0.0))
    assert certificate.x == 3.0
    assert certificate.y == -2.0
    assert certificate.error_radius == 0.0
    assert certificate.x_interval.lower == certificate.x_interval.upper == Fraction(3)
    assert certificate.y_interval.lower == certificate.y_interval.upper == Fraction(-2)


def test_global_to_ego_interval_contains_libm_diagnostic() -> None:
    rng = np.random.default_rng(20261024)
    for _ in range(200):
        tx, ty = rng.uniform(-1_000_000.0, 1_000_000.0, size=2)
        yaw = float(rng.uniform(-math.pi, math.pi))
        gx, gy = rng.uniform(-1_000_000.0, 1_000_000.0, size=2)
        certificate = certify_global_to_ego(
            (float(gx), float(gy)), Pose2D(float(tx), float(ty), yaw)
        )
        dx, dy = gx - tx, gy - ty
        expected_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        expected_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        observed_error = math.hypot(
            expected_x - certificate.x, expected_y - certificate.y
        )
        scale = max(1.0, abs(dx), abs(dy))
        assert observed_error <= certificate.error_radius + 32.0 * math.ulp(scale)


def test_bidirectional_area_is_zero_for_identical_sweeps() -> None:
    body = VehicleRectangle(rear_overhang=1.127, front_extent=4.049, half_width=1.1485)
    grid = RectangularGrid(x_min=-16.0, y_min=-16.0, columns=64, rows=64, resolution=0.5)
    poses = (Pose2D(0.25, 0.25, 0.0), Pose2D(5.25, 0.25, 0.0))
    certificate = certify_bidirectional_disk_noninclusion(body, poses, poses, grid)
    assert certificate.challenger_minus_native_area_lower_m2 == 0.0
    assert certificate.native_minus_challenger_area_lower_m2 == 0.0


def test_bidirectional_area_has_sound_witnesses_for_separated_sweeps() -> None:
    body = VehicleRectangle(rear_overhang=1.127, front_extent=4.049, half_width=1.1485)
    grid = RectangularGrid(x_min=-20.0, y_min=-20.0, columns=80, rows=80, resolution=0.5)
    challenger = (Pose2D(-5.25, -5.25, 0.0), Pose2D(5.25, -5.25, 0.0))
    native = (Pose2D(-5.25, 5.25, 0.0), Pose2D(5.25, 5.25, 0.0))
    certificate = certify_bidirectional_disk_noninclusion(
        body, challenger, native, grid
    )
    assert certificate.challenger_minus_native_area_lower_m2 > 0.0
    assert certificate.native_minus_challenger_area_lower_m2 > 0.0
    assert certificate.challenger_minus_native_area_lower_m2 == (
        len(certificate.challenger_witness_cells) * grid.resolution**2
    )
    assert certificate.native_minus_challenger_area_lower_m2 == (
        len(certificate.native_witness_cells) * grid.resolution**2
    )


def test_pose_error_cannot_increase_a_noninclusion_lower_bound() -> None:
    body = VehicleRectangle(rear_overhang=1.127, front_extent=4.049, half_width=1.1485)
    grid = RectangularGrid(x_min=-20.0, y_min=-20.0, columns=80, rows=80, resolution=0.5)
    challenger = (Pose2D(-5.25, -4.25, 0.0), Pose2D(5.25, -4.25, 0.0))
    native = (Pose2D(-5.25, 4.25, 0.0), Pose2D(5.25, 4.25, 0.0))
    exact = certify_bidirectional_disk_noninclusion(body, challenger, native, grid)
    uncertain = certify_bidirectional_disk_noninclusion(
        body,
        challenger,
        native,
        grid,
        challenger_pose_error=0.2,
        native_pose_error=0.2,
    )
    assert (
        uncertain.challenger_minus_native_area_lower_m2
        <= exact.challenger_minus_native_area_lower_m2
    )
    assert (
        uncertain.native_minus_challenger_area_lower_m2
        <= exact.native_minus_challenger_area_lower_m2
    )
