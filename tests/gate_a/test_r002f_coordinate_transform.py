"""R002F: proof-carrying ego-local to global coordinate transform."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from aorta_drive.certification.geometry import Pose2D
from aorta_drive.certification.transform import certify_ego_to_global


def test_zero_heading_exact_case_has_no_error() -> None:
    certificate = certify_ego_to_global((3.0, -2.0), Pose2D(10.0, 20.0, 0.0))
    assert certificate.x == 13.0
    assert certificate.y == 18.0
    assert certificate.error_radius == 0.0
    assert certificate.x_interval.lower == certificate.x_interval.upper == Fraction(13)
    assert certificate.y_interval.lower == certificate.y_interval.upper == Fraction(18)


def test_registered_random_transforms_separate_proof_radius_from_libm_diagnostic() -> None:
    rng = np.random.default_rng(20261030)
    for _ in range(500):
        x, y = rng.uniform(-1_000_000.0, 1_000_000.0, size=2)
        yaw = float(rng.uniform(-math.pi, math.pi))
        qx, qy = rng.uniform(-50.0, 50.0, size=2)
        certificate = certify_ego_to_global(
            (float(qx), float(qy)),
            Pose2D(float(x), float(y), yaw),
        )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        observed_x = x + cosine * qx - sine * qy
        observed_y = y + sine * qx + cosine * qy
        observed_error = math.hypot(observed_x - certificate.x, observed_y - certificate.y)
        coordinate_scale = max(1.0, abs(x), abs(y))
        compatibility_budget = certificate.error_radius + 32.0 * math.ulp(coordinate_scale)
        assert observed_error <= compatibility_budget
        assert certificate.error_radius < 2e-10


def test_transform_radius_is_outward_safe_for_interval_corners() -> None:
    certificate = certify_ego_to_global(
        (49.9, -37.7),
        Pose2D(365432.125, 4123456.75, 2.7),
    )
    center_x = Fraction.from_float(certificate.x)
    center_y = Fraction.from_float(certificate.y)
    radius_squared = Fraction.from_float(certificate.error_radius) ** 2
    for x in (certificate.x_interval.lower, certificate.x_interval.upper):
        for y in (certificate.y_interval.lower, certificate.y_interval.upper):
            squared_error = (x - center_x) ** 2 + (y - center_y) ** 2
            assert squared_error <= radius_squared


def test_transform_rejects_out_of_protocol_inputs() -> None:
    pose = Pose2D(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="two finite"):
        certify_ego_to_global((math.inf, 0.0), pose)
    with pytest.raises(ValueError, match="4-radian"):
        certify_ego_to_global((0.0, 0.0), Pose2D(0.0, 0.0, 4.01))
