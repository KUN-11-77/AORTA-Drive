from __future__ import annotations

import pytest

from aorta_drive.certification.geometry import Pose2D, VehicleRectangle
from aorta_drive.evaluation.polygon_reference import polygon_sweep_reference


BODY = VehicleRectangle(rear_overhang=1.0, front_extent=3.0, half_width=1.0)


def test_identical_polygon_references_have_zero_set_difference() -> None:
    poses = (Pose2D(0.0, 0.0, 0.0), Pose2D(2.0, 0.0, 0.1))
    result = polygon_sweep_reference(BODY, poses, poses, 8)
    assert result.challenger_minus_native_area_m2 == 0.0
    assert result.native_minus_challenger_area_m2 == 0.0


def test_single_pose_translation_has_analytic_bidirectional_area() -> None:
    native = (Pose2D(0.0, 0.0, 0.0),)
    challenger = (Pose2D(0.0, 1.0, 0.0),)
    result = polygon_sweep_reference(BODY, challenger, native, 1)
    assert result.challenger_minus_native_area_m2 == pytest.approx(4.0)
    assert result.native_minus_challenger_area_m2 == pytest.approx(4.0)


def test_reference_rejects_invalid_mesh_or_yaw_contract() -> None:
    poses = (Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, 3.2))
    with pytest.raises(ValueError, match="subdivisions"):
        polygon_sweep_reference(BODY, poses[:1], poses[:1], 0)
    with pytest.raises(ValueError, match="unwrap"):
        polygon_sweep_reference(BODY, poses, poses, 1)

