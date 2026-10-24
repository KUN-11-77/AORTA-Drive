from __future__ import annotations

import numpy as np

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.support_adequacy import (
    group_equal_descriptive_difference,
    rectangle_corner_series,
    support_profile,
)


def _grid() -> RectangularGrid:
    return RectangularGrid(x_min=-2.0, y_min=-2.0, columns=4, rows=4, resolution=1.0)


def _body() -> VehicleRectangle:
    return VehicleRectangle(rear_overhang=0.5, front_extent=1.0, half_width=0.5)


def test_rectangle_corner_series_respects_yaw() -> None:
    poses = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, np.pi / 2)), dtype=np.float64)
    corners = rectangle_corner_series(poses, body=_body())
    assert corners.shape == (2, 4, 2)
    assert np.allclose(np.max(corners[0], axis=0), (1.0, 0.5))
    assert np.allclose(np.max(corners[1], axis=0), (0.5, 1.0))


def test_support_profile_attributes_first_boundary_and_expansion() -> None:
    poses = np.asarray(((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)), dtype=np.float64)
    profile = support_profile(poses, grid=_grid(), body=_body())
    assert profile.first_violation_index == 1
    assert profile.first_violation_boundaries == ("x_max",)
    assert profile.required_expansion_m["x_max"] == 0.5
    assert profile.minimum_slack_m == -0.5
    assert profile.initial_translation_residual_m == 0.0
    assert profile.initial_yaw_residual_rad == 0.0


def test_support_profile_handles_wrapped_yaw() -> None:
    poses = np.asarray(
        ((0.0, 0.0, np.pi - 0.1), (0.1, 0.0, -np.pi + 0.1)), dtype=np.float64
    )
    profile = support_profile(poses, grid=_grid(), body=_body())
    assert np.isclose(profile.total_yaw_variation_rad, 0.2)


def test_group_equal_difference_uses_groups_not_anchor_pooling() -> None:
    rows = [
        {"recording_group": "a", "panel_reason": "valid", "value": 0.0},
        {"recording_group": "a", "panel_reason": "native_invalid", "value": 2.0},
        {"recording_group": "b", "panel_reason": "valid", "value": 10.0},
        {"recording_group": "b", "panel_reason": "native_invalid", "value": 14.0},
    ]
    result = group_equal_descriptive_difference(
        rows, value_key="value", invalid_reasons=frozenset(("native_invalid",))
    )
    assert result["paired_group_count"] == 2
    assert result["mean"] == 3.0
    assert result["median"] == 3.0
