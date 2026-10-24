from __future__ import annotations

import numpy as np

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.support_adequacy import (
    evaluate_prefix_panel,
    group_equal_descriptive_difference,
    rectangle_corner_series,
    summarize_prefix_qualification,
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


def test_prefix_panel_changes_only_when_support_is_crossed() -> None:
    native = np.asarray(((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)), dtype=np.float64)
    challenger = np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype=np.float64)
    realized = np.stack((native, challenger))
    short = evaluate_prefix_panel(
        realized,
        challenger_eligible=(True,),
        prefix_index=0,
        deterministic_execution=True,
        grid=_grid(),
        body=_body(),
    )
    long = evaluate_prefix_panel(
        realized,
        challenger_eligible=(True,),
        prefix_index=1,
        deterministic_execution=True,
        grid=_grid(),
        body=_body(),
    )
    assert short.panel_valid
    assert not long.panel_valid
    assert long.panel_reason == "native_invalid"


def test_prefix_summary_selects_largest_all_group_feasible_prefix() -> None:
    rows = []
    for group in ("a", "b"):
        for anchor in range(2):
            rows.append(
                {
                    "recording_group": group,
                    "anchor_index": anchor,
                    "prefix_statuses": {
                        "1": {
                            "panel_valid": True,
                            "panel_reason": "valid",
                            "candidate_reasons": ["valid", "valid"],
                            "candidate_minimum_slack_m": [1.0, 1.0],
                        },
                        "2": {
                            "panel_valid": group == "a",
                            "panel_reason": "valid" if group == "a" else "native_invalid",
                            "candidate_reasons": [
                                "valid" if group == "a" else "oriented footprint leaves the frozen ROI",
                                "valid",
                            ],
                            "candidate_minimum_slack_m": [0.0 if group == "a" else -1.0, 1.0],
                        },
                    },
                }
            )
    summary = summarize_prefix_qualification(
        rows,
        group_names=("a", "b"),
        prefix_indices=(1, 2),
        anchors_per_group=2,
        minimum_valid_anchors_per_group=2,
    )
    assert summary["selected_prefix_index"] == 1
    assert summary["by_prefix"]["2"]["groups_meeting_minimum_coverage"] == 1


def test_prefix_summary_fails_closed_when_no_prefix_is_feasible() -> None:
    rows = [
        {
            "recording_group": "a",
            "anchor_index": 0,
            "prefix_statuses": {
                "1": {
                    "panel_valid": False,
                    "panel_reason": "no_valid_challenger",
                    "candidate_reasons": ["valid", "intent_ineligible"],
                    "candidate_minimum_slack_m": [1.0, 1.0],
                }
            },
        }
    ]
    summary = summarize_prefix_qualification(
        rows,
        group_names=("a",),
        prefix_indices=(1,),
        anchors_per_group=1,
        minimum_valid_anchors_per_group=1,
    )
    assert summary["selected_prefix_index"] is None
    assert summary["decision"] == "STOP_CURRENT_FIXED_GRID_SHADOW_ROUTE"
