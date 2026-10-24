from __future__ import annotations

import copy
from typing import cast

import numpy as np
import numpy.typing as npt

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.shadow_decision import (
    binary_auroc,
    build_shadow_panel,
    evaluate_shadow_condition,
    rasterize_discrete_footprint_nodes,
    spearman_correlation,
    summarize_shadow_ledgers,
)


def _grid() -> RectangularGrid:
    return RectangularGrid(x_min=-4.0, y_min=-4.0, columns=8, rows=8, resolution=1.0)


def _body() -> VehicleRectangle:
    return VehicleRectangle(rear_overhang=1.0, front_extent=2.0, half_width=1.0)


def _poses(y: float = 0.0) -> npt.NDArray[np.float64]:
    return np.asarray(((0.0, y, 0.0), (1.0, y, 0.0)), dtype=np.float64)


def test_discrete_oriented_footprint_is_yaw_aware_and_rejects_clipping() -> None:
    horizontal = rasterize_discrete_footprint_nodes(_poses(), grid=_grid(), body=_body())
    vertical_poses = _poses().copy()
    vertical_poses[:, 2] = np.pi / 2
    vertical = rasterize_discrete_footprint_nodes(vertical_poses, grid=_grid(), body=_body())
    assert horizontal.size > 0
    assert vertical.size > 0
    assert not np.array_equal(horizontal, vertical)

    clipped = _poses().copy()
    clipped[:, 0] = 3.5
    with np.testing.assert_raises_regex(ValueError, "leaves the frozen ROI"):
        rasterize_discrete_footprint_nodes(clipped, grid=_grid(), body=_body())


def test_panel_keeps_invalid_challenger_but_requires_native_and_supply() -> None:
    trajectories = np.stack((_poses(), _poses(1.0), _poses(2.0)))
    panel = build_shadow_panel(
        trajectories,
        challenger_eligible=(False, True),
        identifiers=("native", "left", "right"),
        deterministic_execution=True,
        grid=_grid(),
        body=_body(),
    )
    assert panel.valid
    assert [candidate.valid for candidate in panel.candidates] == [True, False, True]
    assert [candidate.registered_index for candidate in panel.query_candidates] == [0, 2]

    nondeterministic = build_shadow_panel(
        trajectories,
        challenger_eligible=(True, True),
        identifiers=("native", "left", "right"),
        deterministic_execution=False,
        grid=_grid(),
        body=_body(),
    )
    assert not nondeterministic.valid
    assert nondeterministic.reason == "native_invalid"


def test_shadow_metrics_obey_exact_bounds_and_offset_invariance() -> None:
    trajectories = np.stack((_poses(), _poses(1.0)))
    panel = build_shadow_panel(
        trajectories,
        challenger_eligible=(True,),
        identifiers=("native", "challenger"),
        deterministic_execution=True,
        grid=_grid(),
        body=_body(),
    )
    teacher = np.arange(64, dtype=np.float64).reshape(8, 8) / 10.0
    prediction = teacher.copy()
    prediction[2:5, 2:5] += 0.3
    first = evaluate_shadow_condition(prediction, teacher, panel, dino_drift=0.25)
    shifted = evaluate_shadow_condition(prediction + 11.0, teacher - 7.0, panel, dino_drift=0.25)
    assert first["oscillation_bound_verified"]
    assert first["two_eta_regret_bound_verified"]
    assert np.isclose(
        float(first["decision_contrast_error_m"]),
        float(shifted["decision_contrast_error_m"]),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert first["teacher_selected_registered_index"] == shifted[
        "teacher_selected_registered_index"
    ]
    assert first["predicted_selected_registered_index"] == shifted[
        "predicted_selected_registered_index"
    ]


def test_tie_aware_spearman_and_auroc() -> None:
    assert np.isclose(spearman_correlation((1.0, 2.0, 3.0), (2.0, 4.0, 8.0)), 1.0)
    assert np.isclose(spearman_correlation((1.0, 2.0, 3.0), (8.0, 4.0, 2.0)), -1.0)
    assert binary_auroc((0.1, 0.2, 0.8, 0.9), (0, 0, 1, 1)) == 1.0
    assert binary_auroc((0.5, 0.5), (0, 1)) == 0.5
    assert np.isnan(binary_auroc((0.1, 0.2), (0, 0)))


def _synthetic_ledgers(*, groups: int = 12, anchors: int = 4) -> list[dict[str, object]]:
    ledgers: list[dict[str, object]] = []
    for group_index in range(groups):
        severity = float(group_index + 1)
        for anchor in range(anchors):
            condition_metrics = {}
            for condition_index, condition in enumerate(("clean", "appearance")):
                scale = severity + 0.1 * anchor + 0.01 * condition_index
                condition_metrics[condition] = {
                    "decision_contrast_error_m": scale,
                    "edge_mae_m": float((group_index * 7 + anchor * 3 + condition_index) % 11),
                    "centered_node_mae_m": float(
                        (group_index * 5 + anchor + condition_index * 2) % 13
                    ),
                    "dino_drift": float((group_index * 3 + anchor * 5 + condition_index) % 17),
                    "teacher_regret_m": scale,
                    "ranking_flip": group_index >= groups // 2,
                    "exact_teacher_tie": False,
                }
            ledgers.append(
                {
                    "recording_group": f"group-{group_index:02d}",
                    "anchor_index": anchor,
                    "panel_valid": True,
                    "conditions": condition_metrics,
                }
            )
    return ledgers


def test_group_first_summary_can_support_only_decision_diagnostic() -> None:
    summary = summarize_shadow_ledgers(
        _synthetic_ledgers(),
        conditions=("clean", "appearance"),
        seed=3,
        replicates=300,
        expected_groups=12,
        anchors_per_group=4,
        minimum_valid_anchors_per_group=3,
        minimum_valid_bootstrap_fraction=0.90,
    )
    assert summary["coverage"]["passed"]
    assert summary["association"]["spearman"]["decision_contrast_error_m"]["estimate"] == 1.0
    assert summary["association"]["flip_auroc"]["decision_contrast_error_m"]["estimate"] == 1.0


def test_panel_coverage_failure_retains_invalid_anchors() -> None:
    ledgers = _synthetic_ledgers(groups=3, anchors=4)
    for row in ledgers:
        if row["recording_group"] == "group-01" and cast(int, row["anchor_index"]) < 2:
            row["panel_valid"] = False
            row["conditions"] = None
    summary = summarize_shadow_ledgers(
        ledgers,
        conditions=("clean", "appearance"),
        replicates=20,
        expected_groups=3,
        anchors_per_group=4,
        minimum_valid_anchors_per_group=3,
    )
    assert summary["decision"] == "INSUFFICIENT_PANEL_COVERAGE"
    assert summary["coverage"]["valid_anchor_count_by_group"]["group-01"] == 2


def test_duplicate_anchor_and_metrics_on_invalid_panel_fail_closed() -> None:
    ledgers = _synthetic_ledgers(groups=3, anchors=4)
    duplicate = copy.deepcopy(ledgers)
    duplicate.append(copy.deepcopy(duplicate[0]))
    with np.testing.assert_raises_regex(ValueError, "duplicate"):
        summarize_shadow_ledgers(
            duplicate,
            conditions=("clean", "appearance"),
            replicates=20,
            expected_groups=3,
            anchors_per_group=4,
            minimum_valid_anchors_per_group=3,
        )

    ledgers[0]["panel_valid"] = False
    with np.testing.assert_raises_regex(ValueError, "must not carry"):
        summarize_shadow_ledgers(
            ledgers,
            conditions=("clean", "appearance"),
            replicates=20,
            expected_groups=3,
            anchors_per_group=4,
            minimum_valid_anchors_per_group=3,
        )
