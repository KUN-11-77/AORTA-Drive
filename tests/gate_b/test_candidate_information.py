from __future__ import annotations

import numpy as np

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.candidate_information import (
    evaluate_native_challenger_information,
    rasterize_unbounded_footprint_lattice,
    summarize_candidate_information,
)
from aorta_drive.evaluation.shadow_decision import rasterize_discrete_footprint_nodes


BODY = VehicleRectangle(rear_overhang=1.0, front_extent=2.0, half_width=0.75)


def test_unbounded_lattice_matches_bounded_grid_when_supported() -> None:
    poses = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.25, 0.1]], dtype=np.float64)
    grid = RectangularGrid(x_min=-4.0, y_min=-4.0, columns=8, rows=8, resolution=1.0)
    bounded = rasterize_discrete_footprint_nodes(poses, grid=grid, body=BODY)
    centers = grid.centers()[bounded]
    expected = {
        (int(round(x - 0.5)), int(round(y - 0.5))) for x, y in centers
    }
    unbounded = rasterize_unbounded_footprint_lattice(poses, body=BODY)
    assert {(int(x), int(y)) for x, y in unbounded} == expected


def test_prefix_node_equivalence_detects_late_structural_counterexample() -> None:
    native = np.zeros((5, 3), dtype=np.float64)
    native[:, 0] = np.arange(5, dtype=np.float64)
    challenger = native.copy()
    challenger[3:, 1] = 2.0
    result = evaluate_native_challenger_information(
        native,
        challenger,
        challenger_index=1,
        prefix_index=2,
        divergence_thresholds_m=(0.5, 1.0),
        body=BODY,
    )
    assert result.prefix_nodes_equal
    assert not result.full_nodes_equal
    assert result.prefix_equal_full_distinct
    assert result.first_divergence_index_by_threshold["0.5"] == 3
    assert result.late_divergence_by_threshold["0.5"]


def test_summary_falsifies_prefix_and_selects_minimum_cost_roi() -> None:
    rows = []
    for group in ("g0", "g1"):
        for anchor in range(2):
            rows.append(
                {
                    "recording_group": group,
                    "anchor_index": anchor,
                    "pair_information": [
                        {
                            "prefix_equal_full_distinct": group == "g0" and anchor == 0,
                            "late_divergence_by_threshold": {"0.5": anchor == 0},
                            "full_symmetric_difference_nodes": 4,
                            "full_distinguishing_fraction_absent_from_prefix": 0.5,
                        }
                    ],
                    "roi_statuses": {
                        "small": {"panel_valid": anchor == 0},
                        "shifted": {"panel_valid": True},
                        "large": {"panel_valid": True},
                    },
                }
            )
    result = summarize_candidate_information(
        rows,
        group_names=("g0", "g1"),
        anchors_per_group=2,
        prefix_index=2,
        primary_divergence_threshold_m=0.5,
        divergence_thresholds_m=(0.5,),
        roi_precedence=("small", "shifted", "large"),
        roi_node_counts={"small": 4, "shifted": 4, "large": 8},
        minimum_valid_anchors_per_group=2,
        bootstrap_seed=7,
        bootstrap_replicates=100,
    )
    assert result["prefix_information_verdict"] == "PREFIX_UNIVERSAL_SUFFICIENCY_FALSIFIED"
    assert result["selected_minimum_cost_full_horizon_roi"] == "shifted"
    assert result["roi_verdict"] == "FIT_FULL_HORIZON_ROI_SELECTED"
