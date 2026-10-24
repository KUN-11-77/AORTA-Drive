from __future__ import annotations

import numpy as np
import pytest

from aorta_drive.evaluation.operator_anatomy import (
    analyze_set_min_pair,
    analyze_trajectory_pair,
)


def test_discrete_support_duplicate_has_priority() -> None:
    result = analyze_set_min_pair((0, 1), (1, 0), np.array((0.2, -0.4, 1.0)))
    assert result.category == "discrete_support_duplicate"
    assert result.common_minimum_controls_both
    assert result.contrast == 0.0


def test_common_minimum_annihilates_distinct_branches() -> None:
    result = analyze_set_min_pair((0, 1), (0, 2), np.array((-1.0, 0.4, 0.8)))
    assert result.category == "common_minimum_domination"
    assert result.intersection_node_count == 1
    assert result.symmetric_difference_node_count == 2
    assert result.contrast == 0.0


def test_distinct_minimum_teacher_tie_is_separate() -> None:
    result = analyze_set_min_pair((0, 1), (0, 2), np.array((1.0, -0.5, -0.5)))
    assert result.category == "distinct_minimum_teacher_tie"
    assert not result.common_minimum_controls_both


def test_nonzero_contrast_is_not_absorbed_by_tie_categories() -> None:
    result = analyze_set_min_pair((0, 1), (0, 2), np.array((1.0, -0.5, -0.4)))
    assert result.category == "nonzero_contrast"
    assert result.contrast == pytest.approx(0.1)


def test_trajectory_anatomy_uses_grid_derived_thresholds() -> None:
    reference = np.zeros((4, 3), dtype=np.float64)
    comparison = reference.copy()
    comparison[1:, 0] = (1.0e-8, 0.6, 1.1)
    comparison[3, 2] = np.pi / 2.0
    result = analyze_trajectory_pair(
        reference,
        comparison,
        grid_resolution_m=1.0,
    )
    assert not result.poses_exactly_equal
    assert result.first_center_separation_index_gt_1e_9_m == 1
    assert result.first_center_separation_index_gt_half_grid_m == 2
    assert result.first_center_separation_index_gt_one_grid_m == 3
    assert result.maximum_center_separation_m == pytest.approx(1.1)
    assert result.maximum_wrapped_yaw_separation_rad == pytest.approx(np.pi / 2.0)


def test_invalid_duplicate_or_empty_nodes_fail_closed() -> None:
    with pytest.raises(ValueError):
        analyze_set_min_pair((), (0,), np.ones(2))
    with pytest.raises(ValueError):
        analyze_set_min_pair((0, 0), (1,), np.ones(2))
