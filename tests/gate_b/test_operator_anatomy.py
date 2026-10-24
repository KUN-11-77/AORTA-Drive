from __future__ import annotations

import numpy as np
import pytest

from aorta_drive.evaluation.operator_anatomy import (
    analyze_set_min_pair,
    analyze_trajectory_pair,
    evaluate_spatial_fork,
    evaluate_temporal_fork,
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


def test_spatial_fork_retains_shared_gate_and_branch_contrast() -> None:
    result = evaluate_spatial_fork(
        (0, 1),
        (0, 2),
        np.array((0.5, -0.2, 0.4), dtype=np.float64),
    )
    assert result.valid
    assert result.shared_strictly_safe
    assert not result.reference_branch_strictly_safe
    assert result.comparison_branch_strictly_safe
    assert result.road_rescue_opportunity
    assert result.branch_contrast == pytest.approx(0.6)


def test_spatial_fork_does_not_call_unsafe_shared_region_a_rescue() -> None:
    result = evaluate_spatial_fork(
        (0, 1),
        (0, 2),
        np.array((-0.1, -0.2, 0.4), dtype=np.float64),
    )
    assert result.valid
    assert not result.shared_strictly_safe
    assert not result.road_rescue_opportunity


def test_temporal_fork_uses_first_exact_footprint_difference() -> None:
    result = evaluate_temporal_fork(
        ((0, 1), (1, 2), (2, 3)),
        ((1, 0), (1, 4), (4, 5)),
        np.array((1.0, 0.8, -0.2, -0.4, 0.3, 0.5), dtype=np.float64),
    )
    assert result.valid
    assert result.fork_index == 1
    assert result.shared_node_count == 2
    assert result.reference_branch_minimum == pytest.approx(-0.4)
    assert result.comparison_branch_minimum == pytest.approx(0.3)
    assert result.road_rescue_opportunity


def test_temporal_fork_rejects_no_fork_and_empty_trunk() -> None:
    no_fork = evaluate_temporal_fork(
        ((0,), (1,)),
        ((0,), (1,)),
        np.ones(2),
    )
    assert not no_fork.valid
    assert no_fork.reason == "no_discrete_temporal_fork"
    empty_trunk = evaluate_temporal_fork(
        ((0,), (1,)),
        ((2,), (3,)),
        np.ones(4),
    )
    assert not empty_trunk.valid
    assert empty_trunk.reason == "empty_shared_or_branch_support"


def test_fork_contrast_is_gauge_invariant_but_absolute_flags_are_not() -> None:
    labels = np.array((0.1, -0.2, 0.4), dtype=np.float64)
    shifted = labels - 1.0
    original = evaluate_spatial_fork((0, 1), (0, 2), labels)
    translated = evaluate_spatial_fork((0, 1), (0, 2), shifted)
    assert original.branch_contrast == pytest.approx(translated.branch_contrast)
    assert original.shared_strictly_safe
    assert not translated.shared_strictly_safe
