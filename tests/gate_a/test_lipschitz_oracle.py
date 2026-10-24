"""Tests for support-conditioned Lipschitz oracle sets."""

from __future__ import annotations

import numpy as np

from aorta_drive.certification.difference_constraints import (
    robust_set_min_lower_bound_directed,
)
from aorta_drive.certification.lipschitz_oracle import (
    build_oracle_lipschitz_problem,
    independent_candidate_contrast_interval,
)


def test_oracle_problem_contains_teacher_and_preserves_observed_intervals() -> None:
    labels = np.array([[0.0, 0.5, 1.0], [-0.25, 0.25, 0.75]])
    errors = np.full_like(labels, 0.01)
    observed = np.array([[True, False, True], [False, False, True]])
    problem = build_oracle_lipschitz_problem(
        labels,
        errors,
        observed,
        rows=2,
        columns=3,
        resolution=1.0,
    )
    assert problem.teacher_included
    assert problem.observed_node_count == 3
    assert np.all(problem.marginal_lower <= labels.reshape(-1))
    assert np.all(problem.marginal_upper >= labels.reshape(-1))
    assert problem.maximum_numerical_widening < 1e-12


def test_joint_set_is_never_wider_than_shared_node_box_on_query() -> None:
    labels = np.array([[0.0, 0.2, 0.4], [0.1, 0.3, 0.5]])
    errors = np.full_like(labels, 0.02)
    observed = np.array([[True, False, False], [False, False, True]])
    problem = build_oracle_lipschitz_problem(
        labels,
        errors,
        observed,
        rows=2,
        columns=3,
        resolution=1.0,
    )
    values = problem.node_values_with_reference
    challenger = (1, 2, 5)
    native = (0, 1, 4)
    joint_lower = robust_set_min_lower_bound_directed(
        values,
        problem.joint_constraints,
        challenger,
        native,
    )
    joint_upper = -robust_set_min_lower_bound_directed(
        values,
        problem.joint_constraints,
        native,
        challenger,
    )
    box_lower = robust_set_min_lower_bound_directed(
        values,
        problem.box_constraints,
        challenger,
        native,
    )
    box_upper = -robust_set_min_lower_bound_directed(
        values,
        problem.box_constraints,
        native,
        challenger,
    )
    assert joint_lower >= box_lower
    assert joint_upper <= box_upper
    assert joint_lower <= joint_upper


def test_candidate_independent_interval_is_conservative_for_shared_box() -> None:
    labels = np.array([[0.0, 0.2], [0.4, 0.6]])
    errors = np.full_like(labels, 0.01)
    observed = np.array([[True, False], [False, True]])
    problem = build_oracle_lipschitz_problem(
        labels,
        errors,
        observed,
        rows=2,
        columns=2,
        resolution=1.0,
    )
    challenger = (0, 1)
    native = (0, 2)
    shared_lower = robust_set_min_lower_bound_directed(
        problem.node_values_with_reference,
        problem.box_constraints,
        challenger,
        native,
    )
    shared_upper = -robust_set_min_lower_bound_directed(
        problem.node_values_with_reference,
        problem.box_constraints,
        native,
        challenger,
    )
    independent_lower, independent_upper = independent_candidate_contrast_interval(
        problem.marginal_lower,
        problem.marginal_upper,
        challenger,
        native,
    )
    assert independent_lower <= shared_lower + 1e-12
    assert independent_upper >= shared_upper - 1e-12


def test_no_observed_nodes_is_rejected() -> None:
    with np.testing.assert_raises_regex(ValueError, "at least one observed"):
        build_oracle_lipschitz_problem(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2), dtype=bool),
            rows=2,
            columns=2,
            resolution=1.0,
        )
