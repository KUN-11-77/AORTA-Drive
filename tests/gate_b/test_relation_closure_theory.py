from __future__ import annotations

import itertools

import numpy as np
import pytest

from aorta_drive.theory.relation_guarantees import (
    RelationDecisionReason,
    arbitrate_native_relative,
    group_simultaneous_nonconformity,
    group_split_quantile_or_infinity,
)


def test_native_relative_arbitration_uses_the_largest_strict_lower_bound() -> None:
    decision = arbitrate_native_relative(
        point_margins=(0.9, 0.7, 0.8),
        radii=(0.8, 0.1, 0.4),
    )
    assert decision.selected_index == 2
    assert decision.intervened
    assert decision.reason is RelationDecisionReason.REPLACE_WITH_POSITIVE_LOWER_BOUND
    assert decision.selected_lower_bound == pytest.approx(0.6)


def test_native_relative_arbitration_is_strict_and_fail_closed() -> None:
    no_challenger = arbitrate_native_relative((), ())
    assert no_challenger.selected_index == 0
    assert no_challenger.reason is RelationDecisionReason.KEEP_NATIVE_NO_CHALLENGER

    boundary = arbitrate_native_relative((0.2, -0.1), (0.2, 0.0))
    assert boundary.selected_index == 0
    assert boundary.reason is RelationDecisionReason.KEEP_NATIVE_NO_POSITIVE_LOWER_BOUND

    invalid = arbitrate_native_relative((np.nan,), (0.1,))
    assert invalid.selected_index == 0
    assert invalid.reason is RelationDecisionReason.KEEP_NATIVE_INVALID_NUMERICS

    with pytest.raises(ValueError, match="aligned one-dimensional"):
        arbitrate_native_relative((0.1, 0.2), (0.1,))


def test_positive_lower_bound_implies_true_native_improvement_exhaustive() -> None:
    values = (-1.0, -0.25, 0.0, 0.25, 1.0)
    errors = (-0.5, 0.0, 0.5)
    for true_margin in itertools.product(values, repeat=2):
        truth = np.asarray(true_margin, dtype=np.float64)
        for residual in itertools.product(errors, repeat=2):
            error = np.asarray(residual, dtype=np.float64)
            point = truth + error
            radius = np.abs(error)
            decision = arbitrate_native_relative(point, radius)
            if decision.intervened:
                assert truth[decision.selected_index - 1] > 0.0


def test_appearance_accuracy_bounds_relation_drift_but_not_conversely() -> None:
    truth = np.asarray((-0.7, 0.2, 1.1), dtype=np.float64)
    epsilon = 0.15
    residuals = (-epsilon, 0.0, epsilon)
    for clean_error in itertools.product(residuals, repeat=truth.size):
        clean = truth + np.asarray(clean_error)
        for shifted_error in itertools.product(residuals, repeat=truth.size):
            shifted = truth + np.asarray(shifted_error)
            assert np.max(np.abs(shifted - clean)) <= np.nextafter(
                2.0 * epsilon,
                np.inf,
            )

    collapsed_clean = np.zeros_like(truth)
    collapsed_shifted = np.zeros_like(truth)
    assert np.array_equal(collapsed_clean, collapsed_shifted)
    assert np.max(np.abs(collapsed_clean - truth)) > 0.0


def test_group_nonconformity_absorbs_all_within_group_comparisons() -> None:
    predicted = np.asarray([[0.0, 0.4], [0.6, -0.2]])
    truth = np.asarray([[0.1, 0.0], [0.5, 0.0]])
    scales = np.asarray([[0.1, 0.2], [0.1, 0.4]])
    assert group_simultaneous_nonconformity(predicted, truth, scales) == pytest.approx(
        2.0
    )


def test_group_split_quantile_is_finite_at_95_and_vacuous_at_99_percent() -> None:
    scores = np.arange(42, dtype=np.float64)
    assert group_split_quantile_or_infinity(scores, alpha=0.05) == 40.0
    assert np.isinf(group_split_quantile_or_infinity(scores, alpha=0.01))


def test_group_calibration_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        group_simultaneous_nonconformity((0.0,), (0.0, 1.0), (1.0,))
    with pytest.raises(ValueError, match="scales positive"):
        group_simultaneous_nonconformity((0.0,), (0.0,), (0.0,))
    with pytest.raises(ValueError, match="alpha"):
        group_split_quantile_or_infinity((0.0,), alpha=0.0)
