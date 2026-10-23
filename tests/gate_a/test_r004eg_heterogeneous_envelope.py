"""R004E-G: heterogeneous envelope, split, and arithmetic falsification tests."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from aorta_drive.certification.calibration import split_conformal_threshold
from aorta_drive.certification.graph import UndirectedEdge, robust_set_min_lower_bound
from aorta_drive.certification.heterogeneous import (
    deterministic_group_roles,
    outward_abs_edge_residuals,
    outward_edge_budgets,
    outward_session_max_score,
    outward_standardized_scores,
)


SCALE_MIN = 2.0**-10
SCALE_MAX = 2.0**4


def test_registered_role_assignment_is_disjoint_exhaustive_and_order_invariant() -> None:
    groups = [f"group-{index:02d}" for index in range(66)]
    forward = deterministic_group_roles(
        groups,
        namespace="AORTA-R004EG-roles-v1",
        fit_count=42,
        selection_count=8,
        proxy_cal_count=16,
    )
    reverse = deterministic_group_roles(
        list(reversed(groups)),
        namespace="AORTA-R004EG-roles-v1",
        fit_count=42,
        selection_count=8,
        proxy_cal_count=16,
    )
    assert forward == reverse
    assert (len(forward.fit), len(forward.selection), len(forward.proxy_cal)) == (42, 8, 16)
    assigned = forward.fit + forward.selection + forward.proxy_cal
    assert len(set(assigned)) == 66
    assert set(assigned) == set(groups)


def test_role_assignment_rejects_duplicates_or_non_exhaustive_counts() -> None:
    with pytest.raises(ValueError, match="unique"):
        deterministic_group_roles(
            ["same", "same"],
            namespace="x",
            fit_count=1,
            selection_count=0,
            proxy_cal_count=1,
        )
    with pytest.raises(ValueError, match="exhaust"):
        deterministic_group_roles(
            ["a", "b"],
            namespace="x",
            fit_count=1,
            selection_count=0,
            proxy_cal_count=0,
        )


def test_four_term_interval_catches_registered_cancellation_counterexample() -> None:
    potentials = np.array([[-2.0**-53, 1.0]])
    labels = np.array([[0.0, 1.0]])
    residual = outward_abs_edge_residuals(potentials, labels, [0], [1])
    exact = abs(
        (Fraction.from_float(1.0) - Fraction.from_float(-2.0**-53))
        - (Fraction.from_float(1.0) - Fraction.from_float(0.0))
    )
    assert exact == Fraction.from_float(2.0**-53)
    assert Fraction.from_float(float(residual[0, 0])) >= exact
    naive = abs(
        (potentials[0, 1] - potentials[0, 0])
        - (labels[0, 1] - labels[0, 0])
    )
    assert naive == 0.0


def test_four_term_residual_and_standardization_are_never_optimistic() -> None:
    rng = np.random.default_rng(20261024)
    for _ in range(500):
        values = rng.normal(size=4) * 2.0 ** int(rng.integers(-30, 31))
        scale = float(2.0 ** int(rng.integers(-9, 5)))
        potentials = np.array([[values[0], values[1]]])
        labels = np.array([[values[2], values[3]]])
        upper = outward_abs_edge_residuals(potentials, labels, [0], [1])
        score = outward_standardized_scores(
            upper,
            np.array([[scale]]),
            scale_min=SCALE_MIN,
            scale_max=SCALE_MAX,
        )
        exact_residual = abs(
            (Fraction.from_float(float(values[1])) - Fraction.from_float(float(values[0])))
            - (Fraction.from_float(float(values[3])) - Fraction.from_float(float(values[2])))
        )
        exact_score = exact_residual / Fraction.from_float(scale)
        assert Fraction.from_float(float(upper[0, 0])) >= exact_residual
        assert Fraction.from_float(float(score[0, 0])) >= exact_score


def test_edge_budget_multiplication_is_never_optimistic() -> None:
    rng = np.random.default_rng(20261025)
    for _ in range(500):
        threshold = float(2.0 ** int(rng.integers(-30, 31)) * rng.uniform(0.5, 1.0))
        scale = float(2.0 ** int(rng.integers(-9, 5)) * rng.uniform(0.5, 1.0))
        scale = min(max(scale, SCALE_MIN), SCALE_MAX)
        budget = outward_edge_budgets(
            threshold,
            np.array([scale]),
            scale_min=SCALE_MIN,
            scale_max=SCALE_MAX,
        )[0]
        exact = Fraction.from_float(threshold) * Fraction.from_float(scale)
        assert Fraction.from_float(float(budget)) >= exact


def test_zero_threshold_remains_exact_zero() -> None:
    budgets = outward_edge_budgets(
        0.0,
        np.array([SCALE_MIN, 1.0, SCALE_MAX]),
        scale_min=SCALE_MIN,
        scale_max=SCALE_MAX,
    )
    assert np.array_equal(budgets, np.zeros(3))


@pytest.mark.parametrize(
    "bad_scales",
    [
        np.array([[0.0]]),
        np.array([[-1.0]]),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[SCALE_MIN / 2.0]]),
        np.array([[SCALE_MAX * 2.0]]),
    ],
)
def test_invalid_scales_are_rejected_without_filtering(bad_scales: np.ndarray) -> None:
    with pytest.raises(ValueError):
        outward_session_max_score(
            [[0.0, 1.0]],
            [[0.0, 0.0]],
            [0],
            [1],
            bad_scales,
            scale_min=SCALE_MIN,
            scale_max=SCALE_MAX,
        )


def test_nonfinite_threshold_is_a_failed_calibration_not_a_budget() -> None:
    with pytest.raises(ValueError, match="finite"):
        outward_edge_budgets(
            math.inf,
            [1.0],
            scale_min=SCALE_MIN,
            scale_max=SCALE_MAX,
        )


@pytest.mark.parametrize(("n", "alpha", "covered"), [(9, 0.1, 9), (16, 0.1, 16), (19, 0.05, 19)])
def test_joint_exchangeable_rank_enumeration(n: int, alpha: float, covered: int) -> None:
    successes = 0
    for new_rank in range(n + 1):
        calibration = np.array([rank for rank in range(n + 1) if rank != new_rank], dtype=float)
        threshold = split_conformal_threshold(calibration, alpha)
        successes += int(float(new_rank) <= threshold)
    assert successes == covered
    assert successes / (n + 1) >= 1.0 - alpha


def test_tail_augmented_scale_loss_moves_the_stationary_point() -> None:
    absolute_residual = 2.0
    tail_weight = 0.5
    proper_optimum = absolute_residual
    augmented_optimum = (1.0 + tail_weight) * absolute_residual

    def derivative(scale: float, multiplier: float) -> float:
        return -(multiplier * absolute_residual) / scale**2 + 1.0 / scale

    assert derivative(proper_optimum, 1.0) == 0.0
    assert derivative(augmented_optimum, 1.0 + tail_weight) == 0.0
    assert augmented_optimum > proper_optimum


def test_heterogeneous_budgets_remain_conservative_in_graph_readout() -> None:
    threshold = 0.75
    scales = np.array([0.5, 2.0])
    budgets = outward_edge_budgets(
        threshold,
        scales,
        scale_min=SCALE_MIN,
        scale_max=SCALE_MAX,
    )
    values = [0.2, -0.1, 0.7]
    edges = [
        UndirectedEdge(0, 1, float(budgets[0])),
        UndirectedEdge(1, 2, float(budgets[1])),
    ]
    computed = robust_set_min_lower_bound(values, edges, (2,), (0,))
    exact = (
        Fraction.from_float(values[2])
        - Fraction.from_float(values[0])
        - Fraction.from_float(float(budgets[0]))
        - Fraction.from_float(float(budgets[1]))
    )
    assert Fraction.from_float(computed) <= exact


def test_power_of_two_scale_gauge_is_conservative_on_both_parameterizations() -> None:
    potentials = np.array([[0.0, 0.25, 0.75]])
    labels = np.zeros_like(potentials)
    sources = np.array([0, 1])
    targets = np.array([1, 2])
    scales = np.array([[0.5, 1.0]])
    factor = 2.0
    score = outward_session_max_score(
        potentials,
        labels,
        sources,
        targets,
        scales,
        scale_min=SCALE_MIN,
        scale_max=SCALE_MAX,
    )
    rescaled_score = outward_session_max_score(
        potentials,
        labels,
        sources,
        targets,
        scales * factor,
        scale_min=SCALE_MIN,
        scale_max=SCALE_MAX,
    )
    budgets = outward_edge_budgets(
        score,
        scales,
        scale_min=SCALE_MIN,
        scale_max=SCALE_MAX,
    )
    rescaled_budgets = outward_edge_budgets(
        rescaled_score,
        scales * factor,
        scale_min=SCALE_MIN,
        scale_max=SCALE_MAX,
    )
    exact = [Fraction(1, 4), Fraction(1, 2)]
    for original, rescaled, reference in zip(budgets[0], rescaled_budgets[0], exact, strict=True):
        assert Fraction.from_float(float(original)) >= reference
        assert Fraction.from_float(float(rescaled)) >= reference
        spacing = max(math.ulp(float(original)), math.ulp(float(rescaled)))
        assert abs(float(original) - float(rescaled)) <= 8.0 * spacing
