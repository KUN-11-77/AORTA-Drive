"""R003: session calibration, authorization, and risk-event checks."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from aorta_drive.certification.calibration import (
    minimum_calibration_sessions,
    session_max_score,
    split_conformal_threshold,
)
from aorta_drive.certification.graph import UndirectedEdge, robust_set_min_lower_bound
from aorta_drive.certification.risk import authorization_mask, relative_margin_failure


@pytest.mark.parametrize(
    ("alpha", "minimum"),
    [(0.10, 9), (0.05, 19), (0.01, 99), (0.005, 199)],
)
def test_finite_threshold_sample_size(alpha: float, minimum: int) -> None:
    assert minimum_calibration_sessions(alpha) == minimum
    with pytest.raises(ValueError, match="not enough"):
        split_conformal_threshold(np.arange(minimum - 1, dtype=float), alpha)
    threshold = split_conformal_threshold(np.arange(minimum, dtype=float), alpha)
    assert threshold == minimum - 1


def test_session_score_uses_maximum_over_time_and_edges() -> None:
    predicted = np.array([[0.0, 0.0], [0.1, 2.0]])
    labels = np.zeros_like(predicted)
    scales = np.array([0.1, 0.5])
    assert session_max_score(predicted, labels, scales) == 4.0


def test_all_authorized_pairs_meet_relative_margin_contract() -> None:
    gamma = 0.1
    certified_bounds = np.array([-0.2, 0.1, 0.7])
    authorized = authorization_mask(certified_bounds, gamma)
    assert np.array_equal(authorized, [False, True, True])
    assert not relative_margin_failure(authorized, [100.0, 0.1, 0.9], gamma)
    assert relative_margin_failure(authorized, [100.0, 0.09, 0.9], gamma)


def test_safe_to_unsafe_is_a_corollary_when_gamma_is_nonnegative() -> None:
    native_margin = 0.0
    challenger_margin = -np.finfo(float).eps
    true_relative_margin = challenger_margin - native_margin
    assert true_relative_margin < 0.0
    assert relative_margin_failure([True], [true_relative_margin], gamma_task=0.0)


def test_negative_gamma_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        authorization_mask([0.0], -1e-9)


def _fraction_oracle(
    values: list[float],
    edges: list[UndirectedEdge],
    challenger: tuple[int, ...],
    native: tuple[int, ...],
) -> Fraction:
    node_count = len(values)
    infinity: Fraction | None = None
    distances: list[list[Fraction | None]] = [
        [infinity for _ in range(node_count)] for _ in range(node_count)
    ]
    for node in range(node_count):
        distances[node][node] = Fraction(0)
    for edge in edges:
        budget = Fraction.from_float(edge.budget)
        distances[edge.source][edge.target] = budget
        distances[edge.target][edge.source] = budget
    for pivot in range(node_count):
        for source in range(node_count):
            left = distances[source][pivot]
            if left is None:
                continue
            for target in range(node_count):
                right = distances[pivot][target]
                if right is None:
                    continue
                candidate = left + right
                current = distances[source][target]
                if current is None or candidate < current:
                    distances[source][target] = candidate

    exact_values = [Fraction.from_float(value) for value in values]
    per_challenger: list[Fraction] = []
    for a in challenger:
        terms: list[Fraction] = []
        for b in native:
            distance = distances[a][b]
            assert distance is not None
            terms.append(exact_values[a] - exact_values[b] - distance)
        per_challenger.append(max(terms))
    return min(per_challenger)


def test_outward_rounded_solver_is_never_optimistic_against_fraction_oracle() -> None:
    rng = np.random.default_rng(20261025)
    for _ in range(500):
        node_count = int(rng.integers(2, 12))
        values = [float(rng.normal() * 10.0 ** rng.integers(-8, 9)) for _ in range(node_count)]
        edges = [
            UndirectedEdge(
                node,
                node + 1,
                float(abs(rng.normal()) * 10.0 ** rng.integers(-8, 9)),
            )
            for node in range(node_count - 1)
        ]
        challenger = tuple(sorted(rng.choice(node_count, size=min(3, node_count), replace=False)))
        native = tuple(sorted(rng.choice(node_count, size=min(3, node_count), replace=False)))
        computed = robust_set_min_lower_bound(values, edges, challenger, native)
        exact = _fraction_oracle(values, edges, challenger, native)
        assert Fraction.from_float(computed) <= exact
