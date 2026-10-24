"""Independent LP tests for directed robust set-min bounds."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

from aorta_drive.certification.difference_constraints import (
    DirectedConstraint,
    directed_single_source_distances,
    robust_set_min_lower_bound_directed,
)


def _lp_oracle(
    values: np.ndarray,
    constraints: list[DirectedConstraint],
    challenger: tuple[int, ...],
    native: tuple[int, ...],
) -> float:
    node_count = values.size
    optima: list[float] = []
    for challenger_node in challenger:
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for edge in constraints:
            row = np.zeros(node_count + 1)
            row[edge.target] = 1.0
            row[edge.source] -= 1.0
            rows.append(row)
            rhs.append(edge.budget)
        for native_node in native:
            epigraph = np.zeros(node_count + 1)
            epigraph[challenger_node] = 1.0
            epigraph[native_node] -= 1.0
            epigraph[-1] = -1.0
            rows.append(epigraph)
            rhs.append(float(-values[challenger_node] + values[native_node]))

        objective = np.zeros(node_count + 1)
        objective[-1] = 1.0
        bounds = [(0.0, 0.0), *[(None, None)] * (node_count - 1), (None, None)]
        result = linprog(
            objective,
            A_ub=np.asarray(rows),
            b_ub=np.asarray(rhs),
            bounds=bounds,
            method="highs",
        )
        assert result.success, result.message
        optima.append(float(result.fun))
    return min(optima)


def _strongly_connected_constraints(
    node_count: int,
    rng: np.random.Generator,
) -> list[DirectedConstraint]:
    constraints: list[DirectedConstraint] = []
    for node in range(node_count - 1):
        constraints.append(
            DirectedConstraint(node, node + 1, float(rng.uniform(0.0, 2.0)))
        )
        constraints.append(
            DirectedConstraint(node + 1, node, float(rng.uniform(0.0, 2.0)))
        )
    if node_count > 2:
        constraints.append(
            DirectedConstraint(node_count - 1, 0, float(rng.uniform(0.0, 2.0)))
        )
        constraints.append(
            DirectedConstraint(0, node_count - 1, float(rng.uniform(0.0, 2.0)))
        )
    return constraints


def test_directed_closed_form_matches_independent_lp() -> None:
    rng = np.random.default_rng(20261025)
    for node_count in range(2, 8):
        for _ in range(40):
            values = rng.normal(size=node_count)
            constraints = _strongly_connected_constraints(node_count, rng)
            challenger = tuple(
                sorted(
                    rng.choice(
                        node_count,
                        size=rng.integers(1, node_count + 1),
                        replace=False,
                    )
                )
            )
            native = tuple(
                sorted(
                    rng.choice(
                        node_count,
                        size=rng.integers(1, node_count + 1),
                        replace=False,
                    )
                )
            )
            exact = robust_set_min_lower_bound_directed(
                values,
                constraints,
                challenger,
                native,
            )
            oracle = _lp_oracle(values, constraints, challenger, native)
            assert np.isclose(exact, oracle, atol=1e-9, rtol=1e-9)


def test_directed_solver_preserves_common_shift_and_set_inclusion() -> None:
    values = np.array([0.2, -1.0, 3.0, 0.5])
    constraints = [
        DirectedConstraint(0, 1, 0.1),
        DirectedConstraint(1, 0, 0.7),
        DirectedConstraint(1, 2, 1.2),
        DirectedConstraint(2, 1, 0.3),
        DirectedConstraint(2, 3, 0.6),
        DirectedConstraint(3, 2, 0.9),
        DirectedConstraint(3, 0, 1.1),
        DirectedConstraint(0, 3, 0.4),
    ]
    base = robust_set_min_lower_bound_directed(values, constraints, (0, 1), (0, 1, 2))
    shifted = robust_set_min_lower_bound_directed(
        values + 1234.5,
        constraints,
        (0, 1),
        (0, 1, 2),
    )
    equal = robust_set_min_lower_bound_directed(values, constraints, (0, 1), (0, 1))
    assert np.isclose(base, shifted)
    assert base >= -1e-12
    assert np.isclose(equal, 0.0)


def test_asymmetric_direction_is_not_silently_symmetrized() -> None:
    values = np.zeros(3)
    constraints = [
        DirectedConstraint(0, 1, 0.1),
        DirectedConstraint(1, 0, 3.0),
        DirectedConstraint(1, 2, 0.2),
        DirectedConstraint(2, 1, 4.0),
    ]
    forward = robust_set_min_lower_bound_directed(values, constraints, (0,), (2,))
    reverse = robust_set_min_lower_bound_directed(values, constraints, (2,), (0,))
    assert np.isclose(forward, -0.3)
    assert np.isclose(reverse, -7.0)


def test_unreachable_terminal_is_rejected() -> None:
    values = np.zeros(3)
    constraints = [DirectedConstraint(0, 1, 1.0)]
    with np.testing.assert_raises_regex(ValueError, "reach a native terminal"):
        robust_set_min_lower_bound_directed(values, constraints, (2,), (0,))


def test_all_nonempty_candidate_subsets_on_small_problem() -> None:
    values = np.array([0.0, 0.4, -0.7])
    constraints = [
        DirectedConstraint(0, 1, 0.3),
        DirectedConstraint(1, 0, 0.8),
        DirectedConstraint(1, 2, 1.1),
        DirectedConstraint(2, 1, 0.2),
        DirectedConstraint(0, 2, 0.9),
        DirectedConstraint(2, 0, 0.6),
    ]
    subsets = [
        tuple(index for index in range(3) if mask & (1 << index))
        for mask in range(1, 1 << 3)
    ]
    for challenger, native in itertools.product(subsets, repeat=2):
        exact = robust_set_min_lower_bound_directed(
            values,
            constraints,
            challenger,
            native,
        )
        oracle = _lp_oracle(values, constraints, challenger, native)
        assert np.isclose(exact, oracle, atol=1e-9, rtol=1e-9)


def test_forward_and_reverse_single_source_distances() -> None:
    constraints = [
        DirectedConstraint(0, 1, 0.5),
        DirectedConstraint(1, 0, 2.0),
        DirectedConstraint(1, 2, 0.25),
        DirectedConstraint(2, 1, 4.0),
        DirectedConstraint(0, 2, 3.0),
    ]
    forward = directed_single_source_distances(3, constraints, 0)
    to_zero = directed_single_source_distances(3, constraints, 0, reverse=True)
    assert np.allclose(forward, [0.0, 0.5, 0.75])
    assert np.allclose(to_zero, [0.0, 2.0, 6.0])
