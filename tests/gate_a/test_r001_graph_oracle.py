"""R001: independent LP checks for the exact graph certificate."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

from aorta_drive.certification.graph import (
    UndirectedEdge,
    multi_source_envelope,
    robust_set_min_lower_bound,
)


def _lp_oracle(
    values: np.ndarray,
    edges: list[UndirectedEdge],
    challenger: tuple[int, ...],
    native: tuple[int, ...],
) -> float:
    """Solve the robust set-min problem through a separate LP formulation."""

    node_count = values.size
    optima: list[float] = []
    for challenger_node in challenger:
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for edge in edges:
            forward = np.zeros(node_count + 1)
            forward[edge.source] = 1.0
            forward[edge.target] = -1.0
            rows.append(forward)
            rhs.append(edge.budget)
            rows.append(-forward)
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


def _chain(node_count: int, rng: np.random.Generator) -> list[UndirectedEdge]:
    return [
        UndirectedEdge(node, node + 1, float(rng.uniform(0.0, 2.0)))
        for node in range(node_count - 1)
    ]


def _cycle(node_count: int, rng: np.random.Generator) -> list[UndirectedEdge]:
    return [*_chain(node_count, rng), UndirectedEdge(node_count - 1, 0, float(rng.uniform(0, 2)))]


def _grid(side: int, rng: np.random.Generator) -> list[UndirectedEdge]:
    edges: list[UndirectedEdge] = []
    for row in range(side):
        for column in range(side):
            node = row * side + column
            if column + 1 < side:
                edges.append(UndirectedEdge(node, node + 1, float(rng.uniform(0.0, 2.0))))
            if row + 1 < side:
                edges.append(UndirectedEdge(node, node + side, float(rng.uniform(0.0, 2.0))))
    return edges


def test_closed_form_matches_independent_lp_on_random_chains_and_cycles() -> None:
    rng = np.random.default_rng(20261023)
    for node_count, graph_factory in itertools.product(range(2, 8), (_chain, _cycle)):
        for _ in range(20):
            values = rng.normal(size=node_count)
            edges = graph_factory(node_count, rng)
            challenger = tuple(
                sorted(rng.choice(node_count, size=rng.integers(1, node_count + 1), replace=False))
            )
            native = tuple(
                sorted(rng.choice(node_count, size=rng.integers(1, node_count + 1), replace=False))
            )
            closed_form = robust_set_min_lower_bound(values, edges, challenger, native)
            oracle = _lp_oracle(values, edges, challenger, native)
            assert np.isclose(closed_form, oracle, atol=1e-9, rtol=1e-9)


def test_closed_form_matches_independent_lp_on_random_grids() -> None:
    rng = np.random.default_rng(20261024)
    for side in (2, 3):
        node_count = side * side
        for _ in range(20):
            values = rng.normal(size=node_count)
            edges = _grid(side, rng)
            challenger = tuple(sorted(rng.choice(node_count, size=3, replace=False)))
            native = tuple(sorted(rng.choice(node_count, size=3, replace=False)))
            closed_form = robust_set_min_lower_bound(values, edges, challenger, native)
            oracle = _lp_oracle(values, edges, challenger, native)
            assert np.isclose(closed_form, oracle, atol=1e-9, rtol=1e-9)


def test_nonzero_multi_source_labels_are_preserved() -> None:
    values = np.array([8.0, 50.0, -3.0, 20.0])
    edges = [
        UndirectedEdge(0, 1, 1.0),
        UndirectedEdge(1, 2, 1.0),
        UndirectedEdge(2, 3, 1.0),
    ]
    envelope = multi_source_envelope(values, edges, (0, 2))
    assert np.allclose(envelope, [-1.0, -2.0, -3.0, -2.0])


def test_common_shift_zero_edges_and_inclusion_contracts() -> None:
    values = np.array([0.2, -1.0, 3.0, 0.5])
    edges = [
        UndirectedEdge(0, 1, 0.0),
        UndirectedEdge(1, 2, 0.7),
        UndirectedEdge(2, 3, 1.3),
        UndirectedEdge(3, 0, 0.4),
    ]
    base = robust_set_min_lower_bound(values, edges, (0, 1), (0, 1, 2))
    shifted = robust_set_min_lower_bound(values + 1234.5, edges, (0, 1), (0, 1, 2))
    equal = robust_set_min_lower_bound(values, edges, (0, 1), (0, 1))
    assert np.isclose(base, shifted)
    assert base >= -1e-12
    assert np.isclose(equal, 0.0)


def test_disconnected_graph_is_rejected() -> None:
    values = np.zeros(3)
    edges = [UndirectedEdge(0, 1, 1.0)]
    try:
        robust_set_min_lower_bound(values, edges, (2,), (0,))
    except ValueError as error:
        assert "connected" in str(error)
    else:
        raise AssertionError("disconnected graph must not receive a certificate")
