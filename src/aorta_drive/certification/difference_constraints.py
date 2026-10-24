"""Exact robust set-min bounds under directed difference constraints."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DirectedConstraint:
    """Represent ``x[target] - x[source] <= budget`` with a non-negative budget."""

    source: int
    target: int
    budget: float


def _add_up(left: float, right: float) -> float:
    result = left + right
    if right == 0.0:
        return result
    return math.nextafter(result, math.inf)


def _subtract_down(left: float, right: float) -> float:
    if left == right:
        return 0.0
    return math.nextafter(left - right, -math.inf)


def _validated_problem(
    node_values: npt.ArrayLike,
    constraints: Iterable[DirectedConstraint],
) -> tuple[npt.NDArray[np.float64], tuple[DirectedConstraint, ...]]:
    values = np.asarray(node_values, dtype=np.float64)
    edges = tuple(constraints)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("node_values must be a non-empty finite vector")
    if not edges and values.size > 1:
        raise ValueError("a multi-node problem must contain constraints")
    for edge in edges:
        if not 0 <= edge.source < values.size or not 0 <= edge.target < values.size:
            raise ValueError("constraint endpoint is out of range")
        if edge.source == edge.target:
            raise ValueError("self constraints are not supported")
        if not math.isfinite(edge.budget) or edge.budget < 0.0:
            raise ValueError("constraint budgets must be finite and non-negative")
    return values, edges


def _validated_nodes(nodes: Sequence[int], node_count: int, name: str) -> tuple[int, ...]:
    result = tuple(int(node) for node in nodes)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if any(node < 0 or node >= node_count for node in result):
        raise ValueError(f"{name} contains an out-of-range node")
    return result


def reverse_multi_source_envelope(
    node_values: npt.ArrayLike,
    constraints: Iterable[DirectedConstraint],
    terminal_nodes: Sequence[int],
) -> npt.NDArray[np.float64]:
    """Return ``min_b value[b] + dist(a, b)`` for every node ``a``.

    A constraint edge ``source -> target`` bounds ``x[target] - x[source]``.
    The required distances run from each query node to a terminal, so Dijkstra
    starts at the terminals and traverses the reversed constraint graph. Every
    path addition is rounded toward positive infinity.
    """

    values, edges = _validated_problem(node_values, constraints)
    terminals = _validated_nodes(terminal_nodes, values.size, "terminal_nodes")
    reverse_adjacency: list[list[tuple[int, float]]] = [
        [] for _ in range(values.size)
    ]
    for edge in edges:
        reverse_adjacency[edge.target].append((edge.source, edge.budget))

    envelope = np.full(values.size, np.inf, dtype=np.float64)
    queue: list[tuple[float, int]] = []
    for node in terminals:
        seed = float(values[node])
        if seed < envelope[node]:
            envelope[node] = seed
            heapq.heappush(queue, (seed, node))

    while queue:
        distance, node = heapq.heappop(queue)
        if distance > envelope[node]:
            continue
        for predecessor, budget in reverse_adjacency[node]:
            candidate = _add_up(distance, budget)
            if candidate < envelope[predecessor]:
                envelope[predecessor] = candidate
                heapq.heappush(queue, (candidate, predecessor))
    return envelope


def directed_single_source_distances(
    node_count: int,
    constraints: Iterable[DirectedConstraint],
    source_node: int,
    *,
    reverse: bool = False,
) -> npt.NDArray[np.float64]:
    """Return outward-rounded shortest-path distances from one source.

    With ``reverse=True``, every registered constraint edge is traversed in the
    opposite direction. This is used to compute distances *to* an anchor node.
    """

    if node_count <= 0:
        raise ValueError("node_count must be positive")
    if not 0 <= source_node < node_count:
        raise ValueError("source_node is out of range")
    _, edges = _validated_problem(np.zeros(node_count, dtype=np.float64), constraints)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
    for edge in edges:
        left, right = (edge.target, edge.source) if reverse else (edge.source, edge.target)
        adjacency[left].append((right, edge.budget))

    distances = np.full(node_count, np.inf, dtype=np.float64)
    distances[source_node] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source_node)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        for neighbor, budget in adjacency[node]:
            candidate = _add_up(distance, budget)
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def robust_set_min_lower_bound_directed(
    node_values: npt.ArrayLike,
    constraints: Iterable[DirectedConstraint],
    challenger_nodes: Sequence[int],
    native_nodes: Sequence[int],
) -> float:
    """Return the exact infimum of ``min_A(h+x) - min_B(h+x)``.

    The shared residual vector ``x`` obeys every registered directed constraint
    ``x[target] - x[source] <= budget``. Budgets must already be non-negative;
    a feasible potential may be used to reweight arbitrary difference constraints
    into this form. Common nodes remain the same variable in both candidates.
    """

    values, edges = _validated_problem(node_values, constraints)
    challenger = _validated_nodes(challenger_nodes, values.size, "challenger_nodes")
    native = _validated_nodes(native_nodes, values.size, "native_nodes")
    envelope = reverse_multi_source_envelope(values, edges, native)
    challenger_array = np.asarray(challenger, dtype=np.int64)
    if not np.all(np.isfinite(envelope[challenger_array])):
        raise ValueError("every challenger node must reach a native terminal")
    candidates = [
        _subtract_down(float(values[node]), float(envelope[node]))
        for node in challenger
    ]
    if not all(math.isfinite(candidate) for candidate in candidates):
        raise ValueError("conservative arithmetic overflowed")
    return min(candidates)
