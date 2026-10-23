"""Exact graph-difference certificates for relative set-min tasks."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class UndirectedEdge:
    """An undirected graph edge with a non-negative difference budget."""

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


def _validated_graph(
    node_values: npt.ArrayLike,
    edges: Iterable[UndirectedEdge],
) -> tuple[npt.NDArray[np.float64], tuple[UndirectedEdge, ...]]:
    values = np.asarray(node_values, dtype=np.float64)
    graph_edges = tuple(edges)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("node_values must be a non-empty finite vector")
    if not graph_edges and values.size > 1:
        raise ValueError("a multi-node graph must contain edges")
    for edge in graph_edges:
        if not 0 <= edge.source < values.size or not 0 <= edge.target < values.size:
            raise ValueError("edge endpoint is out of range")
        if edge.source == edge.target:
            raise ValueError("self edges are not supported")
        if not np.isfinite(edge.budget) or edge.budget < 0:
            raise ValueError("edge budgets must be finite and non-negative")
    return values, graph_edges


def _validated_nodes(nodes: Sequence[int], node_count: int, name: str) -> tuple[int, ...]:
    result = tuple(int(node) for node in nodes)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if any(node < 0 or node >= node_count for node in result):
        raise ValueError(f"{name} contains an out-of-range node")
    return result


def multi_source_envelope(
    node_values: npt.ArrayLike,
    edges: Iterable[UndirectedEdge],
    source_nodes: Sequence[int],
) -> npt.NDArray[np.float64]:
    """Compute a conservative upper envelope by multi-source Dijkstra.

    Every path addition is rounded toward positive infinity. Consequently the
    returned envelope cannot be smaller than the real-arithmetic envelope of
    the represented float inputs.
    """

    values, graph_edges = _validated_graph(node_values, edges)
    sources = _validated_nodes(source_nodes, values.size, "source_nodes")
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(values.size)]
    for edge in graph_edges:
        adjacency[edge.source].append((edge.target, edge.budget))
        adjacency[edge.target].append((edge.source, edge.budget))

    envelope = np.full(values.size, np.inf, dtype=np.float64)
    queue: list[tuple[float, int]] = []
    for node in sources:
        seed = float(values[node])
        if seed < envelope[node]:
            envelope[node] = seed
            heapq.heappush(queue, (seed, node))

    while queue:
        distance, node = heapq.heappop(queue)
        if distance > envelope[node]:
            continue
        for neighbor, budget in adjacency[node]:
            candidate = _add_up(distance, budget)
            if candidate < envelope[neighbor]:
                envelope[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))

    if not np.all(np.isfinite(envelope)):
        raise ValueError("graph must be connected to at least one source node")
    return envelope


def robust_set_min_lower_bound(
    node_values: npt.ArrayLike,
    edges: Iterable[UndirectedEdge],
    challenger_nodes: Sequence[int],
    native_nodes: Sequence[int],
) -> float:
    """Return the exact robust lower bound on ``min_A d - min_B d``.

    The uncertainty set is ``|D(d-u)|_e <= budget_e`` on the supplied
    connected undirected graph.
    """

    values, graph_edges = _validated_graph(node_values, edges)
    challenger = _validated_nodes(challenger_nodes, values.size, "challenger_nodes")
    native = _validated_nodes(native_nodes, values.size, "native_nodes")
    envelope = multi_source_envelope(values, graph_edges, native)
    candidates = [_subtract_down(float(values[node]), float(envelope[node])) for node in challenger]
    if not all(math.isfinite(candidate) for candidate in candidates):
        raise ValueError("conservative arithmetic overflowed")
    return min(candidates)
