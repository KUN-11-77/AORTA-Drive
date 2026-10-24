"""Oracle support sets induced by signed-distance Lipschitz structure."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aorta_drive.certification.difference_constraints import (
    DirectedConstraint,
    directed_single_source_distances,
)
from aorta_drive.certification.graph import UndirectedEdge, multi_source_envelope


@dataclass(frozen=True)
class OracleLipschitzProblem:
    """Joint and marginal-box descriptions of the same support evidence."""

    node_values_with_reference: npt.NDArray[np.float64]
    joint_constraints: tuple[DirectedConstraint, ...]
    box_constraints: tuple[DirectedConstraint, ...]
    marginal_lower: npt.NDArray[np.float64]
    marginal_upper: npt.NDArray[np.float64]
    reference_node: int
    observed_node_count: int
    maximum_numerical_widening: float
    teacher_included: bool


def _add_up(left: float, right: float) -> float:
    result = left + right
    if right == 0.0:
        return result
    return math.nextafter(result, math.inf)


def _subtract_up(left: float, right: float) -> float:
    if left == right:
        return 0.0
    return math.nextafter(left - right, math.inf)


def _subtract_down(left: float, right: float) -> float:
    if left == right:
        return 0.0
    return math.nextafter(left - right, -math.inf)


def _effective_slack(bound: float, delta: float) -> tuple[float, float]:
    """Return a non-negative reduced cost and any required numerical widening."""

    raw = _subtract_up(bound, delta)
    if raw >= 0.0:
        return raw, 0.0
    widening = _subtract_up(delta, bound)
    return 0.0, max(widening, 0.0)


def _grid_edges(
    errors: npt.NDArray[np.float64],
    rows: int,
    columns: int,
    resolution: float,
) -> tuple[tuple[UndirectedEdge, ...], dict[tuple[int, int], float]]:
    undirected: list[UndirectedEdge] = []
    bounds: dict[tuple[int, int], float] = {}
    for row in range(rows):
        for column in range(columns):
            source = row * columns + column
            neighbors: list[int] = []
            if column + 1 < columns:
                neighbors.append(source + 1)
            if row + 1 < rows:
                neighbors.append(source + columns)
            for target in neighbors:
                budget = _add_up(_add_up(resolution, float(errors[source])), float(errors[target]))
                undirected.append(UndirectedEdge(source, target, budget))
                bounds[(source, target)] = budget
                bounds[(target, source)] = budget
    return tuple(undirected), bounds


def build_oracle_lipschitz_problem(
    labels: npt.ArrayLike,
    total_errors: npt.ArrayLike,
    observed_mask: npt.ArrayLike,
    *,
    rows: int,
    columns: int,
    resolution: float,
) -> OracleLipschitzProblem:
    """Build a conservative support oracle without reading hidden node labels.

    Projectively supported nodes reveal their registered teacher intervals. The
    remaining nodes are constrained only by the one-Lipschitz signed-distance
    property plus both endpoint error ledgers. Hidden labels are used solely for
    the final containment audit returned by ``teacher_included``.
    """

    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    errors = np.asarray(total_errors, dtype=np.float64).reshape(-1)
    observed = np.asarray(observed_mask, dtype=bool).reshape(-1)
    node_count = rows * columns
    if values.size != node_count or errors.size != node_count or observed.size != node_count:
        raise ValueError("labels, errors, and support mask must match the registered grid")
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be finite and positive")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(errors)):
        raise ValueError("labels and errors must be finite")
    if np.any(errors < 0.0):
        raise ValueError("total errors must be non-negative")
    observed_nodes = tuple(int(node) for node in np.flatnonzero(observed))
    if not observed_nodes:
        raise ValueError("at least one observed node is required")

    lower = np.nextafter(values - errors, -np.inf)
    upper = np.nextafter(values + errors, np.inf)
    grid_edges, original_grid_bounds = _grid_edges(errors, rows, columns, resolution)
    feasible_values = multi_source_envelope(upper, grid_edges, observed_nodes)
    reference = node_count
    node_values = np.concatenate((feasible_values, np.array([0.0], dtype=np.float64)))

    constraints: list[DirectedConstraint] = []
    maximum_widening = 0.0
    for (source, target), bound in original_grid_bounds.items():
        delta = _subtract_down(float(feasible_values[target]), float(feasible_values[source]))
        slack, widening = _effective_slack(bound, delta)
        constraints.append(DirectedConstraint(source, target, slack))
        maximum_widening = max(maximum_widening, widening)

    for node in observed_nodes:
        upper_delta = float(feasible_values[node])
        upper_slack, upper_widening = _effective_slack(float(upper[node]), upper_delta)
        constraints.append(DirectedConstraint(reference, node, upper_slack))
        lower_bound = -float(lower[node])
        lower_delta = -float(feasible_values[node])
        lower_slack, lower_widening = _effective_slack(lower_bound, lower_delta)
        constraints.append(DirectedConstraint(node, reference, lower_slack))
        maximum_widening = max(maximum_widening, upper_widening, lower_widening)

    constraint_tuple = tuple(constraints)
    from_reference = directed_single_source_distances(
        node_count + 1,
        constraint_tuple,
        reference,
    )
    to_reference = directed_single_source_distances(
        node_count + 1,
        constraint_tuple,
        reference,
        reverse=True,
    )
    if not np.all(np.isfinite(from_reference)) or not np.all(np.isfinite(to_reference)):
        raise ValueError("support constraints must connect every node to the reference")

    marginal_lower = np.empty(node_count, dtype=np.float64)
    marginal_upper = np.empty(node_count, dtype=np.float64)
    box_constraints: list[DirectedConstraint] = []
    for node in range(node_count):
        marginal_upper[node] = _add_up(float(feasible_values[node]), float(from_reference[node]))
        marginal_lower[node] = _subtract_down(
            float(feasible_values[node]),
            float(to_reference[node]),
        )
        box_constraints.append(
            DirectedConstraint(reference, node, float(from_reference[node]))
        )
        box_constraints.append(DirectedConstraint(node, reference, float(to_reference[node])))

    teacher_edges_included = True
    for edge in grid_edges:
        for source, target in ((edge.source, edge.target), (edge.target, edge.source)):
            if values[target] - values[source] > original_grid_bounds[(source, target)]:
                teacher_edges_included = False
                break
        if not teacher_edges_included:
            break
    teacher_included = bool(
        np.all(values >= marginal_lower)
        and np.all(values <= marginal_upper)
        and teacher_edges_included
    )
    return OracleLipschitzProblem(
        node_values_with_reference=node_values,
        joint_constraints=constraint_tuple,
        box_constraints=tuple(box_constraints),
        marginal_lower=marginal_lower,
        marginal_upper=marginal_upper,
        reference_node=reference,
        observed_node_count=len(observed_nodes),
        maximum_numerical_widening=maximum_widening,
        teacher_included=teacher_included,
    )


def independent_candidate_contrast_interval(
    marginal_lower: npt.ArrayLike,
    marginal_upper: npt.ArrayLike,
    challenger_nodes: tuple[int, ...],
    native_nodes: tuple[int, ...],
) -> tuple[float, float]:
    """Return the interval after duplicating all variables across candidates."""

    lower = np.asarray(marginal_lower, dtype=np.float64)
    upper = np.asarray(marginal_upper, dtype=np.float64)
    if lower.ndim != 1 or upper.shape != lower.shape or lower.size == 0:
        raise ValueError("marginal bounds must be aligned non-empty vectors")
    if np.any(lower > upper) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("marginal bounds must be finite and ordered")
    challenger = np.asarray(challenger_nodes, dtype=np.int64)
    native = np.asarray(native_nodes, dtype=np.int64)
    if challenger.size == 0 or native.size == 0:
        raise ValueError("candidate node sets must be non-empty")
    if np.any(challenger < 0) or np.any(native < 0):
        raise ValueError("candidate node indices must be non-negative")
    if np.any(challenger >= lower.size) or np.any(native >= lower.size):
        raise ValueError("candidate node index is out of range")
    interval_lower = _subtract_down(
        float(np.min(lower[challenger])),
        float(np.min(upper[native])),
    )
    interval_upper = _subtract_down(
        float(np.min(upper[challenger])),
        float(np.min(lower[native])),
    )
    interval_upper = math.nextafter(interval_upper, math.inf)
    return interval_lower, interval_upper
