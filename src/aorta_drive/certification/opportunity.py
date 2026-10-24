"""Method-independent continuous teacher opportunities for Gate B."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ContinuousTeacherImprovement:
    """Conservative teacher bounds for challenger minus native margin."""

    challenger_margin_lower_m: float
    native_margin_upper_m: float
    improvement_lower_m: float


def _validated_nodes(nodes: Sequence[int], node_count: int, name: str) -> tuple[int, ...]:
    result = tuple(int(node) for node in nodes)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if any(node < 0 or node >= node_count for node in result):
        raise ValueError(f"{name} contains an out-of-range node")
    return result


def _subtract_down(left: float, right: float) -> float:
    nearest = left - right
    if not math.isfinite(nearest):
        raise ValueError("teacher subtraction overflowed")
    if right == 0.0:
        return nearest
    return math.nextafter(nearest, -math.inf)


def _add_up(left: float, right: float) -> float:
    nearest = left + right
    if not math.isfinite(nearest):
        raise ValueError("teacher addition overflowed")
    if right == 0.0:
        return nearest
    return math.nextafter(nearest, math.inf)


def continuous_teacher_improvement_lower_bound(
    node_labels_m: npt.ArrayLike,
    node_errors_m: npt.ArrayLike,
    challenger_outer_nodes: Sequence[int],
    native_inner_nodes: Sequence[int],
    *,
    challenger_cover_radius_m: float,
) -> ContinuousTeacherImprovement:
    """Lower-bound continuous challenger-minus-native road-margin improvement.

    ``challenger_outer_nodes`` must be an ``h``-net of a continuous outer cover,
    where ``h=challenger_cover_radius_m``.  ``native_inner_nodes`` must be physical
    points inside the native footprint.  Node labels approximate the same
    1-Lipschitz continuous teacher, with pointwise absolute errors bounded by
    ``node_errors_m``.
    """

    labels = np.asarray(node_labels_m, dtype=np.float64).reshape(-1)
    errors = np.asarray(node_errors_m, dtype=np.float64).reshape(-1)
    if labels.size == 0 or errors.shape != labels.shape:
        raise ValueError("node labels and errors must be aligned non-empty vectors")
    if not np.all(np.isfinite(labels)):
        raise ValueError("node labels must be finite")
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("node errors must be finite and non-negative")
    if not math.isfinite(challenger_cover_radius_m) or challenger_cover_radius_m < 0.0:
        raise ValueError("challenger_cover_radius_m must be finite and non-negative")
    challenger = _validated_nodes(
        challenger_outer_nodes, labels.size, "challenger_outer_nodes"
    )
    native = _validated_nodes(native_inner_nodes, labels.size, "native_inner_nodes")

    challenger_node_lower = min(
        _subtract_down(float(labels[node]), float(errors[node])) for node in challenger
    )
    challenger_lower = _subtract_down(
        challenger_node_lower, challenger_cover_radius_m
    )
    native_upper = min(
        _add_up(float(labels[node]), float(errors[node])) for node in native
    )
    improvement = _subtract_down(challenger_lower, native_upper)
    return ContinuousTeacherImprovement(
        challenger_margin_lower_m=challenger_lower,
        native_margin_upper_m=native_upper,
        improvement_lower_m=improvement,
    )
