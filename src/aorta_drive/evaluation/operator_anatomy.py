"""Exact anatomy of scalar set-min contrasts on frozen candidate supports."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SetMinPairAnatomy:
    """Mechanically exclusive explanation for one candidate contrast."""

    category: str
    reference_minimum: float
    comparison_minimum: float
    contrast: float
    reference_node_count: int
    comparison_node_count: int
    intersection_node_count: int
    reference_exclusive_node_count: int
    comparison_exclusive_node_count: int
    symmetric_difference_node_count: int
    support_sets_equal: bool
    common_minimum: float | None
    common_minimum_controls_both: bool


@dataclass(frozen=True)
class TrajectoryPairAnatomy:
    """Continuous-pose differences that may be hidden by node rasterization."""

    poses_exactly_equal: bool
    maximum_center_separation_m: float
    maximum_wrapped_yaw_separation_rad: float
    first_center_separation_index_gt_1e_9_m: int | None
    first_center_separation_index_gt_half_grid_m: int | None
    first_center_separation_index_gt_one_grid_m: int | None


@dataclass(frozen=True)
class ForkSetMinEvaluation:
    """Shared qualification and branch contrast for one factorization."""

    valid: bool
    reason: str
    fork_index: int | None
    shared_node_count: int
    reference_branch_node_count: int
    comparison_branch_node_count: int
    shared_minimum: float | None
    reference_branch_minimum: float | None
    comparison_branch_minimum: float | None
    branch_contrast: float | None
    shared_strictly_safe: bool
    reference_branch_strictly_safe: bool
    comparison_branch_strictly_safe: bool
    nonzero_branch_contrast: bool
    road_rescue_opportunity: bool
    road_induction_risk: bool


def _unique_nodes(values: npt.ArrayLike, node_count: int, name: str) -> tuple[int, ...]:
    nodes = tuple(int(value) for value in np.asarray(values, dtype=np.int64).reshape(-1))
    if not nodes:
        raise ValueError(f"{name} must be nonempty")
    if len(nodes) != len(set(nodes)):
        raise ValueError(f"{name} must contain unique nodes")
    if min(nodes) < 0 or max(nodes) >= node_count:
        raise ValueError(f"{name} contains an out-of-range node")
    return nodes


def analyze_set_min_pair(
    reference_nodes: npt.ArrayLike,
    comparison_nodes: npt.ArrayLike,
    labels: npt.ArrayLike,
) -> SetMinPairAnatomy:
    """Classify an exact scalar set-min contrast without an epsilon tie rule.

    Exact zero contrasts are assigned in this frozen priority order:
    identical discrete supports, a shared minimum controlling both candidates,
    then equal minima attained outside that shared-min condition.
    """

    field = np.asarray(labels, dtype=np.float64).reshape(-1)
    if field.size == 0 or not np.all(np.isfinite(field)):
        raise ValueError("labels must be a nonempty finite vector")
    reference = _unique_nodes(reference_nodes, field.size, "reference_nodes")
    comparison = _unique_nodes(comparison_nodes, field.size, "comparison_nodes")
    reference_set = set(reference)
    comparison_set = set(comparison)
    intersection = reference_set & comparison_set
    reference_exclusive = reference_set - intersection
    comparison_exclusive = comparison_set - intersection
    reference_minimum = float(np.min(field[np.asarray(reference, dtype=np.int64)]))
    comparison_minimum = float(np.min(field[np.asarray(comparison, dtype=np.int64)]))
    contrast = comparison_minimum - reference_minimum
    common_minimum = (
        float(np.min(field[np.asarray(sorted(intersection), dtype=np.int64)]))
        if intersection
        else None
    )
    common_controls = bool(
        common_minimum is not None
        and common_minimum == reference_minimum
        and common_minimum == comparison_minimum
    )
    supports_equal = reference_set == comparison_set
    if contrast != 0.0:
        category = "nonzero_contrast"
    elif supports_equal:
        category = "discrete_support_duplicate"
    elif common_controls:
        category = "common_minimum_domination"
    else:
        category = "distinct_minimum_teacher_tie"
    return SetMinPairAnatomy(
        category=category,
        reference_minimum=reference_minimum,
        comparison_minimum=comparison_minimum,
        contrast=contrast,
        reference_node_count=len(reference_set),
        comparison_node_count=len(comparison_set),
        intersection_node_count=len(intersection),
        reference_exclusive_node_count=len(reference_exclusive),
        comparison_exclusive_node_count=len(comparison_exclusive),
        symmetric_difference_node_count=len(reference_set ^ comparison_set),
        support_sets_equal=supports_equal,
        common_minimum=common_minimum,
        common_minimum_controls_both=common_controls,
    )


def _first_index_greater(values: npt.NDArray[np.float64], threshold: float) -> int | None:
    hits = np.flatnonzero(values > threshold)
    return int(hits[0]) if hits.size else None


def analyze_trajectory_pair(
    reference_poses: npt.ArrayLike,
    comparison_poses: npt.ArrayLike,
    *,
    grid_resolution_m: float,
) -> TrajectoryPairAnatomy:
    """Report continuous-pose separation at fixed, geometry-derived thresholds."""

    reference = np.asarray(reference_poses, dtype=np.float64)
    comparison = np.asarray(comparison_poses, dtype=np.float64)
    if (
        reference.ndim != 2
        or reference.shape[1] != 3
        or reference.shape != comparison.shape
        or reference.shape[0] == 0
    ):
        raise ValueError("candidate poses must have aligned nonempty shape [time,3]")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(comparison)):
        raise ValueError("candidate poses must be finite")
    if not math.isfinite(grid_resolution_m) or grid_resolution_m <= 0.0:
        raise ValueError("grid_resolution_m must be finite and positive")
    center_separation = np.linalg.norm(comparison[:, :2] - reference[:, :2], axis=1)
    yaw_delta = comparison[:, 2] - reference[:, 2]
    wrapped_yaw = np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)))
    return TrajectoryPairAnatomy(
        poses_exactly_equal=bool(np.array_equal(reference, comparison)),
        maximum_center_separation_m=float(np.max(center_separation)),
        maximum_wrapped_yaw_separation_rad=float(np.max(wrapped_yaw)),
        first_center_separation_index_gt_1e_9_m=_first_index_greater(
            center_separation, 1.0e-9
        ),
        first_center_separation_index_gt_half_grid_m=_first_index_greater(
            center_separation, 0.5 * grid_resolution_m
        ),
        first_center_separation_index_gt_one_grid_m=_first_index_greater(
            center_separation, grid_resolution_m
        ),
    )


def _node_set(values: npt.ArrayLike, node_count: int, name: str) -> set[int]:
    return set(_unique_nodes(values, node_count, name))


def _fork_evaluation(
    shared: set[int],
    reference_branch: set[int],
    comparison_branch: set[int],
    labels: npt.NDArray[np.float64],
    *,
    fork_index: int | None,
    invalid_reason: str | None = None,
) -> ForkSetMinEvaluation:
    if invalid_reason is not None or not shared or not reference_branch or not comparison_branch:
        reason = invalid_reason or "empty_shared_or_branch_support"
        return ForkSetMinEvaluation(
            valid=False,
            reason=reason,
            fork_index=fork_index,
            shared_node_count=len(shared),
            reference_branch_node_count=len(reference_branch),
            comparison_branch_node_count=len(comparison_branch),
            shared_minimum=None,
            reference_branch_minimum=None,
            comparison_branch_minimum=None,
            branch_contrast=None,
            shared_strictly_safe=False,
            reference_branch_strictly_safe=False,
            comparison_branch_strictly_safe=False,
            nonzero_branch_contrast=False,
            road_rescue_opportunity=False,
            road_induction_risk=False,
        )
    shared_minimum = float(np.min(labels[np.asarray(sorted(shared), dtype=np.int64)]))
    reference_minimum = float(
        np.min(labels[np.asarray(sorted(reference_branch), dtype=np.int64)])
    )
    comparison_minimum = float(
        np.min(labels[np.asarray(sorted(comparison_branch), dtype=np.int64)])
    )
    contrast = comparison_minimum - reference_minimum
    shared_safe = shared_minimum > 0.0
    reference_safe = reference_minimum > 0.0
    comparison_safe = comparison_minimum > 0.0
    return ForkSetMinEvaluation(
        valid=True,
        reason="valid",
        fork_index=fork_index,
        shared_node_count=len(shared),
        reference_branch_node_count=len(reference_branch),
        comparison_branch_node_count=len(comparison_branch),
        shared_minimum=shared_minimum,
        reference_branch_minimum=reference_minimum,
        comparison_branch_minimum=comparison_minimum,
        branch_contrast=contrast,
        shared_strictly_safe=shared_safe,
        reference_branch_strictly_safe=reference_safe,
        comparison_branch_strictly_safe=comparison_safe,
        nonzero_branch_contrast=contrast != 0.0,
        road_rescue_opportunity=shared_safe and not reference_safe and comparison_safe,
        road_induction_risk=shared_safe and reference_safe and not comparison_safe,
    )


def evaluate_spatial_fork(
    reference_nodes: npt.ArrayLike,
    comparison_nodes: npt.ArrayLike,
    labels: npt.ArrayLike,
) -> ForkSetMinEvaluation:
    """Factor complete supports into exact spatial overlap and exclusive branches."""

    field = np.asarray(labels, dtype=np.float64).reshape(-1)
    if field.size == 0 or not np.all(np.isfinite(field)):
        raise ValueError("labels must be a nonempty finite vector")
    reference = _node_set(reference_nodes, field.size, "reference_nodes")
    comparison = _node_set(comparison_nodes, field.size, "comparison_nodes")
    shared = reference & comparison
    return _fork_evaluation(
        shared,
        reference - shared,
        comparison - shared,
        field,
        fork_index=None,
    )


def evaluate_temporal_fork(
    reference_nodes_by_time: tuple[npt.ArrayLike, ...],
    comparison_nodes_by_time: tuple[npt.ArrayLike, ...],
    labels: npt.ArrayLike,
) -> ForkSetMinEvaluation:
    """Use the first exact per-time footprint-set difference as the fork index."""

    field = np.asarray(labels, dtype=np.float64).reshape(-1)
    if field.size == 0 or not np.all(np.isfinite(field)):
        raise ValueError("labels must be a nonempty finite vector")
    if not reference_nodes_by_time or len(reference_nodes_by_time) != len(
        comparison_nodes_by_time
    ):
        raise ValueError("temporal node sequences must be aligned and nonempty")
    reference = tuple(
        _node_set(nodes, field.size, f"reference_nodes_by_time[{index}]")
        for index, nodes in enumerate(reference_nodes_by_time)
    )
    comparison = tuple(
        _node_set(nodes, field.size, f"comparison_nodes_by_time[{index}]")
        for index, nodes in enumerate(comparison_nodes_by_time)
    )
    fork_index = next(
        (index for index, (left, right) in enumerate(zip(reference, comparison, strict=True)) if left != right),
        None,
    )
    if fork_index is None:
        return _fork_evaluation(
            set(),
            set(),
            set(),
            field,
            fork_index=None,
            invalid_reason="no_discrete_temporal_fork",
        )
    shared = set().union(*reference[:fork_index]) if fork_index > 0 else set()
    reference_branch = set().union(*reference[fork_index:])
    comparison_branch = set().union(*comparison[fork_index:])
    return _fork_evaluation(
        shared,
        reference_branch,
        comparison_branch,
        field,
        fork_index=fork_index,
    )
