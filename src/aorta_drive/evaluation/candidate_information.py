"""Outcome-blind candidate-information and support diagnostics.

This module operates only on frozen realized trajectories and vehicle geometry.
It deliberately does not read visual fields, teachers, scores, or decision
outcomes.  All footprint sets are discrete-time lattice queries, not continuous
swept-footprint certificates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.shadow_decision import build_shadow_panel

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PairInformation:
    """Prefix versus complete-horizon information for one candidate pair."""

    challenger_index: int
    prefix_nodes_equal: bool
    full_nodes_equal: bool
    prefix_equal_full_distinct: bool
    prefix_jaccard: float
    full_jaccard: float
    prefix_symmetric_difference_nodes: int
    full_symmetric_difference_nodes: int
    full_distinguishing_nodes_absent_from_prefix: int
    full_distinguishing_fraction_absent_from_prefix: float
    maximum_center_separation_m: float
    maximum_prefix_center_separation_m: float
    first_divergence_index_by_threshold: Mapping[str, int | None]
    late_divergence_by_threshold: Mapping[str, bool]


def _poses(values: npt.ArrayLike) -> FloatArray:
    poses = np.asarray(values, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] == 0:
        raise ValueError("trajectory must have shape [time,3]")
    if not np.all(np.isfinite(poses)):
        raise ValueError("trajectory must be finite")
    return poses


def _corners(pose: FloatArray, body: VehicleRectangle) -> FloatArray:
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    forward = np.asarray((cosine, sine), dtype=np.float64)
    left = np.asarray((-sine, cosine), dtype=np.float64)
    return np.asarray(
        [
            pose[:2] + longitudinal * forward + lateral * left
            for longitudinal, lateral in (
                (body.front_extent, body.half_width),
                (-body.rear_overhang, body.half_width),
                (-body.rear_overhang, -body.half_width),
                (body.front_extent, -body.half_width),
            )
        ],
        dtype=np.float64,
    )


def rasterize_unbounded_footprint_lattice(
    realized_poses: npt.ArrayLike,
    *,
    body: VehicleRectangle,
    resolution_m: float = 1.0,
    center_origin_xy_m: tuple[float, float] = (0.5, 0.5),
) -> IntArray:
    """Rasterize oriented rectangles on an unbounded, fixed cell-center lattice.

    Returned integer pairs ``(ix, iy)`` identify centers
    ``origin + resolution * (ix, iy)``.  This removes ROI clipping while
    preserving the original one-metre lattice phase.
    """

    poses = _poses(realized_poses)
    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("resolution must be finite and positive")
    origin = np.asarray(center_origin_xy_m, dtype=np.float64)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("lattice origin must be two finite coordinates")
    selected: set[tuple[int, int]] = set()
    longitudinal_low = np.nextafter(-body.rear_overhang, -np.inf)
    longitudinal_high = np.nextafter(body.front_extent, np.inf)
    lateral_low = np.nextafter(-body.half_width, -np.inf)
    lateral_high = np.nextafter(body.half_width, np.inf)
    for pose in poses:
        corners = _corners(pose, body)
        scaled_low = (np.min(corners, axis=0) - origin) / resolution_m
        scaled_high = (np.max(corners, axis=0) - origin) / resolution_m
        x_low = math.ceil(float(np.nextafter(scaled_low[0], -np.inf)))
        x_high = math.floor(float(np.nextafter(scaled_high[0], np.inf)))
        y_low = math.ceil(float(np.nextafter(scaled_low[1], -np.inf)))
        y_high = math.floor(float(np.nextafter(scaled_high[1], np.inf)))
        if x_low > x_high or y_low > y_high:
            continue
        xs = np.arange(x_low, x_high + 1, dtype=np.int64)
        ys = np.arange(y_low, y_high + 1, dtype=np.int64)
        xx, yy = np.meshgrid(xs, ys)
        indices = np.column_stack((xx.ravel(), yy.ravel()))
        centers = origin + resolution_m * indices
        relative = centers - pose[:2]
        cosine = math.cos(float(pose[2]))
        sine = math.sin(float(pose[2]))
        longitudinal = relative[:, 0] * cosine + relative[:, 1] * sine
        lateral = -relative[:, 0] * sine + relative[:, 1] * cosine
        inside = (
            (longitudinal >= longitudinal_low)
            & (longitudinal <= longitudinal_high)
            & (lateral >= lateral_low)
            & (lateral <= lateral_high)
        )
        selected.update((int(x), int(y)) for x, y in indices[inside])
    if not selected:
        raise ValueError("discrete footprint contains no lattice node")
    return np.asarray(sorted(selected), dtype=np.int64)


def _node_set(values: IntArray) -> set[tuple[int, int]]:
    return {(int(row[0]), int(row[1])) for row in values}


def _jaccard(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    union = left | right
    if not union:
        raise ValueError("node-set union must be nonempty")
    return len(left & right) / len(union)


def evaluate_native_challenger_information(
    native_poses: npt.ArrayLike,
    challenger_poses: npt.ArrayLike,
    *,
    challenger_index: int,
    prefix_index: int,
    divergence_thresholds_m: Sequence[float],
    body: VehicleRectangle,
    lattice_resolution_m: float = 1.0,
    lattice_origin_xy_m: tuple[float, float] = (0.5, 0.5),
) -> PairInformation:
    """Measure exactly what a fixed prefix removes from one frozen pair."""

    native = _poses(native_poses)
    challenger = _poses(challenger_poses)
    if native.shape != challenger.shape:
        raise ValueError("candidate pair trajectories must have identical shape")
    if prefix_index < 0 or prefix_index >= native.shape[0] - 1:
        raise ValueError("prefix must be a proper prefix of the complete horizon")
    thresholds = tuple(float(value) for value in divergence_thresholds_m)
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("divergence thresholds must be unique and nonempty")
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("divergence thresholds must be finite and positive")

    native_prefix = _node_set(
        rasterize_unbounded_footprint_lattice(
            native[: prefix_index + 1],
            body=body,
            resolution_m=lattice_resolution_m,
            center_origin_xy_m=lattice_origin_xy_m,
        )
    )
    challenger_prefix = _node_set(
        rasterize_unbounded_footprint_lattice(
            challenger[: prefix_index + 1],
            body=body,
            resolution_m=lattice_resolution_m,
            center_origin_xy_m=lattice_origin_xy_m,
        )
    )
    native_full = _node_set(
        rasterize_unbounded_footprint_lattice(
            native,
            body=body,
            resolution_m=lattice_resolution_m,
            center_origin_xy_m=lattice_origin_xy_m,
        )
    )
    challenger_full = _node_set(
        rasterize_unbounded_footprint_lattice(
            challenger,
            body=body,
            resolution_m=lattice_resolution_m,
            center_origin_xy_m=lattice_origin_xy_m,
        )
    )
    separation = np.linalg.norm(challenger[:, :2] - native[:, :2], axis=1)
    first: dict[str, int | None] = {}
    late: dict[str, bool] = {}
    for threshold in thresholds:
        hits = np.flatnonzero(separation > threshold)
        first_index = int(hits[0]) if hits.size else None
        key = format(threshold, ".12g")
        first[key] = first_index
        late[key] = bool(first_index is not None and first_index > prefix_index)
    prefix_equal = native_prefix == challenger_prefix
    full_equal = native_full == challenger_full
    prefix_difference = native_prefix ^ challenger_prefix
    full_difference = native_full ^ challenger_full
    absent_from_prefix = full_difference - prefix_difference
    return PairInformation(
        challenger_index=challenger_index,
        prefix_nodes_equal=prefix_equal,
        full_nodes_equal=full_equal,
        prefix_equal_full_distinct=prefix_equal and not full_equal,
        prefix_jaccard=_jaccard(native_prefix, challenger_prefix),
        full_jaccard=_jaccard(native_full, challenger_full),
        prefix_symmetric_difference_nodes=len(prefix_difference),
        full_symmetric_difference_nodes=len(full_difference),
        full_distinguishing_nodes_absent_from_prefix=len(absent_from_prefix),
        full_distinguishing_fraction_absent_from_prefix=(
            len(absent_from_prefix) / len(full_difference) if full_difference else 0.0
        ),
        maximum_center_separation_m=float(np.max(separation)),
        maximum_prefix_center_separation_m=float(np.max(separation[: prefix_index + 1])),
        first_divergence_index_by_threshold=first,
        late_divergence_by_threshold=late,
    )


def evaluate_roi_panel(
    realized_local_poses: npt.ArrayLike,
    *,
    challenger_eligible: Sequence[bool],
    deterministic_execution: bool,
    grid: RectangularGrid,
    body: VehicleRectangle,
) -> dict[str, Any]:
    """Evaluate complete-horizon panel support for one frozen ROI candidate."""

    trajectories = np.asarray(realized_local_poses, dtype=np.float64)
    identifiers = ("native",) + tuple(
        f"challenger_{index}" for index in range(1, trajectories.shape[0])
    )
    panel = build_shadow_panel(
        trajectories,
        challenger_eligible=challenger_eligible,
        identifiers=identifiers,
        deterministic_execution=deterministic_execution,
        grid=grid,
        body=body,
    )
    return {
        "panel_valid": panel.valid,
        "panel_reason": panel.reason,
        "candidate_valid": [candidate.valid for candidate in panel.candidates],
        "candidate_reasons": [candidate.reason for candidate in panel.candidates],
    }


def _percentile_interval(values: npt.ArrayLike) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a nonempty finite vector")
    return {
        "mean": float(np.mean(array)),
        "lower": float(np.quantile(array, 0.025)),
        "upper": float(np.quantile(array, 0.975)),
    }


def summarize_candidate_information(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_names: Sequence[str],
    anchors_per_group: int,
    prefix_index: int,
    primary_divergence_threshold_m: float,
    divergence_thresholds_m: Sequence[float],
    roi_precedence: Sequence[str],
    roi_node_counts: Mapping[str, int],
    minimum_valid_anchors_per_group: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Apply the frozen group-level information and ROI selection rules."""

    groups = tuple(group_names)
    if len(groups) != len(set(groups)) or not groups:
        raise ValueError("recording groups must be unique and nonempty")
    expected_keys = {
        (group, anchor) for group in groups for anchor in range(anchors_per_group)
    }
    observed_keys = {
        (str(row["recording_group"]), int(row["anchor_index"])) for row in rows
    }
    if observed_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("rows do not exactly cover the frozen anchor panel")
    roi_ids = tuple(roi_precedence)
    if len(roi_ids) != len(set(roi_ids)) or set(roi_ids) != set(roi_node_counts):
        raise ValueError("ROI precedence and node-count definitions must align")

    threshold_keys = tuple(format(float(value), ".12g") for value in divergence_thresholds_m)
    threshold_key = format(primary_divergence_threshold_m, ".12g")
    if threshold_key not in threshold_keys or len(set(threshold_keys)) != len(threshold_keys):
        raise ValueError("primary and sensitivity divergence thresholds must align")
    group_summaries: dict[str, Any] = {}
    group_exact_rates: list[float] = []
    group_late_rates: dict[str, list[float]] = {key: [] for key in threshold_keys}
    group_tail_fractions: list[float] = []
    exact_counterexample_count = 0
    late_divergence_count = 0
    eligible_pair_count = 0
    for group in groups:
        group_rows = [row for row in rows if row["recording_group"] == group]
        pairs = [
            pair
            for row in group_rows
            for pair in cast(Sequence[Mapping[str, Any]], row["pair_information"])
        ]
        pair_count = len(pairs)
        exact = sum(bool(pair["prefix_equal_full_distinct"]) for pair in pairs)
        late_by_threshold = {
            key: sum(
                bool(cast(Mapping[str, bool], pair["late_divergence_by_threshold"])[key])
                for pair in pairs
            )
            for key in threshold_keys
        }
        tail_fractions = [
            float(pair["full_distinguishing_fraction_absent_from_prefix"])
            for pair in pairs
            if int(pair["full_symmetric_difference_nodes"]) > 0
        ]
        eligible_pair_count += pair_count
        exact_counterexample_count += exact
        late_divergence_count += late_by_threshold[threshold_key]
        exact_rate = exact / pair_count if pair_count else 0.0
        late_rates = {
            key: value / pair_count if pair_count else 0.0
            for key, value in late_by_threshold.items()
        }
        tail_fraction = float(np.mean(tail_fractions)) if tail_fractions else 0.0
        group_exact_rates.append(exact_rate)
        for key in threshold_keys:
            group_late_rates[key].append(late_rates[key])
        group_tail_fractions.append(tail_fraction)
        group_summaries[group] = {
            "eligible_native_challenger_pairs": pair_count,
            "prefix_equal_full_distinct_pairs": exact,
            "late_divergence_pairs_by_threshold_m": late_by_threshold,
            "prefix_equal_full_distinct_rate": exact_rate,
            "late_divergence_rate_by_threshold_m": late_rates,
            "mean_full_distinguishing_fraction_absent_from_prefix": tail_fraction,
        }

    rng = np.random.default_rng(bootstrap_seed)
    exact_draws = np.empty(bootstrap_replicates, dtype=np.float64)
    exact_array = np.asarray(group_exact_rates, dtype=np.float64)
    late_arrays = {
        key: np.asarray(values, dtype=np.float64) for key, values in group_late_rates.items()
    }
    tail_array = np.asarray(group_tail_fractions, dtype=np.float64)
    late_draws = {
        key: np.empty(bootstrap_replicates, dtype=np.float64) for key in threshold_keys
    }
    tail_draws = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        draw = rng.integers(0, len(groups), size=len(groups))
        exact_draws[index] = float(np.mean(exact_array[draw]))
        for key in threshold_keys:
            late_draws[key][index] = float(np.mean(late_arrays[key][draw]))
        tail_draws[index] = float(np.mean(tail_array[draw]))

    roi_summary: dict[str, Any] = {}
    feasible: list[str] = []
    for roi_id in roi_ids:
        counts = {
            group: sum(
                bool(cast(Mapping[str, Any], row["roi_statuses"])[roi_id]["panel_valid"])
                for row in rows
                if row["recording_group"] == group
            )
            for group in groups
        }
        passing = sum(value >= minimum_valid_anchors_per_group for value in counts.values())
        if passing == len(groups):
            feasible.append(roi_id)
        roi_summary[roi_id] = {
            "node_count": int(roi_node_counts[roi_id]),
            "valid_anchor_count": int(sum(counts.values())),
            "valid_anchor_count_by_group": counts,
            "groups_meeting_minimum_coverage": passing,
            "coverage_pass": passing == len(groups),
        }
    precedence_index = {roi_id: index for index, roi_id in enumerate(roi_ids)}
    selected_roi = min(
        feasible,
        key=lambda roi_id: (roi_node_counts[roi_id], precedence_index[roi_id]),
        default=None,
    )
    prefix_verdict = (
        "PREFIX_UNIVERSAL_SUFFICIENCY_FALSIFIED"
        if exact_counterexample_count > 0
        else "NO_EXACT_COUNTEREXAMPLE_OBSERVED_ON_FIT"
    )
    return {
        "group_count": len(groups),
        "anchor_count": len(rows),
        "eligible_native_challenger_pair_count": eligible_pair_count,
        "prefix_index": prefix_index,
        "primary_divergence_threshold_m": primary_divergence_threshold_m,
        "prefix_equal_full_distinct_pair_count": exact_counterexample_count,
        "primary_late_divergence_pair_count": late_divergence_count,
        "group_summaries": group_summaries,
        "group_equal_prefix_collapse_rate": {
            "estimate": float(np.mean(exact_array)),
            "bootstrap_95_interval": _percentile_interval(exact_draws),
        },
        "group_equal_late_divergence_rate_by_threshold_m": {
            key: {
                "estimate": float(np.mean(late_arrays[key])),
                "bootstrap_95_interval": _percentile_interval(late_draws[key]),
            }
            for key in threshold_keys
        },
        "group_equal_full_distinguishing_fraction_absent_from_prefix": {
            "estimate": float(np.mean(tail_array)),
            "bootstrap_95_interval": _percentile_interval(tail_draws),
        },
        "prefix_information_verdict": prefix_verdict,
        "roi_support": roi_summary,
        "selected_minimum_cost_full_horizon_roi": selected_roi,
        "roi_verdict": (
            "FIT_FULL_HORIZON_ROI_SELECTED"
            if selected_roi is not None
            else "STOP_FIXED_UNIFORM_GRID_ROUTE"
        ),
        "next_status": (
            "STOP_PREFIX_AS_FINAL_METHOD__QUALIFY_FULL_HORIZON_ROI"
            if exact_counterexample_count > 0
            else "PREFIX_REMAINS_DIAGNOSTIC_ONLY__OUTCOME_TEST_STILL_REQUIRED"
        ),
    }
