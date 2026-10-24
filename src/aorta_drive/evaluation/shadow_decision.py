"""Read-only shadow-decision diagnostics for R004H7-B.

The geometry in this module is a discrete query proxy on frozen grid nodes. It
is deliberately not a continuous swept-footprint certificate and must not be
used to claim collision or driving safety.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.appearance_mechanism import oriented_grid_edges
from aorta_drive.theory.decision_quotient import (
    decision_error_eta,
    selection_comparison,
    set_min_contrasts,
    set_min_oscillation_bounds,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

DIAGNOSTICS = (
    "decision_contrast_error_m",
    "edge_mae_m",
    "centered_node_mae_m",
    "dino_drift",
)
CONTROL_DIAGNOSTICS = DIAGNOSTICS[1:]


@dataclass(frozen=True)
class ShadowCandidate:
    """One registered candidate and its discrete-node query status."""

    registered_index: int
    identifier: str
    valid: bool
    reason: str
    node_indices: IntArray


@dataclass(frozen=True)
class ShadowPanel:
    """Native-first candidate panel with invalid candidates retained."""

    candidates: tuple[ShadowCandidate, ...]
    valid: bool
    reason: str

    @property
    def query_candidates(self) -> tuple[ShadowCandidate, ...]:
        """Return native and valid challengers in registered order."""

        if not self.valid:
            raise ValueError("invalid panel has no decision query")
        return tuple(candidate for candidate in self.candidates if candidate.valid)

    @property
    def query_node_sets(self) -> tuple[IntArray, ...]:
        """Return node sets aligned to :attr:`query_candidates`."""

        return tuple(candidate.node_indices for candidate in self.query_candidates)


def _as_realized_poses(values: npt.ArrayLike) -> FloatArray:
    poses = np.asarray(values, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] == 0:
        raise ValueError("realized poses must have shape [time,3]")
    if not np.all(np.isfinite(poses)):
        raise ValueError("realized poses must be finite")
    return poses


def _rectangle_corners(
    pose: FloatArray, body: VehicleRectangle
) -> FloatArray:
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    forward = np.asarray((cosine, sine), dtype=np.float64)
    left = np.asarray((-sine, cosine), dtype=np.float64)
    origin = pose[:2]
    corners = [
        origin + longitudinal * forward + lateral * left
        for longitudinal, lateral in (
            (body.front_extent, body.half_width),
            (-body.rear_overhang, body.half_width),
            (-body.rear_overhang, -body.half_width),
            (body.front_extent, -body.half_width),
        )
    ]
    return np.asarray(corners, dtype=np.float64)


def rasterize_discrete_footprint_nodes(
    realized_poses: npt.ArrayLike,
    *,
    grid: RectangularGrid,
    body: VehicleRectangle,
) -> IntArray:
    """Return grid centers inside any discrete-time oriented vehicle rectangle.

    Every oriented rectangle must lie inside the full grid-cell support. This
    rejects ROI clipping instead of silently turning a partial footprint into a
    complete decision query. Boundary comparisons are expanded by one binary64
    step only to make inclusive analytic boundaries deterministic; this remains
    an empirical node query, not conservative continuous geometry.
    """

    poses = _as_realized_poses(realized_poses)
    centers = grid.centers()
    selected = np.zeros(centers.shape[0], dtype=np.bool_)
    longitudinal_low = np.nextafter(-body.rear_overhang, -np.inf)
    longitudinal_high = np.nextafter(body.front_extent, np.inf)
    lateral_low = np.nextafter(-body.half_width, -np.inf)
    lateral_high = np.nextafter(body.half_width, np.inf)
    for pose in poses:
        corners = _rectangle_corners(pose, body)
        if (
            np.min(corners[:, 0]) < grid.x_min
            or np.max(corners[:, 0]) > grid.x_max
            or np.min(corners[:, 1]) < grid.y_min
            or np.max(corners[:, 1]) > grid.y_max
        ):
            raise ValueError("oriented footprint leaves the frozen ROI")
        relative = centers - pose[:2]
        cosine = math.cos(float(pose[2]))
        sine = math.sin(float(pose[2]))
        longitudinal = relative[:, 0] * cosine + relative[:, 1] * sine
        lateral = -relative[:, 0] * sine + relative[:, 1] * cosine
        selected |= (
            (longitudinal >= longitudinal_low)
            & (longitudinal <= longitudinal_high)
            & (lateral >= lateral_low)
            & (lateral <= lateral_high)
        )
    indices = np.flatnonzero(selected).astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError("discrete footprint contains no frozen grid node")
    return indices


def build_shadow_panel(
    realized_local_poses: npt.ArrayLike,
    *,
    challenger_eligible: Sequence[bool],
    identifiers: Sequence[str],
    deterministic_execution: bool,
    grid: RectangularGrid,
    body: VehicleRectangle,
) -> ShadowPanel:
    """Build a native-first panel while preserving every failure reason.

    ``realized_local_poses`` includes native at index zero. Eligibility has one
    entry per challenger and may only encode the frozen, field-independent
    intent contract.
    """

    trajectories = np.asarray(realized_local_poses, dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[2] != 3:
        raise ValueError("realized trajectory batch must have shape [candidate,time,3]")
    if not np.all(np.isfinite(trajectories)):
        raise ValueError("realized trajectory batch must be finite")
    if len(identifiers) != trajectories.shape[0]:
        raise ValueError("candidate identifiers do not align with trajectory batch")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate identifiers must be unique")
    if len(challenger_eligible) != trajectories.shape[0] - 1:
        raise ValueError("challenger eligibility does not align with trajectory batch")

    rows: list[ShadowCandidate] = []
    for index, (identifier, poses) in enumerate(zip(identifiers, trajectories, strict=True)):
        eligible = index == 0 or bool(challenger_eligible[index - 1])
        if not deterministic_execution:
            rows.append(
                ShadowCandidate(
                    index,
                    identifier,
                    False,
                    "execution_not_bit_exact",
                    np.empty(0, dtype=np.int64),
                )
            )
            continue
        if not eligible:
            rows.append(
                ShadowCandidate(
                    index,
                    identifier,
                    False,
                    "intent_ineligible",
                    np.empty(0, dtype=np.int64),
                )
            )
            continue
        try:
            nodes = rasterize_discrete_footprint_nodes(poses, grid=grid, body=body)
        except ValueError as error:
            rows.append(
                ShadowCandidate(index, identifier, False, str(error), np.empty(0, dtype=np.int64))
            )
        else:
            rows.append(ShadowCandidate(index, identifier, True, "valid", nodes))

    native_valid = bool(rows and rows[0].valid)
    challenger_count = sum(candidate.valid for candidate in rows[1:])
    if not native_valid:
        return ShadowPanel(tuple(rows), False, "native_invalid")
    if challenger_count == 0:
        return ShadowPanel(tuple(rows), False, "no_valid_challenger")
    return ShadowPanel(tuple(rows), True, "valid")


def evaluate_shadow_condition(
    prediction: npt.ArrayLike,
    teacher: npt.ArrayLike,
    panel: ShadowPanel,
    *,
    dino_drift: float,
) -> dict[str, Any]:
    """Evaluate one frozen appearance condition on one valid shadow panel."""

    predicted = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(teacher, dtype=np.float64)
    if (
        predicted.ndim != 2
        or predicted.shape != target.shape
        or predicted.shape[0] != predicted.shape[1]
    ):
        raise ValueError("prediction and teacher must be aligned square fields")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(target)):
        raise ValueError("prediction and teacher fields must be finite")
    if not math.isfinite(dino_drift) or dino_drift < 0.0:
        raise ValueError("DINO drift must be finite and nonnegative")
    query_candidates = panel.query_candidates
    node_sets = panel.query_node_sets
    if any(np.max(nodes) >= predicted.size for nodes in node_sets):
        raise ValueError("panel node lies outside the field")

    teacher_contrasts = set_min_contrasts(target.reshape(-1), node_sets)
    predicted_contrasts = set_min_contrasts(predicted.reshape(-1), node_sets)
    eta = decision_error_eta(teacher_contrasts, predicted_contrasts)
    comparison = selection_comparison(teacher_contrasts, predicted_contrasts)
    oscillation_bounds = set_min_oscillation_bounds(
        predicted.reshape(-1), target.reshape(-1), node_sets
    )
    contrast_errors = np.abs(predicted_contrasts - teacher_contrasts)
    teacher_with_native = np.concatenate((np.zeros(1, dtype=np.float64), teacher_contrasts))
    ordered = np.sort(teacher_with_native)
    predicted_centered = predicted - np.mean(predicted)
    teacher_centered = target - np.mean(target)
    return {
        "query_registered_indices": [candidate.registered_index for candidate in query_candidates],
        "teacher_contrasts_m": teacher_contrasts.tolist(),
        "predicted_contrasts_m": predicted_contrasts.tolist(),
        "decision_contrast_error_m": eta,
        "decision_contrast_mae_m": float(np.mean(contrast_errors)),
        "oscillation_bound_max_m": float(np.max(oscillation_bounds)),
        "oscillation_bound_verified": bool(
            np.all(contrast_errors <= np.nextafter(oscillation_bounds, np.inf))
        ),
        "two_eta_regret_bound_verified": bool(comparison.regret <= np.nextafter(2.0 * eta, np.inf)),
        "teacher_selected_registered_index": (
            query_candidates[comparison.teacher_index].registered_index
        ),
        "predicted_selected_registered_index": (
            query_candidates[comparison.predicted_index].registered_index
        ),
        "teacher_regret_m": comparison.regret,
        "ranking_flip": comparison.flip,
        "exact_teacher_tie": comparison.exact_teacher_tie,
        "teacher_top_gap_m": float(ordered[-1] - ordered[-2]),
        "edge_mae_m": float(
            np.mean(np.abs(oriented_grid_edges(predicted) - oriented_grid_edges(target)))
        ),
        "centered_node_mae_m": float(np.mean(np.abs(predicted_centered - teacher_centered))),
        "dino_drift": float(dino_drift),
    }


def _average_ranks(values: FloatArray) -> FloatArray:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("rank input must be a finite non-empty vector")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(first: npt.ArrayLike, second: npt.ArrayLike) -> float:
    """Return tie-aware Spearman correlation, or NaN for a constant vector."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or left.size < 2:
        raise ValueError("Spearman inputs must be aligned vectors of length at least two")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_rank -= np.mean(left_rank)
    right_rank -= np.mean(right_rank)
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left_rank, right_rank) / denominator)


def binary_auroc(scores: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Return tie-aware empirical AUROC, or NaN when one class is absent."""

    values = np.asarray(scores, dtype=np.float64)
    outcomes = np.asarray(labels)
    if values.ndim != 1 or outcomes.shape != values.shape or values.size < 2:
        raise ValueError("AUROC inputs must be aligned vectors of length at least two")
    if not np.all(np.isfinite(values)) or not np.all(np.isin(outcomes, (0, 1, False, True))):
        raise ValueError("AUROC requires finite scores and binary labels")
    positive = outcomes.astype(np.bool_)
    positive_count = int(np.sum(positive))
    negative_count = values.size - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = _average_ranks(values)
    statistic = float(np.sum(ranks[positive]) - positive_count * (positive_count + 1) / 2)
    return statistic / (positive_count * negative_count)


def _ci(values: FloatArray, *, minimum_valid_fraction: float) -> tuple[float, float, float]:
    valid = values[np.isfinite(values)]
    fraction = valid.size / values.size
    if fraction < minimum_valid_fraction or valid.size == 0:
        return float("nan"), float("nan"), float(fraction)
    return (
        float(np.quantile(valid, 0.025)),
        float(np.quantile(valid, 0.975)),
        float(fraction),
    )


def _validate_anchor_ledgers(
    ledgers: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    expected_groups: int,
    anchors_per_group: int,
) -> tuple[tuple[str, ...], dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in ledgers:
        group = str(row["recording_group"])
        anchor = int(row["anchor_index"])
        key = (group, anchor)
        if key in seen:
            raise ValueError("duplicate group/anchor shadow ledger")
        seen.add(key)
        grouped[group].append(row)
    if len(grouped) != expected_groups or any(
        len(rows) != anchors_per_group for rows in grouped.values()
    ):
        raise ValueError("shadow ledger does not contain the complete frozen group panel")
    for rows in grouped.values():
        indices = sorted(int(row["anchor_index"]) for row in rows)
        if indices != list(range(anchors_per_group)):
            raise ValueError("anchor indices are not the frozen contiguous panel")
        for row in rows:
            valid = bool(row["panel_valid"])
            metrics = row.get("conditions")
            if valid:
                if not isinstance(metrics, Mapping) or set(metrics) != set(conditions):
                    raise ValueError("valid panel lacks the exact registered conditions")
                for condition in conditions:
                    item = metrics[condition]
                    if not isinstance(item, Mapping):
                        raise ValueError("condition metric bundle is invalid")
                    required = (
                        *DIAGNOSTICS,
                        "teacher_regret_m",
                        "ranking_flip",
                        "exact_teacher_tie",
                    )
                    if any(name not in item for name in required):
                        raise ValueError("condition metric bundle is incomplete")
                    numeric = [float(item[name]) for name in (*DIAGNOSTICS, "teacher_regret_m")]
                    if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
                        raise ValueError(
                            "shadow diagnostics and outcomes must be finite and nonnegative"
                        )
            elif metrics not in (None, {}):
                raise ValueError("invalid panel must not carry scientific condition metrics")
    names = tuple(sorted(grouped))
    return names, grouped


def summarize_shadow_ledgers(
    ledgers: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    seed: int = 20261024,
    replicates: int = 10_000,
    expected_groups: int = 24,
    anchors_per_group: int = 16,
    minimum_valid_anchors_per_group: int = 12,
    minimum_valid_bootstrap_fraction: float = 0.95,
) -> dict[str, Any]:
    """Apply the frozen group-first H7-B coverage and association gates."""

    if len(conditions) == 0 or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be unique and non-empty")
    if replicates <= 0 or not 0.0 < minimum_valid_bootstrap_fraction <= 1.0:
        raise ValueError("invalid bootstrap contract")
    group_names, grouped = _validate_anchor_ledgers(
        ledgers,
        conditions=conditions,
        expected_groups=expected_groups,
        anchors_per_group=anchors_per_group,
    )
    valid_counts = {
        group: sum(bool(row["panel_valid"]) for row in grouped[group]) for group in group_names
    }
    coverage_passed = all(
        count >= minimum_valid_anchors_per_group for count in valid_counts.values()
    )
    coverage = {
        "required_group_count": expected_groups,
        "observed_group_count": len(group_names),
        "anchors_per_group": anchors_per_group,
        "minimum_valid_anchors_per_group": minimum_valid_anchors_per_group,
        "valid_anchor_count_by_group": valid_counts,
        "valid_anchor_count": int(sum(valid_counts.values())),
        "total_anchor_count": expected_groups * anchors_per_group,
        "passed": coverage_passed,
    }
    if not coverage_passed:
        return {
            "coverage": coverage,
            "decision": "INSUFFICIENT_PANEL_COVERAGE",
            "claim_supported": False,
            "group_summaries": {},
            "association": {},
        }

    group_summaries: dict[str, dict[str, float]] = {}
    observation_by_group: dict[str, list[dict[str, float]]] = {}
    for group in group_names:
        observations: list[dict[str, float]] = []
        for row in grouped[group]:
            if not bool(row["panel_valid"]):
                continue
            condition_map = row["conditions"]
            for condition in conditions:
                item = condition_map[condition]
                observations.append(
                    {
                        **{name: float(item[name]) for name in DIAGNOSTICS},
                        "teacher_regret_m": float(item["teacher_regret_m"]),
                        "ranking_flip": float(bool(item["ranking_flip"])),
                        "exact_teacher_tie": float(bool(item["exact_teacher_tie"])),
                    }
                )
        observation_by_group[group] = observations
        group_summaries[group] = {
            name: float(np.mean([row[name] for row in observations]))
            for name in (*DIAGNOSTICS, "teacher_regret_m", "ranking_flip", "exact_teacher_tie")
        }

    regret = np.asarray([group_summaries[group]["teacher_regret_m"] for group in group_names])
    point_spearman = {
        name: spearman_correlation(
            np.asarray([group_summaries[group][name] for group in group_names]), regret
        )
        for name in DIAGNOSTICS
    }
    all_observations = [row for group in group_names for row in observation_by_group[group]]
    flip = np.asarray([row["ranking_flip"] for row in all_observations])
    point_auroc = {
        name: binary_auroc(np.asarray([row[name] for row in all_observations]), flip)
        for name in DIAGNOSTICS
    }

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(group_names), size=(replicates, len(group_names)))
    spearman_draws = {name: np.full(replicates, np.nan, dtype=np.float64) for name in DIAGNOSTICS}
    auroc_draws = {name: np.full(replicates, np.nan, dtype=np.float64) for name in DIAGNOSTICS}
    for replicate_index, draw in enumerate(draws):
        sampled_groups = [group_names[int(index)] for index in draw]
        sampled_regret = np.asarray(
            [group_summaries[group]["teacher_regret_m"] for group in sampled_groups]
        )
        sampled_observations = [
            row for group in sampled_groups for row in observation_by_group[group]
        ]
        sampled_flip = np.asarray([row["ranking_flip"] for row in sampled_observations])
        for name in DIAGNOSTICS:
            sampled_diagnostic = np.asarray(
                [group_summaries[group][name] for group in sampled_groups]
            )
            spearman_draws[name][replicate_index] = spearman_correlation(
                sampled_diagnostic, sampled_regret
            )
            auroc_draws[name][replicate_index] = binary_auroc(
                np.asarray([row[name] for row in sampled_observations]), sampled_flip
            )

    association: dict[str, Any] = {"spearman": {}, "flip_auroc": {}}
    for name in DIAGNOSTICS:
        low, high, fraction = _ci(
            spearman_draws[name], minimum_valid_fraction=minimum_valid_bootstrap_fraction
        )
        association["spearman"][name] = {
            "estimate": point_spearman[name],
            "ci95": [low, high],
            "valid_bootstrap_fraction": fraction,
        }
        low, high, fraction = _ci(
            auroc_draws[name], minimum_valid_fraction=minimum_valid_bootstrap_fraction
        )
        association["flip_auroc"][name] = {
            "estimate": point_auroc[name],
            "ci95": [low, high],
            "valid_bootstrap_fraction": fraction,
        }

    association["spearman_differences"] = {}
    association["flip_auroc_differences"] = {}
    decision_name = DIAGNOSTICS[0]
    for control in CONTROL_DIAGNOSTICS:
        spearman_difference = spearman_draws[decision_name] - spearman_draws[control]
        low, high, fraction = _ci(
            spearman_difference, minimum_valid_fraction=minimum_valid_bootstrap_fraction
        )
        association["spearman_differences"][control] = {
            "estimate": point_spearman[decision_name] - point_spearman[control],
            "ci95": [low, high],
            "valid_bootstrap_fraction": fraction,
        }
        auroc_difference = auroc_draws[decision_name] - auroc_draws[control]
        low, high, fraction = _ci(
            auroc_difference, minimum_valid_fraction=minimum_valid_bootstrap_fraction
        )
        association["flip_auroc_differences"][control] = {
            "estimate": point_auroc[decision_name] - point_auroc[control],
            "ci95": [low, high],
            "valid_bootstrap_fraction": fraction,
        }

    decision_spearman = association["spearman"][decision_name]
    decision_auroc = association["flip_auroc"][decision_name]
    supported = bool(
        math.isfinite(point_spearman[decision_name])
        and point_spearman[decision_name] > 0.0
        and decision_spearman["ci95"][0] > 0.0
        and decision_auroc["ci95"][0] > 0.5
        and all(
            association["spearman_differences"][control]["ci95"][0] > 0.0
            and association["flip_auroc_differences"][control]["ci95"][0] > 0.0
            for control in CONTROL_DIAGNOSTICS
        )
    )
    return {
        "coverage": coverage,
        "group_summaries": group_summaries,
        "association": association,
        "bootstrap": {
            "seed": seed,
            "replicates": replicates,
            "independent_unit": "recording_group",
            "minimum_valid_fraction": minimum_valid_bootstrap_fraction,
        },
        "decision": "SUPPORTED" if supported else "UNSUPPORTED",
        "claim_supported": supported,
    }
