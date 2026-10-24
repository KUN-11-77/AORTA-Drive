"""Outcome-blind support diagnostics for discrete shadow trajectory panels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.shadow_decision import build_shadow_panel

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PrefixPanelStatus:
    """Outcome-blind validity of one panel at one frozen time prefix."""

    prefix_index: int
    panel_valid: bool
    panel_reason: str
    candidate_valid: tuple[bool, ...]
    candidate_reasons: tuple[str, ...]
    candidate_minimum_slack_m: tuple[float, ...]


@dataclass(frozen=True)
class SupportProfile:
    """Spatial support and geometry diagnostics for one realized trajectory."""

    minimum_slack_m: float
    directional_minimum_slack_m: Mapping[str, float]
    required_expansion_m: Mapping[str, float]
    first_violation_index: int | None
    first_violation_boundaries: tuple[str, ...]
    initial_translation_residual_m: float
    initial_yaw_residual_rad: float
    path_length_m: float
    terminal_x_m: float
    terminal_y_m: float
    terminal_displacement_m: float
    maximum_absolute_lateral_m: float
    total_yaw_variation_rad: float
    maximum_discrete_curvature_proxy_per_m: float


def _poses(values: npt.ArrayLike) -> FloatArray:
    poses = np.asarray(values, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] == 0:
        raise ValueError("trajectory must have shape [time,3]")
    if not np.all(np.isfinite(poses)):
        raise ValueError("trajectory must be finite")
    return poses


def rectangle_corner_series(
    realized_poses: npt.ArrayLike, *, body: VehicleRectangle
) -> FloatArray:
    """Return analytic vehicle corners with shape ``[time,4,2]``."""

    poses = _poses(realized_poses)
    yaw = poses[:, 2]
    forward = np.stack((np.cos(yaw), np.sin(yaw)), axis=1)
    left = np.stack((-np.sin(yaw), np.cos(yaw)), axis=1)
    offsets = (
        (body.front_extent, body.half_width),
        (-body.rear_overhang, body.half_width),
        (-body.rear_overhang, -body.half_width),
        (body.front_extent, -body.half_width),
    )
    return np.stack(
        [
            poses[:, :2]
            + longitudinal * forward
            + lateral * left
            for longitudinal, lateral in offsets
        ],
        axis=1,
    )


def _wrapped_differences(yaw: FloatArray) -> FloatArray:
    differences = np.diff(yaw)
    return cast(FloatArray, np.arctan2(np.sin(differences), np.cos(differences)))


def support_profile(
    realized_poses: npt.ArrayLike,
    *,
    grid: RectangularGrid,
    body: VehicleRectangle,
) -> SupportProfile:
    """Measure exact discrete-pose support without reading predictions or labels."""

    poses = _poses(realized_poses)
    corners = rectangle_corner_series(poses, body=body)
    directional_by_time = {
        "x_min": np.min(corners[:, :, 0], axis=1) - grid.x_min,
        "x_max": grid.x_max - np.max(corners[:, :, 0], axis=1),
        "y_min": np.min(corners[:, :, 1], axis=1) - grid.y_min,
        "y_max": grid.y_max - np.max(corners[:, :, 1], axis=1),
    }
    directional_minimum = {
        name: float(np.min(values)) for name, values in directional_by_time.items()
    }
    required_expansion = {
        name: max(0.0, -value) for name, value in directional_minimum.items()
    }
    violation = np.stack(
        [directional_by_time[name] < 0.0 for name in ("x_min", "x_max", "y_min", "y_max")],
        axis=1,
    )
    violating_indices = np.flatnonzero(np.any(violation, axis=1))
    first = int(violating_indices[0]) if violating_indices.size else None
    boundaries = (
        tuple(
            name
            for index, name in enumerate(("x_min", "x_max", "y_min", "y_max"))
            if violation[first, index]
        )
        if first is not None
        else ()
    )
    translations = np.diff(poses[:, :2], axis=0)
    segment_lengths = np.linalg.norm(translations, axis=1)
    yaw_differences = _wrapped_differences(poses[:, 2])
    curvature = np.divide(
        np.abs(yaw_differences),
        segment_lengths,
        out=np.zeros_like(segment_lengths),
        where=segment_lengths > 1.0e-9,
    )
    initial_yaw = math.atan2(math.sin(float(poses[0, 2])), math.cos(float(poses[0, 2])))
    return SupportProfile(
        minimum_slack_m=min(directional_minimum.values()),
        directional_minimum_slack_m=directional_minimum,
        required_expansion_m=required_expansion,
        first_violation_index=first,
        first_violation_boundaries=boundaries,
        initial_translation_residual_m=float(np.linalg.norm(poses[0, :2])),
        initial_yaw_residual_rad=abs(initial_yaw),
        path_length_m=float(np.sum(segment_lengths)),
        terminal_x_m=float(poses[-1, 0]),
        terminal_y_m=float(poses[-1, 1]),
        terminal_displacement_m=float(np.linalg.norm(poses[-1, :2] - poses[0, :2])),
        maximum_absolute_lateral_m=float(np.max(np.abs(poses[:, 1]))),
        total_yaw_variation_rad=float(np.sum(np.abs(yaw_differences))),
        maximum_discrete_curvature_proxy_per_m=float(np.max(curvature, initial=0.0)),
    )


def evaluate_prefix_panel(
    realized_local_poses: npt.ArrayLike,
    *,
    challenger_eligible: Sequence[bool],
    prefix_index: int,
    deterministic_execution: bool,
    grid: RectangularGrid,
    body: VehicleRectangle,
) -> PrefixPanelStatus:
    """Evaluate support at a fixed prefix without fields, labels, or outcomes.

    Eligibility is frozen from the complete official execution and therefore
    does not change while the support horizon is varied.  The prefix includes
    indices ``0..prefix_index``.
    """

    trajectories = np.asarray(realized_local_poses, dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[2] != 3:
        raise ValueError("realized trajectory batch must have shape [candidate,time,3]")
    if prefix_index < 0 or prefix_index >= trajectories.shape[1]:
        raise ValueError("prefix index is outside the realized trajectory")
    identifiers = ("native",) + tuple(
        f"challenger_{index}" for index in range(1, trajectories.shape[0])
    )
    prefix = trajectories[:, : prefix_index + 1, :]
    panel = build_shadow_panel(
        prefix,
        challenger_eligible=challenger_eligible,
        identifiers=identifiers,
        deterministic_execution=deterministic_execution,
        grid=grid,
        body=body,
    )
    slacks = tuple(
        support_profile(poses, grid=grid, body=body).minimum_slack_m for poses in prefix
    )
    return PrefixPanelStatus(
        prefix_index=prefix_index,
        panel_valid=panel.valid,
        panel_reason=panel.reason,
        candidate_valid=tuple(candidate.valid for candidate in panel.candidates),
        candidate_reasons=tuple(candidate.reason for candidate in panel.candidates),
        candidate_minimum_slack_m=slacks,
    )


def summarize_prefix_qualification(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_names: Sequence[str],
    prefix_indices: Sequence[int],
    anchors_per_group: int,
    minimum_valid_anchors_per_group: int,
) -> dict[str, Any]:
    """Apply the preregistered maximum-feasible-prefix selection rule."""

    groups = tuple(group_names)
    prefixes = tuple(int(value) for value in prefix_indices)
    if len(groups) != len(set(groups)) or not groups:
        raise ValueError("recording groups must be unique and non-empty")
    if tuple(sorted(set(prefixes))) != prefixes or not prefixes:
        raise ValueError("prefix indices must be unique and strictly increasing")
    expected = len(groups) * anchors_per_group
    if len(rows) != expected:
        raise ValueError(f"expected {expected} anchor rows, received {len(rows)}")
    observed_keys = {
        (str(row["recording_group"]), int(row["anchor_index"])) for row in rows
    }
    expected_keys = {
        (group, anchor) for group in groups for anchor in range(anchors_per_group)
    }
    if observed_keys != expected_keys:
        raise ValueError("anchor rows do not exactly cover the registered group panel")

    by_prefix: dict[str, Any] = {}
    feasible: list[int] = []
    for prefix in prefixes:
        group_counts = {
            group: sum(
                bool(row["prefix_statuses"][str(prefix)]["panel_valid"])
                for row in rows
                if row["recording_group"] == group
            )
            for group in groups
        }
        reason_counts: dict[str, int] = {}
        candidate_reason_counts: dict[str, int] = {}
        minimum_slacks: list[float] = []
        for row in rows:
            status = cast(Mapping[str, Any], row["prefix_statuses"][str(prefix)])
            reason = str(status["panel_reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for candidate_reason in cast(Sequence[str], status["candidate_reasons"]):
                candidate_reason_counts[candidate_reason] = (
                    candidate_reason_counts.get(candidate_reason, 0) + 1
                )
            minimum_slacks.extend(
                float(value)
                for value in cast(Sequence[float], status["candidate_minimum_slack_m"])
            )
        passing_groups = sum(
            count >= minimum_valid_anchors_per_group for count in group_counts.values()
        )
        if passing_groups == len(groups):
            feasible.append(prefix)
        by_prefix[str(prefix)] = {
            "valid_anchor_count": sum(group_counts.values()),
            "valid_anchor_count_by_group": group_counts,
            "groups_meeting_minimum_coverage": passing_groups,
            "required_group_count": len(groups),
            "minimum_valid_anchors_per_group": minimum_valid_anchors_per_group,
            "panel_reason_counts": reason_counts,
            "candidate_reason_counts": candidate_reason_counts,
            "candidate_minimum_slack_m": {
                "minimum": float(np.min(minimum_slacks)),
                "median": float(np.median(minimum_slacks)),
                "maximum": float(np.max(minimum_slacks)),
            },
            "coverage_pass": passing_groups == len(groups),
        }
    selected = max(feasible) if feasible else None
    return {
        "group_count": len(groups),
        "anchors_per_group": anchors_per_group,
        "total_anchor_count": expected,
        "prefix_indices": list(prefixes),
        "by_prefix": by_prefix,
        "selected_prefix_index": selected,
        "selected_prefix_seconds": None if selected is None else selected / 10.0,
        "decision": (
            "FIT_PREFIX_SELECTED" if selected is not None else "STOP_CURRENT_FIXED_GRID_SHADOW_ROUTE"
        ),
    }
def group_equal_descriptive_difference(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    invalid_reasons: frozenset[str],
) -> dict[str, object]:
    """Describe invalid-minus-valid differences with recording groups equally weighted."""

    groups: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        group = str(row["recording_group"])
        reason = str(row["panel_reason"])
        bucket = "invalid" if reason in invalid_reasons else "valid"
        groups.setdefault(group, {"invalid": [], "valid": []})[bucket].append(
            float(row[value_key])
        )
    differences = [
        float(np.mean(values["invalid"]) - np.mean(values["valid"]))
        for values in groups.values()
        if values["invalid"] and values["valid"]
    ]
    if not differences:
        return {"paired_group_count": 0, "mean": None, "median": None, "range": None}
    return {
        "paired_group_count": len(differences),
        "mean": float(np.mean(differences)),
        "median": float(np.median(differences)),
        "range": [float(np.min(differences)), float(np.max(differences))],
    }
