"""Outcome-blind support diagnostics for discrete shadow trajectory panels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle

FloatArray = npt.NDArray[np.float64]


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
