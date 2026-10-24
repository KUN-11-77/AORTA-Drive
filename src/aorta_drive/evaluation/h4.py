"""Frozen method-independent eligibility rules for the R004H4 probe."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class IntentEligibilityLimits:
    """Physical budgets that keep challengers comparable to the native intent."""

    minimum_native_terminal_displacement_m: float
    minimum_progress_ratio: float
    maximum_progress_ratio: float
    maximum_terminal_cross_track_m: float
    maximum_terminal_heading_difference_rad: float
    maximum_terminal_speed_difference_mps: float

    def __post_init__(self) -> None:
        values = (
            self.minimum_native_terminal_displacement_m,
            self.minimum_progress_ratio,
            self.maximum_progress_ratio,
            self.maximum_terminal_cross_track_m,
            self.maximum_terminal_heading_difference_rad,
            self.maximum_terminal_speed_difference_mps,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("eligibility limits must be finite and non-negative")
        if self.minimum_progress_ratio > self.maximum_progress_ratio:
            raise ValueError("progress-ratio limits are reversed")


@dataclass(frozen=True)
class IntentEligibility:
    """One deterministic challenger/native intent comparison."""

    eligible: bool
    reason: str
    progress_ratio: float
    terminal_cross_track_m: float
    terminal_heading_difference_rad: float
    terminal_speed_difference_mps: float


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def qualify_intent_match(
    challenger_poses: npt.ArrayLike,
    native_poses: npt.ArrayLike,
    *,
    challenger_terminal_speed_mps: float,
    native_terminal_speed_mps: float,
    limits: IntentEligibilityLimits,
) -> IntentEligibility:
    """Fail closed unless endpoint progress, cross-track, yaw and speed all match.

    Both trajectories must use the same frozen time samples and include the
    current rear-axle origin as their first pose.  Progress and cross-track are
    measured in the terminal displacement frame of the native trajectory, so
    this filter does not read the road teacher or any method prediction.
    """

    challenger = np.asarray(challenger_poses, dtype=np.float64)
    native = np.asarray(native_poses, dtype=np.float64)
    if (
        challenger.ndim != 2
        or challenger.shape != native.shape
        or challenger.shape[0] < 2
        or challenger.shape[1] != 3
    ):
        raise ValueError("trajectories must have aligned shape [time>=2, 3]")
    if not np.all(np.isfinite(challenger)) or not np.all(np.isfinite(native)):
        raise ValueError("trajectory poses must be finite")
    if not np.array_equal(challenger[0], np.zeros(3)) or not np.array_equal(
        native[0], np.zeros(3)
    ):
        raise ValueError("both trajectories must begin at the exact ego origin")
    if not all(
        math.isfinite(speed) and speed >= 0.0
        for speed in (challenger_terminal_speed_mps, native_terminal_speed_mps)
    ):
        raise ValueError("terminal speeds must be finite and non-negative")

    native_terminal = native[-1, :2]
    native_displacement = float(np.linalg.norm(native_terminal))
    if native_displacement < limits.minimum_native_terminal_displacement_m:
        return IntentEligibility(
            False,
            "native_progress_too_small",
            math.nan,
            math.nan,
            math.nan,
            math.nan,
        )
    direction = native_terminal / native_displacement
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    challenger_terminal = challenger[-1, :2]
    progress_ratio = float(np.dot(challenger_terminal, direction) / native_displacement)
    cross_track = abs(float(np.dot(challenger_terminal - native_terminal, normal)))
    heading_difference = abs(_wrap_angle(float(challenger[-1, 2] - native[-1, 2])))
    speed_difference = abs(challenger_terminal_speed_mps - native_terminal_speed_mps)
    metrics = (progress_ratio, cross_track, heading_difference, speed_difference)
    if not all(math.isfinite(value) for value in metrics):
        raise ValueError("eligibility computation produced a non-finite metric")

    reason = "eligible"
    if progress_ratio < limits.minimum_progress_ratio:
        reason = "progress_below_budget"
    elif progress_ratio > limits.maximum_progress_ratio:
        reason = "progress_above_budget"
    elif cross_track > limits.maximum_terminal_cross_track_m:
        reason = "cross_track_above_budget"
    elif heading_difference > limits.maximum_terminal_heading_difference_rad:
        reason = "heading_above_budget"
    elif speed_difference > limits.maximum_terminal_speed_difference_mps:
        reason = "speed_above_budget"
    return IntentEligibility(
        eligible=reason == "eligible",
        reason=reason,
        progress_ratio=progress_ratio,
        terminal_cross_track_m=cross_track,
        terminal_heading_difference_rad=heading_difference,
        terminal_speed_difference_mps=speed_difference,
    )
