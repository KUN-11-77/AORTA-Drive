"""Frozen method-independent eligibility rules for the R004H4 probe."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from aorta_drive.certification.geometry import Pose2D
from aorta_drive.certification.transform import certify_global_to_ego


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


@dataclass(frozen=True)
class CachedNativeTrajectory:
    """Frozen PDM-Closed samples represented on the current ego graph."""

    poses: npt.NDArray[np.float64]
    terminal_speed_mps: float
    maximum_transform_error_m: float
    canonical_sample_sha256: str


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def terminal_segment_average_speed(
    poses: npt.ArrayLike, *, interval_s: float
) -> float:
    """Return a representation-neutral speed over the final sampled interval.

    The quantity is derived only from consecutive rear-axle positions, so it is
    comparable even when a trajectory container does not populate dynamic-state
    velocity fields.  It is an interval average, not an instantaneous speed.
    """

    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("poses must have shape [time>=2, dimensions>=2]")
    if not np.all(np.isfinite(array[:, :2])):
        raise ValueError("pose positions must be finite")
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("interval_s must be finite and positive")
    displacement = float(np.linalg.norm(array[-1, :2] - array[-2, :2]))
    return displacement / interval_s


def sample_cached_native_trajectory(cache: Any) -> CachedNativeTrajectory:
    """Sample a NAVSIM metric-cache native at the frozen 0:0.5:4.0 s times."""

    current = cache.ego_state
    if cache.timepoint != current.time_point or cache.trajectory.start_time != cache.timepoint:
        raise ValueError("metric-cache native trajectory does not start at the current ego state")
    samples: list[list[float]] = [[0.0, 0.0, 0.0]]
    transform_errors = [0.0]
    canonical_rows = [
        [
            float(current.time_point.time_us),
            float(current.rear_axle.x),
            float(current.rear_axle.y),
            float(current.rear_axle.heading),
        ]
    ]
    terminal = current
    pose = Pose2D(
        float(current.rear_axle.x),
        float(current.rear_axle.y),
        float(current.rear_axle.heading),
    )
    for step in range(1, 9):
        query = type(cache.timepoint)(cache.timepoint.time_us + step * 500_000)
        terminal = cache.trajectory.get_state_at_time(query)
        transform = certify_global_to_ego(
            (float(terminal.rear_axle.x), float(terminal.rear_axle.y)), pose
        )
        samples.append(
            [
                transform.x,
                transform.y,
                _wrap_angle(float(terminal.rear_axle.heading - current.rear_axle.heading)),
            ]
        )
        transform_errors.append(transform.error_radius)
        canonical_rows.append(
            [
                float(terminal.time_point.time_us),
                float(terminal.rear_axle.x),
                float(terminal.rear_axle.y),
                float(terminal.rear_axle.heading),
            ]
        )
    array = np.asarray(samples, dtype=np.float64)
    if array.shape != (9, 3) or not np.all(np.isfinite(array)):
        raise ValueError("native trajectory did not produce nine finite local poses")
    velocity = terminal.dynamic_car_state.rear_axle_velocity_2d
    terminal_speed = math.hypot(float(velocity.x), float(velocity.y))
    canonical = np.asarray(canonical_rows, dtype="<f8")
    native_hash = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    return CachedNativeTrajectory(
        poses=array,
        terminal_speed_mps=terminal_speed,
        maximum_transform_error_m=max(transform_errors),
        canonical_sample_sha256=native_hash,
    )


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
