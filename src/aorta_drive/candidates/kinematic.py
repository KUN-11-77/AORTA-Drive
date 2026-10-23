"""Deterministic constant-acceleration/curvature candidate trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DEFAULT_ACCELERATIONS_MPS2 = (0.0, -1.0, -2.0, -3.0, 1.0)
DEFAULT_CURVATURES_PER_M = (0.0, -0.01, 0.01, -0.02, 0.02, -0.04, 0.04)


@dataclass(frozen=True)
class KinematicControl:
    """A constant longitudinal acceleration and signed path curvature."""

    acceleration_mps2: float
    curvature_per_m: float

    @property
    def provenance(self) -> str:
        return (
            "constant-acceleration-curvature-v1/"
            f"a={self.acceleration_mps2:+.6f}/k={self.curvature_per_m:+.6f}"
        )


@dataclass(frozen=True)
class KinematicCandidate:
    """One future rear-axle pose sequence, excluding the current origin."""

    control: KinematicControl
    poses: npt.NDArray[np.float64]
    terminal_speed_mps: float


@dataclass(frozen=True)
class CandidateBatch:
    """Accepted candidates plus deterministic control-level rejection counts."""

    candidates: tuple[KinematicCandidate, ...]
    rejection_counts: dict[str, int]


def _travel_and_terminal_speed(speed: float, acceleration: float, dt: float) -> tuple[float, float]:
    if acceleration < 0.0 and speed + acceleration * dt < 0.0:
        stop_time = speed / -acceleration
        distance = speed * stop_time + 0.5 * acceleration * stop_time * stop_time
        return max(distance, 0.0), 0.0
    terminal_speed = speed + acceleration * dt
    distance = speed * dt + 0.5 * acceleration * dt * dt
    return max(distance, 0.0), max(terminal_speed, 0.0)


def _integrate_control(
    initial_speed_mps: float,
    control: KinematicControl,
    *,
    horizon_s: float,
    interval_s: float,
) -> KinematicCandidate:
    steps_float = horizon_s / interval_s
    steps = int(round(steps_float))
    if not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("horizon_s must be an integer multiple of interval_s")

    poses = np.empty((steps, 3), dtype=np.float64)
    x = 0.0
    y = 0.0
    heading = 0.0
    speed = initial_speed_mps
    curvature = control.curvature_per_m
    for index in range(steps):
        distance, speed = _travel_and_terminal_speed(
            speed, control.acceleration_mps2, interval_s
        )
        next_heading = heading + curvature * distance
        if abs(curvature) <= 1e-15:
            x += distance * math.cos(heading)
            y += distance * math.sin(heading)
        else:
            x += (math.sin(next_heading) - math.sin(heading)) / curvature
            y += (-math.cos(next_heading) + math.cos(heading)) / curvature
        heading = next_heading
        poses[index] = (x, y, heading)
    return KinematicCandidate(control=control, poses=poses, terminal_speed_mps=speed)


def generate_kinematic_candidates(
    initial_speed_mps: float,
    *,
    horizon_s: float = 4.0,
    interval_s: float = 0.5,
    accelerations_mps2: tuple[float, ...] = DEFAULT_ACCELERATIONS_MPS2,
    curvatures_per_m: tuple[float, ...] = DEFAULT_CURVATURES_PER_M,
    maximum_speed_mps: float = 20.0,
    maximum_longitudinal_acceleration_mps2: float = 3.0,
    maximum_lateral_acceleration_mps2: float = 4.0,
) -> CandidateBatch:
    """Generate an ordered candidate library without map, label, or planner access.

    The continuous model is ``ds/dt=v``, ``dv/dt=a``, ``dtheta/ds=kappa``.
    Each time interval is integrated analytically in arc length, with braking
    truncated at zero speed so candidates never reverse.  Controls that exceed
    the frozen speed or lateral-acceleration envelope anywhere on the horizon
    are rejected before rollout.  Duplicate stopped trajectories are removed
    deterministically while preserving control order.
    """

    scalars = (
        initial_speed_mps,
        horizon_s,
        interval_s,
        maximum_speed_mps,
        maximum_longitudinal_acceleration_mps2,
        maximum_lateral_acceleration_mps2,
        *accelerations_mps2,
        *curvatures_per_m,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("candidate-provider inputs must be finite")
    if initial_speed_mps < 0.0 or initial_speed_mps > maximum_speed_mps:
        raise ValueError("initial_speed_mps lies outside the frozen speed envelope")
    if horizon_s <= 0.0 or interval_s <= 0.0:
        raise ValueError("horizon_s and interval_s must be positive")
    if (
        maximum_speed_mps <= 0.0
        or maximum_longitudinal_acceleration_mps2 <= 0.0
        or maximum_lateral_acceleration_mps2 <= 0.0
    ):
        raise ValueError("dynamic limits must be positive")
    if not accelerations_mps2 or not curvatures_per_m:
        raise ValueError("the frozen control grid must be non-empty")

    accepted: list[KinematicCandidate] = []
    seen: set[bytes] = set()
    rejected = {
        "longitudinal_acceleration_envelope": 0,
        "speed_envelope": 0,
        "lateral_acceleration_envelope": 0,
        "duplicate": 0,
    }
    for acceleration in accelerations_mps2:
        if abs(acceleration) > maximum_longitudinal_acceleration_mps2:
            rejected["longitudinal_acceleration_envelope"] += len(curvatures_per_m)
            continue
        maximum_candidate_speed = max(
            initial_speed_mps,
            initial_speed_mps + acceleration * horizon_s,
            0.0,
        )
        for curvature in curvatures_per_m:
            if maximum_candidate_speed > maximum_speed_mps:
                rejected["speed_envelope"] += 1
                continue
            if maximum_candidate_speed**2 * abs(curvature) > maximum_lateral_acceleration_mps2:
                rejected["lateral_acceleration_envelope"] += 1
                continue
            candidate = _integrate_control(
                initial_speed_mps,
                KinematicControl(acceleration, curvature),
                horizon_s=horizon_s,
                interval_s=interval_s,
            )
            key = np.round(candidate.poses, decimals=12).tobytes()
            if key in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(key)
            accepted.append(candidate)
    return CandidateBatch(candidates=tuple(accepted), rejection_counts=rejected)
