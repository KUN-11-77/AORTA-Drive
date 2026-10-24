"""Native-conditioned, endpoint-invariant candidate trajectories.

The provider reconstructs a C2 piecewise-quintic rear-axle curve from the
frozen native samples and adds a fixed-direction polynomial bump that is C2
zero at 0 and 3.5 seconds.  The final half-second is copied exactly from the
native.  Dynamic limits are screened over complete polynomial segments with
recursive Bernstein bounds; accepted candidates are never qualified from the
0.5-second output samples alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.polynomial import Polynomial


@dataclass(frozen=True)
class NativeOffsetControl:
    """One signed lateral amplitude and its immutable construction settings."""

    amplitude_m: float
    bump_end_s: float

    @property
    def provenance(self) -> str:
        return (
            "native-fixed-normal-c2-bump-v1/"
            f"delta={self.amplitude_m:+.6f}/end={self.bump_end_s:.6f}"
        )


@dataclass(frozen=True)
class NativeOffsetCandidate:
    """One accepted candidate, excluding the current origin sample."""

    control: NativeOffsetControl
    poses: npt.NDArray[np.float64]
    reference_normal_xy: tuple[float, float]
    maximum_output_sample_speed_mps: float
    maximum_output_sample_total_acceleration_mps2: float
    bernstein_subdivision_count: int


@dataclass(frozen=True)
class NativeOffsetBatch:
    """Accepted candidates and fail-closed rejection counts."""

    candidates: tuple[NativeOffsetCandidate, ...]
    rejection_counts: dict[str, int]


@dataclass(frozen=True)
class _Segment:
    """Two coordinate polynomials over local u in [0, 1]."""

    start_s: float
    duration_s: float
    x: Polynomial
    y: Polynomial


def endpoint_invariant_bump(u: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Evaluate ``64 u^3 (1-u)^3`` on the closed unit interval."""

    values = np.asarray(u, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("bump coordinates must lie in [0, 1]")
    return 64.0 * values**3 * (1.0 - values) ** 3


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _quintic_coefficients(
    p0: float,
    p1: float,
    v0: float,
    v1: float,
    a0: float,
    a1: float,
    duration_s: float,
) -> npt.NDArray[np.float64]:
    c0 = p0
    c1 = duration_s * v0
    c2 = 0.5 * duration_s**2 * a0
    position_residual = p1 - (c0 + c1 + c2)
    velocity_residual = duration_s * v1 - (c1 + 2.0 * c2)
    acceleration_residual = duration_s**2 * a1 - 2.0 * c2
    return np.asarray(
        [
            c0,
            c1,
            c2,
            10.0 * position_residual
            - 4.0 * velocity_residual
            + 0.5 * acceleration_residual,
            -15.0 * position_residual
            + 7.0 * velocity_residual
            - acceleration_residual,
            6.0 * position_residual
            - 3.0 * velocity_residual
            + 0.5 * acceleration_residual,
        ],
        dtype=np.float64,
    )


def _native_segments(poses: npt.NDArray[np.float64], interval_s: float) -> tuple[_Segment, ...]:
    positions = poses[:, :2]
    headings = np.unwrap(poses[:, 2])
    chord_speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / interval_s
    speeds = np.empty(len(poses), dtype=np.float64)
    speeds[0] = chord_speeds[0]
    speeds[-1] = chord_speeds[-1]
    speeds[1:-1] = 0.5 * (chord_speeds[:-1] + chord_speeds[1:])
    velocities = speeds[:, None] * np.column_stack((np.cos(headings), np.sin(headings)))
    accelerations = np.empty_like(velocities)
    accelerations[0] = (velocities[1] - velocities[0]) / interval_s
    accelerations[-1] = (velocities[-1] - velocities[-2]) / interval_s
    accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2.0 * interval_s)

    segments: list[_Segment] = []
    for index in range(len(poses) - 1):
        x = Polynomial(
            _quintic_coefficients(
                positions[index, 0],
                positions[index + 1, 0],
                velocities[index, 0],
                velocities[index + 1, 0],
                accelerations[index, 0],
                accelerations[index + 1, 0],
                interval_s,
            )
        )
        y = Polynomial(
            _quintic_coefficients(
                positions[index, 1],
                positions[index + 1, 1],
                velocities[index, 1],
                velocities[index + 1, 1],
                accelerations[index, 1],
                accelerations[index + 1, 1],
                interval_s,
            )
        )
        segments.append(_Segment(index * interval_s, interval_s, x, y))
    return tuple(segments)


def _bump_on_segment(start_s: float, duration_s: float, bump_end_s: float) -> Polynomial:
    if start_s >= bump_end_s:
        return Polynomial([0.0])
    if start_s + duration_s > bump_end_s + 1e-12:
        raise ValueError("bump endpoint must align with a native sample")
    z = Polynomial([start_s / bump_end_s, duration_s / bump_end_s])
    return 64.0 * z**3 * (1.0 - z) ** 3


def _perturb_segments(
    native: tuple[_Segment, ...],
    *,
    amplitude_m: float,
    normal: npt.NDArray[np.float64],
    bump_end_s: float,
) -> tuple[_Segment, ...]:
    result: list[_Segment] = []
    for segment in native:
        bump = _bump_on_segment(segment.start_s, segment.duration_s, bump_end_s)
        result.append(
            _Segment(
                segment.start_s,
                segment.duration_s,
                segment.x + amplitude_m * normal[0] * bump,
                segment.y + amplitude_m * normal[1] * bump,
            )
        )
    return tuple(result)


def _power_to_bernstein(coefficients: npt.ArrayLike) -> npt.NDArray[np.float64]:
    power = np.asarray(coefficients, dtype=np.float64)
    if power.ndim != 1 or power.size == 0 or not np.all(np.isfinite(power)):
        raise ValueError("power coefficients must be one finite vector")
    degree = power.size - 1
    bernstein = np.empty_like(power)
    for k in range(degree + 1):
        bernstein[k] = sum(
            power[i] * math.comb(k, i) / math.comb(degree, i) for i in range(k + 1)
        )
    return bernstein


def _subdivide_bernstein(
    coefficients: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    levels = [coefficients]
    while levels[-1].size > 1:
        previous = levels[-1]
        levels.append(0.5 * (previous[:-1] + previous[1:]))
    left = np.asarray([level[0] for level in levels], dtype=np.float64)
    right = np.asarray([level[-1] for level in reversed(levels)], dtype=np.float64)
    return left, right


def _certify_nonpositive(
    polynomial: Polynomial,
    *,
    strict_margin: float,
    maximum_depth: int,
) -> tuple[bool, int]:
    initial = _power_to_bernstein(polynomial.coef)
    stack: list[tuple[npt.NDArray[np.float64], int]] = [(initial, 0)]
    subdivisions = 0
    while stack:
        coefficients, depth = stack.pop()
        if float(np.max(coefficients)) <= -strict_margin:
            continue
        if float(np.min(coefficients)) > -strict_margin:
            return False, subdivisions
        if depth >= maximum_depth:
            return False, subdivisions
        left, right = _subdivide_bernstein(coefficients)
        subdivisions += 1
        stack.append((right, depth + 1))
        stack.append((left, depth + 1))
    return True, subdivisions


def _segment_dynamics(
    segment: _Segment,
    *,
    maximum_speed_mps: float,
    maximum_longitudinal_acceleration_mps2: float,
    maximum_lateral_acceleration_mps2: float,
    maximum_total_acceleration_mps2: float,
    strict_margin: float,
    maximum_depth: int,
) -> tuple[bool, int]:
    vx = segment.x.deriv() / segment.duration_s
    vy = segment.y.deriv() / segment.duration_s
    ax = vx.deriv() / segment.duration_s
    ay = vy.deriv() / segment.duration_s
    speed_sq = vx * vx + vy * vy
    dot = vx * ax + vy * ay
    cross = vx * ay - vy * ax
    acceleration_sq = ax * ax + ay * ay
    constraints = (
        speed_sq - maximum_speed_mps**2,
        dot * dot - maximum_longitudinal_acceleration_mps2**2 * speed_sq,
        cross * cross - maximum_lateral_acceleration_mps2**2 * speed_sq,
        acceleration_sq - maximum_total_acceleration_mps2**2,
    )
    total_subdivisions = 0
    for constraint in constraints:
        passed, subdivisions = _certify_nonpositive(
            constraint,
            strict_margin=strict_margin,
            maximum_depth=maximum_depth,
        )
        total_subdivisions += subdivisions
        if not passed:
            return False, total_subdivisions
    return True, total_subdivisions


def _sample_segment(
    segment: _Segment, local_u: float
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    position = np.asarray([segment.x(local_u), segment.y(local_u)], dtype=np.float64)
    velocity = np.asarray(
        [segment.x.deriv()(local_u), segment.y.deriv()(local_u)], dtype=np.float64
    ) / segment.duration_s
    acceleration = np.asarray(
        [segment.x.deriv(2)(local_u), segment.y.deriv(2)(local_u)], dtype=np.float64
    ) / segment.duration_s**2
    return position, velocity, acceleration


def generate_native_offset_candidates(
    native_poses: npt.ArrayLike,
    *,
    interval_s: float = 0.5,
    bump_end_s: float = 3.5,
    amplitudes_m: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5),
    minimum_reference_displacement_m: float = 2.0,
    maximum_speed_mps: float = 20.0,
    maximum_longitudinal_acceleration_mps2: float = 3.0,
    maximum_lateral_acceleration_mps2: float = 4.0,
    maximum_total_acceleration_mps2: float = 5.0,
    minimum_sample_tangent_speed_mps: float = 1e-3,
    bernstein_strict_margin: float = 1e-9,
    bernstein_maximum_depth: int = 24,
    enforce_desired_curve_dynamic_screen: bool = True,
) -> NativeOffsetBatch:
    """Generate endpoint-invariant candidates and fail closed on dynamics.

    The reference direction is the normal to the native terminal displacement.
    The output contains samples at the same times as ``native_poses`` except the
    current origin.  From ``bump_end_s`` onward, samples are copied bit-for-bit.
    """

    native = np.asarray(native_poses, dtype=np.float64)
    scalar_values = (
        interval_s,
        bump_end_s,
        minimum_reference_displacement_m,
        maximum_speed_mps,
        maximum_longitudinal_acceleration_mps2,
        maximum_lateral_acceleration_mps2,
        maximum_total_acceleration_mps2,
        minimum_sample_tangent_speed_mps,
        bernstein_strict_margin,
        *amplitudes_m,
    )
    if native.ndim != 2 or native.shape[0] < 3 or native.shape[1] != 3:
        raise ValueError("native_poses must have shape [time>=3, 3]")
    if not np.all(np.isfinite(native)) or not all(math.isfinite(v) for v in scalar_values):
        raise ValueError("provider inputs must be finite")
    if not np.array_equal(native[0], np.zeros(3)):
        raise ValueError("native trajectory must start at the exact ego origin")
    if interval_s <= 0.0 or bump_end_s <= 0.0:
        raise ValueError("time settings must be positive")
    horizon_s = (len(native) - 1) * interval_s
    bump_steps = round(bump_end_s / interval_s)
    if (
        not math.isclose(bump_steps * interval_s, bump_end_s, abs_tol=1e-12, rel_tol=0.0)
        or bump_end_s >= horizon_s
    ):
        raise ValueError("bump endpoint must be an interior native sample")
    if not amplitudes_m or any(amplitude == 0.0 for amplitude in amplitudes_m):
        raise ValueError("amplitudes must be non-empty and nonzero")
    if len(set(amplitudes_m)) != len(amplitudes_m):
        raise ValueError("amplitudes must be unique")
    if any(limit <= 0.0 for limit in scalar_values[:1] + scalar_values[2:9]):
        raise ValueError("physical and arithmetic limits must be positive")
    if bernstein_maximum_depth < 1:
        raise ValueError("Bernstein maximum depth must be positive")

    terminal = native[-1, :2]
    terminal_displacement = float(np.linalg.norm(terminal))
    rejected = {
        "native_reference_displacement": 0,
        "continuous_dynamic_envelope": 0,
        "sample_tangent_degeneracy": 0,
        "duplicate": 0,
    }
    if terminal_displacement < minimum_reference_displacement_m:
        rejected["native_reference_displacement"] = len(amplitudes_m)
        return NativeOffsetBatch(candidates=(), rejection_counts=rejected)
    direction = terminal / terminal_displacement
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    native_segments = _native_segments(native, interval_s)
    accepted: list[NativeOffsetCandidate] = []
    seen: set[bytes] = set()
    for amplitude in amplitudes_m:
        control = NativeOffsetControl(amplitude, bump_end_s)
        segments = _perturb_segments(
            native_segments,
            amplitude_m=amplitude,
            normal=normal,
            bump_end_s=bump_end_s,
        )
        subdivisions = 0
        dynamics_pass = True
        if enforce_desired_curve_dynamic_screen:
            for segment in segments[:bump_steps]:
                passed, count = _segment_dynamics(
                    segment,
                    maximum_speed_mps=maximum_speed_mps,
                    maximum_longitudinal_acceleration_mps2=(
                        maximum_longitudinal_acceleration_mps2
                    ),
                    maximum_lateral_acceleration_mps2=maximum_lateral_acceleration_mps2,
                    maximum_total_acceleration_mps2=maximum_total_acceleration_mps2,
                    strict_margin=bernstein_strict_margin,
                    maximum_depth=bernstein_maximum_depth,
                )
                subdivisions += count
                if not passed:
                    dynamics_pass = False
                    break
        if not dynamics_pass:
            rejected["continuous_dynamic_envelope"] += 1
            continue

        poses = np.empty((len(native) - 1, 3), dtype=np.float64)
        maximum_speed = 0.0
        maximum_acceleration = 0.0
        tangent_degenerate = False
        for sample_index in range(1, len(native)):
            sample_time = sample_index * interval_s
            if sample_time >= bump_end_s:
                poses[sample_index - 1] = native[sample_index]
                continue
            segment = segments[sample_index - 1]
            position, velocity, acceleration = _sample_segment(segment, 1.0)
            speed = float(np.linalg.norm(velocity))
            if speed < minimum_sample_tangent_speed_mps:
                tangent_degenerate = True
                break
            poses[sample_index - 1, :2] = position
            poses[sample_index - 1, 2] = _wrap_angle(math.atan2(velocity[1], velocity[0]))
            maximum_speed = max(maximum_speed, speed)
            maximum_acceleration = max(
                maximum_acceleration, float(np.linalg.norm(acceleration))
            )
        if tangent_degenerate:
            rejected["sample_tangent_degeneracy"] += 1
            continue
        if not np.all(np.isfinite(poses)):
            raise ValueError("candidate construction produced non-finite samples")
        key = np.round(poses, decimals=12).tobytes()
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)
        accepted.append(
            NativeOffsetCandidate(
                control=control,
                poses=poses,
                reference_normal_xy=(float(normal[0]), float(normal[1])),
                maximum_output_sample_speed_mps=maximum_speed,
                maximum_output_sample_total_acceleration_mps2=maximum_acceleration,
                bernstein_subdivision_count=subdivisions,
            )
        )
    return NativeOffsetBatch(candidates=tuple(accepted), rejection_counts=rejected)
