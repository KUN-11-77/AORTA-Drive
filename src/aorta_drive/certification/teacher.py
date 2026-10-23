"""Certified continuous polygon teacher and explicit map-error ledger."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import numpy.typing as npt

Point = tuple[float, float]
Ring = tuple[Point, ...]


def _as_fraction(value: float) -> Fraction:
    return Fraction.from_float(value)


def _validate_ring(ring: Ring) -> None:
    if len(ring) < 3:
        raise ValueError("a polygon ring needs at least three vertices")
    if not all(np.isfinite(coordinate) for point in ring for coordinate in point):
        raise ValueError("polygon coordinates must be finite")
    if ring[0] == ring[-1]:
        raise ValueError("rings must omit the repeated closing vertex")
    for start, end in zip(ring, (*ring[1:], ring[0]), strict=True):
        if start == end:
            raise ValueError("zero-length polygon edges are not allowed")


@dataclass(frozen=True)
class PolygonComponent:
    """One validated raw polygon component with zero or more hole rings."""

    outer: Ring
    holes: tuple[Ring, ...] = ()

    def __post_init__(self) -> None:
        _validate_ring(self.outer)
        for hole in self.holes:
            _validate_ring(hole)


@dataclass(frozen=True)
class SignedDistanceInterval:
    """Closed binary64 interval enclosing the exact signed distance."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in (self.lower, self.upper)):
            raise ValueError("distance interval must be finite")
        if self.lower > self.upper:
            raise ValueError("distance interval endpoints are reversed")


@dataclass(frozen=True)
class CertifiedNodeLabel:
    """Binary64 node label and a deterministic absolute-error upper bound."""

    value: float
    distance_error: float
    interval: SignedDistanceInterval


@dataclass(frozen=True)
class MapErrorLedger:
    """Independent correction terms between a node label and paper teacher."""

    transform_error: float
    polygon_error: float
    distance_error: float
    rounding_error: float

    def __post_init__(self) -> None:
        values = (
            self.transform_error,
            self.polygon_error,
            self.distance_error,
            self.rounding_error,
        )
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("map-error terms must be finite and non-negative")

    @property
    def total(self) -> float:
        total = 0.0
        for value in (
            self.transform_error,
            self.polygon_error,
            self.distance_error,
            self.rounding_error,
        ):
            total = _add_nonnegative_up(total, value)
        return total


def _add_nonnegative_up(left: float, right: float) -> float:
    exact = _as_fraction(left) + _as_fraction(right)
    result = left + right
    if not math.isfinite(result):
        raise ValueError("map-error ledger overflowed")
    if _as_fraction(result) < exact:
        result = math.nextafter(result, math.inf)
    return result


def _fraction_to_float_upper(value: Fraction) -> float:
    if value < 0:
        raise ValueError("upper conversion requires a non-negative value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value is outside finite binary64 range")
    if _as_fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _sqrt_fraction_interval(value: Fraction) -> tuple[float, float]:
    if value < 0:
        raise ValueError("cannot take the square root of a negative value")
    if value == 0:
        return 0.0, 0.0
    estimate = math.sqrt(float(value))
    if not math.isfinite(estimate):
        raise ValueError("distance is outside finite binary64 range")
    squared = _as_fraction(estimate) ** 2
    if squared == value:
        return estimate, estimate
    if squared < value:
        lower = estimate
        upper = math.nextafter(estimate, math.inf)
        while _as_fraction(upper) ** 2 < value:
            upper = math.nextafter(upper, math.inf)
        return lower, upper
    upper = estimate
    lower = math.nextafter(estimate, -math.inf)
    while _as_fraction(lower) ** 2 > value:
        lower = math.nextafter(lower, -math.inf)
    return lower, upper


def _point_segment_squared_distance(point: Point, start: Point, end: Point) -> Fraction:
    px, py = map(_as_fraction, point)
    ax, ay = map(_as_fraction, start)
    bx, by = map(_as_fraction, end)
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denominator = vx * vx + vy * vy
    numerator = wx * vx + wy * vy
    if numerator <= 0:
        return wx * wx + wy * wy
    if numerator >= denominator:
        dx, dy = px - bx, py - by
        return dx * dx + dy * dy
    perpendicular_x = wx * denominator - numerator * vx
    perpendicular_y = wy * denominator - numerator * vy
    return (perpendicular_x * perpendicular_x + perpendicular_y * perpendicular_y) / (
        denominator * denominator
    )


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    px, py = map(_as_fraction, point)
    ax, ay = map(_as_fraction, start)
    bx, by = map(_as_fraction, end)
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if cross != 0:
        return False
    return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)


def _ring_status(point: Point, ring: Ring) -> int:
    """Return -1 outside, 0 on boundary, and 1 inside using exact rationals."""

    px, py = map(_as_fraction, point)
    inside = False
    for start, end in zip(ring, (*ring[1:], ring[0]), strict=True):
        if _point_on_segment(point, start, end):
            return 0
        ax, ay = map(_as_fraction, start)
        bx, by = map(_as_fraction, end)
        if (ay > py) == (by > py):
            continue
        intersection_x = ax + (py - ay) * (bx - ax) / (by - ay)
        if intersection_x > px:
            inside = not inside
    return 1 if inside else -1


def _component_status(point: Point, component: PolygonComponent) -> int:
    outer_status = _ring_status(point, component.outer)
    if outer_status <= 0:
        return outer_status
    for hole in component.holes:
        hole_status = _ring_status(point, hole)
        if hole_status == 0:
            return 0
        if hole_status == 1:
            return -1
    return 1


class ExactPolygonTeacher:
    """Signed clearance to a frozen polygon cover and all of its raw boundaries.

    Membership is the union of all components.  The magnitude is distance to
    the union of every input ring, including boundaries that become internal
    when components overlap.  This remains 1-Lipschitz and is conservative
    inside the polygon union, while avoiding a numerically unverified overlay.

    Binary64 coordinates are interpreted as exact real numbers. Rings must be
    prevalidated as simple and holes must be valid and contained. This class
    deliberately does not silently repair invalid map topology.
    """

    def __init__(self, components: tuple[PolygonComponent, ...]) -> None:
        if not components:
            raise ValueError("polygon teacher needs at least one component")
        self._components = components
        self._component_bounds = tuple(
            (
                min(point[0] for point in component.outer),
                max(point[0] for point in component.outer),
                min(point[1] for point in component.outer),
                max(point[1] for point in component.outer),
            )
            for component in components
        )
        self._segments = tuple(
            (start, end)
            for component in components
            for ring in (component.outer, *component.holes)
            for start, end in zip(ring, (*ring[1:], ring[0]), strict=True)
        )
        segment_array = np.asarray(
            [(*start, *end) for start, end in self._segments],
            dtype=np.float64,
        )
        self._segment_bounds = np.column_stack(
            (
                np.minimum(segment_array[:, 0], segment_array[:, 2]),
                np.maximum(segment_array[:, 0], segment_array[:, 2]),
                np.minimum(segment_array[:, 1], segment_array[:, 3]),
                np.maximum(segment_array[:, 1], segment_array[:, 3]),
            )
        )

    @property
    def component_count(self) -> int:
        """Number of raw polygon components in the frozen teacher object."""

        return len(self._components)

    @property
    def segment_count(self) -> int:
        """Number of raw boundary segments used by the distance teacher."""

        return len(self._segments)

    def _candidate_segment_indices(self, point: Point) -> npt.NDArray[np.int64]:
        """Return a superset containing every exactly nearest raw segment.

        Binary64 AABB-distance operations are rounded one representable value
        toward negative infinity after every positive subtraction, square, and
        sum.  The resulting values are lower bounds on exact squared segment
        distance.  One exact segment distance supplies an upper bound; every
        AABB whose lower bound does not exceed it is evaluated exactly.
        """

        px, py = point
        bounds = self._segment_bounds
        zero = np.zeros(bounds.shape[0], dtype=np.float64)
        dx_approx = np.maximum.reduce((bounds[:, 0] - px, zero, px - bounds[:, 1]))
        dy_approx = np.maximum.reduce((bounds[:, 2] - py, zero, py - bounds[:, 3]))
        dx_lower = np.where(dx_approx > 0, np.nextafter(dx_approx, -np.inf), 0.0)
        dy_lower = np.where(dy_approx > 0, np.nextafter(dy_approx, -np.inf), 0.0)
        dx_squared = dx_lower * dx_lower
        dy_squared = dy_lower * dy_lower
        dx_squared = np.where(
            dx_squared > 0,
            np.nextafter(dx_squared, -np.inf),
            0.0,
        )
        dy_squared = np.where(
            dy_squared > 0,
            np.nextafter(dy_squared, -np.inf),
            0.0,
        )
        lower = dx_squared + dy_squared
        lower = np.where(lower > 0, np.nextafter(lower, -np.inf), 0.0)
        initial_index = int(np.argmin(lower))
        initial_upper = _point_segment_squared_distance(point, *self._segments[initial_index])
        upper_float = _fraction_to_float_upper(initial_upper)
        return np.flatnonzero(lower <= upper_float).astype(np.int64, copy=False)

    def signed_distance_interval(self, point: Point) -> SignedDistanceInterval:
        if not all(np.isfinite(coordinate) for coordinate in point):
            raise ValueError("query point must be finite")
        px, py = point
        statuses = tuple(
            _component_status(point, component)
            for component, (x_min, x_max, y_min, y_max) in zip(
                self._components, self._component_bounds, strict=True
            )
            if x_min <= px <= x_max and y_min <= py <= y_max
        )
        if 0 in statuses:
            return SignedDistanceInterval(0.0, 0.0)
        inside = 1 in statuses
        candidate_indices = self._candidate_segment_indices(point)
        squared_distance = min(
            _point_segment_squared_distance(point, *self._segments[int(index)])
            for index in candidate_indices
        )
        lower, upper = _sqrt_fraction_interval(squared_distance)
        if inside:
            return SignedDistanceInterval(lower, upper)
        return SignedDistanceInterval(-upper, -lower)

    def certified_node_label(self, point: Point) -> CertifiedNodeLabel:
        interval = self.signed_distance_interval(point)
        exact_midpoint = (_as_fraction(interval.lower) + _as_fraction(interval.upper)) / 2
        value = float(exact_midpoint)
        error = max(
            abs(_as_fraction(value) - _as_fraction(interval.lower)),
            abs(_as_fraction(value) - _as_fraction(interval.upper)),
        )
        return CertifiedNodeLabel(
            value=value,
            distance_error=_fraction_to_float_upper(error),
            interval=interval,
        )
