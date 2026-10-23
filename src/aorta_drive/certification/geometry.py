"""Deterministic continuous-lifting contracts for Gate A."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class AxisAlignedRectangle:
    """Closed axis-aligned rectangle in the certified physical plane."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("rectangle coordinates must be finite")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("rectangle must have positive area")


@dataclass(frozen=True)
class RectangleNetCertificate:
    """Exact covering-radius certificate for a Cartesian internal point grid."""

    h: float
    x_radius: float
    y_radius: float
    point_count: int


@dataclass(frozen=True)
class ContinuousCorrection:
    """Ledger that lifts a discrete lower bound to the continuous task."""

    h_challenger: float
    map_error_challenger: float
    map_error_native: float
    numerical_optimism: float

    def __post_init__(self) -> None:
        values = (
            self.h_challenger,
            self.map_error_challenger,
            self.map_error_native,
            self.numerical_optimism,
        )
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("all correction terms must be finite and non-negative")

    def apply(self, computed_discrete_bound: float) -> float:
        """Subtract every optimistic term from a computed discrete bound."""

        if not np.isfinite(computed_discrete_bound):
            raise ValueError("computed_discrete_bound must be finite")
        return float(
            computed_discrete_bound
            - self.h_challenger
            - self.map_error_challenger
            - self.map_error_native
            - self.numerical_optimism
        )


def _axis_covering_radius(axis: npt.NDArray[np.float64], low: float, high: float) -> float:
    gaps = np.diff(axis)
    interior = float(np.max(gaps) / 2.0) if gaps.size else 0.0
    return max(float(axis[0] - low), float(high - axis[-1]), interior)


def certify_rectangle_internal_net(
    rectangle: AxisAlignedRectangle,
    points: npt.ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> RectangleNetCertificate:
    """Certify an exact internal ``h``-net for a Cartesian point grid.

    The routine rejects incomplete grids instead of estimating coverage from
    samples. For a complete Cartesian grid, the returned Euclidean covering
    radius is exact for the closed rectangle.
    """

    samples = np.asarray(points, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.shape[0] == 0:
        raise ValueError("points must be a non-empty N-by-2 matrix")
    if not np.all(np.isfinite(samples)):
        raise ValueError("points must be finite")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    inside_x = (samples[:, 0] >= rectangle.x_min - tolerance) & (
        samples[:, 0] <= rectangle.x_max + tolerance
    )
    inside_y = (samples[:, 1] >= rectangle.y_min - tolerance) & (
        samples[:, 1] <= rectangle.y_max + tolerance
    )
    if not np.all(inside_x & inside_y):
        raise ValueError("every certified node must lie inside the footprint")

    x_axis = np.unique(samples[:, 0])
    y_axis = np.unique(samples[:, 1])
    expected = {(float(x), float(y)) for x in x_axis for y in y_axis}
    actual = {(float(x), float(y)) for x, y in samples}
    if len(actual) != samples.shape[0] or actual != expected:
        raise ValueError("points must form a duplicate-free complete Cartesian grid")

    x_radius = _axis_covering_radius(x_axis, rectangle.x_min, rectangle.x_max)
    y_radius = _axis_covering_radius(y_axis, rectangle.y_min, rectangle.y_max)
    return RectangleNetCertificate(
        h=math.hypot(x_radius, y_radius),
        x_radius=x_radius,
        y_radius=y_radius,
        point_count=samples.shape[0],
    )
