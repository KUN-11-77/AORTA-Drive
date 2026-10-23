"""Deterministic continuous-lifting contracts for Gate A.

The production route is deliberately asymmetric.  Challenger nodes cover a
proved outer approximation of the continuous sweep, while native nodes are
proved physical points inside at least one native footprint.  This is the
geometry contract required by the relative-margin theorem; dense sampling by
itself is not a certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

_PI: Final = math.pi


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


@dataclass(frozen=True)
class Pose2D:
    """Rear-axle pose with an unwrapped yaw in radians."""

    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in (self.x, self.y, self.yaw)):
            raise ValueError("pose values must be finite")


@dataclass(frozen=True)
class VehicleRectangle:
    """Closed vehicle body rectangle in rear-axle coordinates."""

    rear_overhang: float
    front_extent: float
    half_width: float

    def __post_init__(self) -> None:
        values = (self.rear_overhang, self.front_extent, self.half_width)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("vehicle extents must be finite and positive")

    @property
    def radius(self) -> float:
        """Maximum distance from the rear axle to a body point."""

        return math.hypot(max(self.rear_overhang, self.front_extent), self.half_width)


@dataclass(frozen=True)
class RectangularGrid:
    """Axis-aligned square-cell graph whose support is the certified ROI."""

    x_min: float
    y_min: float
    columns: int
    rows: int
    resolution: float

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in (self.x_min, self.y_min, self.resolution)):
            raise ValueError("grid values must be finite")
        if self.columns <= 0 or self.rows <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.resolution <= 0:
            raise ValueError("grid resolution must be positive")

    @property
    def x_max(self) -> float:
        return self.x_min + self.columns * self.resolution

    @property
    def y_max(self) -> float:
        return self.y_min + self.rows * self.resolution

    @property
    def cell_radius(self) -> float:
        return self.resolution / math.sqrt(2.0)

    def centers(self) -> npt.NDArray[np.float64]:
        """Return row-major cell centers as an ``N x 2`` matrix."""

        xs = self.x_min + (np.arange(self.columns, dtype=np.float64) + 0.5) * self.resolution
        ys = self.y_min + (np.arange(self.rows, dtype=np.float64) + 0.5) * self.resolution
        xx, yy = np.meshgrid(xs, ys)
        return np.column_stack((xx.ravel(), yy.ravel()))


@dataclass(frozen=True)
class SegmentOuterCover:
    """Mid-pose rectangle dilated by a closed Euclidean ball."""

    midpoint: Pose2D
    inflation: float
    translation_term: float
    rotation_term: float


@dataclass(frozen=True)
class SweptCoverCertificate:
    """Fixed-grid cover of every continuous challenger footprint point."""

    node_indices: tuple[int, ...]
    h: float
    segment_covers: tuple[SegmentOuterCover, ...]
    interpolation: str = "linear-translation-linear-unwrapped-yaw"


@dataclass(frozen=True)
class NativeInteriorCertificate:
    """Fixed-grid nodes proved inside at least one native footprint."""

    node_indices: tuple[int, ...]


def _validate_pose_sequence(poses: tuple[Pose2D, ...]) -> None:
    if not poses:
        raise ValueError("pose sequence must be non-empty")
    for start, end in zip(poses, poses[1:], strict=False):
        if abs(end.yaw - start.yaw) > _PI:
            raise ValueError("adjacent yaws must be unwrapped with absolute change at most pi")


def _segment_outer_cover(
    body: VehicleRectangle,
    start: Pose2D,
    end: Pose2D,
    *,
    pose_error: float,
) -> SegmentOuterCover:
    if not np.isfinite(pose_error) or pose_error < 0:
        raise ValueError("pose_error must be finite and non-negative")
    yaw_change = abs(end.yaw - start.yaw)
    if yaw_change > _PI:
        raise ValueError("adjacent yaws must be unwrapped with absolute change at most pi")
    translation_term = 0.5 * math.hypot(end.x - start.x, end.y - start.y)
    rotation_term = 2.0 * body.radius * math.sin(0.25 * yaw_change)
    return SegmentOuterCover(
        midpoint=Pose2D(
            x=0.5 * (start.x + end.x),
            y=0.5 * (start.y + end.y),
            yaw=0.5 * (start.yaw + end.yaw),
        ),
        inflation=translation_term + rotation_term + pose_error,
        translation_term=translation_term,
        rotation_term=rotation_term,
    )


def _point_distance_to_body_rectangle(
    points: npt.NDArray[np.float64],
    body: VehicleRectangle,
    pose: Pose2D,
) -> npt.NDArray[np.float64]:
    """Exact Euclidean point distance to one oriented body rectangle."""

    relative = points - np.array([pose.x, pose.y], dtype=np.float64)
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    local_x = cosine * relative[:, 0] + sine * relative[:, 1]
    local_y = -sine * relative[:, 0] + cosine * relative[:, 1]
    dx = np.maximum.reduce(
        (-body.rear_overhang - local_x, np.zeros_like(local_x), local_x - body.front_extent)
    )
    dy = np.maximum(np.abs(local_y) - body.half_width, 0.0)
    return np.hypot(dx, dy)


def _cover_bounds(
    body: VehicleRectangle, cover: SegmentOuterCover
) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of an inflated oriented rectangle."""

    cosine = math.cos(cover.midpoint.yaw)
    sine = math.sin(cover.midpoint.yaw)
    local_corners = (
        (-body.rear_overhang, -body.half_width),
        (-body.rear_overhang, body.half_width),
        (body.front_extent, -body.half_width),
        (body.front_extent, body.half_width),
    )
    xs = [cover.midpoint.x + cosine * x - sine * y for x, y in local_corners]
    ys = [cover.midpoint.y + sine * x + cosine * y for x, y in local_corners]
    return (
        min(xs) - cover.inflation,
        max(xs) + cover.inflation,
        min(ys) - cover.inflation,
        max(ys) + cover.inflation,
    )


def certify_swept_outer_cover(
    body: VehicleRectangle,
    poses: tuple[Pose2D, ...],
    grid: RectangularGrid,
    *,
    pose_error: float = 0.0,
    tolerance: float = 1e-12,
) -> SweptCoverCertificate:
    """Certify graph centers covering a linearly interpolated continuous sweep.

    For every adjacent pose pair, the swept body is contained in the body at
    the midpoint pose dilated by ``delta``.  ``delta`` is the sum of half the
    endpoint translation, the exact chord bound for the maximum angular
    deviation from the midpoint, and a declared pose-error radius.  Selecting
    every grid center whose distance to the midpoint body is at most
    ``delta + h`` then covers the outer set, where ``h`` is the cell radius.

    A single pose is treated as a zero-length segment.  The function rejects
    covers that leave the grid support; clipping is never accepted.
    """

    _validate_pose_sequence(poses)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    pairs = tuple(zip(poses, poses[1:], strict=False)) or ((poses[0], poses[0]),)
    covers = tuple(
        _segment_outer_cover(body, start, end, pose_error=pose_error) for start, end in pairs
    )
    centers = grid.centers()
    selected = np.zeros(centers.shape[0], dtype=bool)
    for cover in covers:
        x_min, x_max, y_min, y_max = _cover_bounds(body, cover)
        if (
            x_min < grid.x_min - tolerance
            or x_max > grid.x_max + tolerance
            or y_min < grid.y_min - tolerance
            or y_max > grid.y_max + tolerance
        ):
            raise ValueError("swept outer cover leaves the certified grid ROI")
        distances = _point_distance_to_body_rectangle(centers, body, cover.midpoint)
        selected |= distances <= cover.inflation + grid.cell_radius + tolerance
    indices = tuple(int(index) for index in np.flatnonzero(selected))
    if not indices:
        raise ValueError("swept outer cover produced an empty challenger node set")
    return SweptCoverCertificate(indices, grid.cell_radius, covers)


def certify_native_interior_nodes(
    body: VehicleRectangle,
    poses: tuple[Pose2D, ...],
    grid: RectangularGrid,
    *,
    tolerance: float = 1e-12,
) -> NativeInteriorCertificate:
    """Select graph centers proved inside at least one native footprint."""

    _validate_pose_sequence(poses)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    centers = grid.centers()
    selected = np.zeros(centers.shape[0], dtype=bool)
    for pose in poses:
        distances = _point_distance_to_body_rectangle(centers, body, pose)
        selected |= distances <= tolerance
    indices = tuple(int(index) for index in np.flatnonzero(selected))
    if not indices:
        raise ValueError("native footprint contains no fixed-grid center")
    return NativeInteriorCertificate(indices)


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
