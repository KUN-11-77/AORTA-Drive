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
from fractions import Fraction
from typing import Final

import numpy as np
import numpy.typing as npt

_PI: Final = math.pi


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(value)


def _sqrt_fraction_upper(value: Fraction) -> float:
    """Return a binary64 value proved no smaller than the exact square root."""

    if value < 0:
        raise ValueError("cannot take the square root of a negative value")
    if value == 0:
        return 0.0
    estimate = math.sqrt(float(value))
    if not math.isfinite(estimate):
        raise ValueError("square root is outside finite binary64 range")
    while _fraction(estimate) ** 2 < value:
        estimate = math.nextafter(estimate, math.inf)
    while True:
        predecessor = math.nextafter(estimate, -math.inf)
        if predecessor <= 0 or _fraction(predecessor) ** 2 < value:
            return estimate
        estimate = predecessor


def _add_up(*values: float) -> float:
    total = Fraction(0)
    for value in values:
        if not np.isfinite(value) or value < 0:
            raise ValueError("outward sum terms must be finite and non-negative")
        total += _fraction(value)
    result = float(total)
    if not math.isfinite(result):
        raise ValueError("outward sum overflowed")
    if _fraction(result) < total:
        result = math.nextafter(result, math.inf)
    return result


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
        result = computed_discrete_bound
        for correction in (
            self.h_challenger,
            self.map_error_challenger,
            self.map_error_native,
            self.numerical_optimism,
        ):
            if correction == 0.0:
                continue
            result = math.nextafter(result - correction, -math.inf)
        if not math.isfinite(result):
            raise ValueError("continuous correction overflowed")
        return result


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
        """Outward-rounded maximum distance from rear axle to a body point."""

        longitudinal = _fraction(max(self.rear_overhang, self.front_extent))
        lateral = _fraction(self.half_width)
        return _sqrt_fraction_upper(longitudinal * longitudinal + lateral * lateral)

    @property
    def inner_radius(self) -> float:
        """Radius of a rear-axle disk contained in the body at every yaw."""

        return min(self.rear_overhang, self.front_extent, self.half_width)


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
        """Certified radius from an ideal cell point to its stored node."""

        resolution = _fraction(self.resolution)
        half = resolution / 2
        x_error = max(
            abs(_fraction(float(self._ideal_x(column))) - self._ideal_x(column))
            for column in range(self.columns)
        )
        y_error = max(
            abs(_fraction(float(self._ideal_y(row))) - self._ideal_y(row))
            for row in range(self.rows)
        )
        x_radius = half + x_error
        y_radius = half + y_error
        return _sqrt_fraction_upper(x_radius * x_radius + y_radius * y_radius)

    def _ideal_x(self, column: int) -> Fraction:
        return _fraction(self.x_min) + (Fraction(column) + Fraction(1, 2)) * _fraction(
            self.resolution
        )

    def _ideal_y(self, row: int) -> Fraction:
        return _fraction(self.y_min) + (Fraction(row) + Fraction(1, 2)) * _fraction(self.resolution)

    def centers(self) -> npt.NDArray[np.float64]:
        """Return row-major cell centers as an ``N x 2`` matrix."""

        xs = np.fromiter(
            (float(self._ideal_x(column)) for column in range(self.columns)),
            dtype=np.float64,
            count=self.columns,
        )
        ys = np.fromiter(
            (float(self._ideal_y(row)) for row in range(self.rows)),
            dtype=np.float64,
            count=self.rows,
        )
        xx, yy = np.meshgrid(xs, ys)
        return np.column_stack((xx.ravel(), yy.ravel()))

    @property
    def x_support(self) -> tuple[Fraction, Fraction]:
        low = _fraction(self.x_min)
        return low, low + self.columns * _fraction(self.resolution)

    @property
    def y_support(self) -> tuple[Fraction, Fraction]:
        low = _fraction(self.y_min)
        return low, low + self.rows * _fraction(self.resolution)


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


@dataclass(frozen=True)
class SegmentDiskCover:
    """Rotation-free rear-axle disk enclosing one continuous body sweep."""

    midpoint_x: Fraction
    midpoint_y: Fraction
    radius: float
    translation_radius: float
    vehicle_radius: float


@dataclass(frozen=True)
class DiskSweptCoverCertificate:
    """Arithmetic-safe fixed-grid cover using rotation-free body disks."""

    node_indices: tuple[int, ...]
    h: float
    segment_covers: tuple[SegmentDiskCover, ...]
    interpolation: str = "linear-translation-arbitrary-yaw"


@dataclass(frozen=True)
class SweptInnerCellCertificate:
    """Grid cells proved contained in at least one continuous footprint."""

    cell_indices: tuple[int, ...]


@dataclass(frozen=True)
class BidirectionalNonInclusionCertificate:
    """Conservative area lower bounds for both continuous set differences."""

    challenger_minus_native_area_lower_m2: float
    native_minus_challenger_area_lower_m2: float
    challenger_witness_cells: tuple[int, ...]
    native_witness_cells: tuple[int, ...]


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


def _exact_squared_distance(
    point_x: float,
    point_y: float,
    center_x: Fraction,
    center_y: Fraction,
) -> Fraction:
    dx = _fraction(point_x) - center_x
    dy = _fraction(point_y) - center_y
    return dx * dx + dy * dy


def certify_swept_disk_cover(
    body: VehicleRectangle,
    poses: tuple[Pose2D, ...],
    grid: RectangularGrid,
    *,
    pose_error: float = 0.0,
) -> DiskSweptCoverCertificate:
    """Certify a rotation-free continuous-sweep cover with exact comparisons.

    Every oriented body is contained in a rear-axle disk of radius R_K. Under
    linear translation, each rear axle stays within half the endpoint
    displacement of the segment midpoint. The construction is independent of
    yaw interpolation and uses exact rational squared-distance comparisons.
    """

    if not poses:
        raise ValueError("pose sequence must be non-empty")
    if not np.isfinite(pose_error) or pose_error < 0:
        raise ValueError("pose_error must be finite and non-negative")
    pairs = tuple(zip(poses, poses[1:], strict=False)) or ((poses[0], poses[0]),)
    vehicle_radius = body.radius
    covers: list[SegmentDiskCover] = []
    centers = grid.centers()
    selected = np.zeros(centers.shape[0], dtype=bool)
    x_low, x_high = grid.x_support
    y_low, y_high = grid.y_support
    h = grid.cell_radius

    for start, end in pairs:
        midpoint_x = (_fraction(start.x) + _fraction(end.x)) / 2
        midpoint_y = (_fraction(start.y) + _fraction(end.y)) / 2
        dx = _fraction(end.x) - _fraction(start.x)
        dy = _fraction(end.y) - _fraction(start.y)
        translation_radius = _sqrt_fraction_upper((dx * dx + dy * dy) / 4)
        radius = _add_up(translation_radius, vehicle_radius, pose_error)
        radius_exact = _fraction(radius)
        if (
            midpoint_x - radius_exact < x_low
            or midpoint_x + radius_exact > x_high
            or midpoint_y - radius_exact < y_low
            or midpoint_y + radius_exact > y_high
        ):
            raise ValueError("swept disk cover leaves the certified grid ROI")

        threshold = _add_up(radius, h)
        threshold_squared = _fraction(threshold) ** 2
        for index, center in enumerate(centers):
            distance_squared = _exact_squared_distance(
                float(center[0]), float(center[1]), midpoint_x, midpoint_y
            )
            if distance_squared <= threshold_squared:
                selected[index] = True
        covers.append(
            SegmentDiskCover(
                midpoint_x=midpoint_x,
                midpoint_y=midpoint_y,
                radius=radius,
                translation_radius=translation_radius,
                vehicle_radius=vehicle_radius,
            )
        )

    indices = tuple(int(index) for index in np.flatnonzero(selected))
    if not indices:
        raise ValueError("swept disk cover produced an empty challenger node set")
    return DiskSweptCoverCertificate(indices, h, tuple(covers))


def certify_native_inner_disk_nodes(
    body: VehicleRectangle,
    poses: tuple[Pose2D, ...],
    grid: RectangularGrid,
    *,
    pose_error: float = 0.0,
) -> NativeInteriorCertificate:
    """Select nodes in rear-axle disks guaranteed inside native rectangles."""

    if not poses:
        raise ValueError("pose sequence must be non-empty")
    if not np.isfinite(pose_error) or pose_error < 0:
        raise ValueError("pose_error must be finite and non-negative")
    certified_radius = _fraction(body.inner_radius) - _fraction(pose_error)
    if certified_radius <= 0:
        raise ValueError("pose_error leaves no non-empty native inner disk")
    radius_squared = certified_radius * certified_radius
    centers = grid.centers()
    selected = np.zeros(centers.shape[0], dtype=bool)
    for pose in poses:
        center_x = _fraction(pose.x)
        center_y = _fraction(pose.y)
        for index, center in enumerate(centers):
            distance_squared = _exact_squared_distance(
                float(center[0]), float(center[1]), center_x, center_y
            )
            if distance_squared <= radius_squared:
                selected[index] = True
    indices = tuple(int(index) for index in np.flatnonzero(selected))
    if not indices:
        raise ValueError("native inner disks contain no fixed-grid center")
    return NativeInteriorCertificate(indices)


def certify_swept_inner_disk_cells(
    body: VehicleRectangle,
    poses: tuple[Pose2D, ...],
    grid: RectangularGrid,
    *,
    pose_error: float = 0.0,
) -> SweptInnerCellCertificate:
    """Select full grid cells contained in at least one footprint.

    Each selected square cell is enclosed by the disk of radius ``grid.cell_radius``
    about its center.  The nominal rear-axle disk of radius
    ``body.inner_radius - pose_error`` is contained in the physical body for every
    yaw, so the complete selected cell is contained in an actual footprint.  The
    construction deliberately does not infer continuous area from point samples.
    """

    if not poses:
        raise ValueError("pose sequence must be non-empty")
    if not np.isfinite(pose_error) or pose_error < 0:
        raise ValueError("pose_error must be finite and non-negative")
    available_radius = (
        _fraction(body.inner_radius) - _fraction(pose_error) - _fraction(grid.cell_radius)
    )
    if available_radius <= 0:
        return SweptInnerCellCertificate(())
    radius_squared = available_radius * available_radius
    centers = grid.centers()
    selected = np.zeros(centers.shape[0], dtype=bool)
    for pose in poses:
        center_x = _fraction(pose.x)
        center_y = _fraction(pose.y)
        for index, center in enumerate(centers):
            if (
                _exact_squared_distance(
                    float(center[0]), float(center[1]), center_x, center_y
                )
                <= radius_squared
            ):
                selected[index] = True
    return SweptInnerCellCertificate(tuple(int(index) for index in np.flatnonzero(selected)))


def _cells_disjoint_from_disk_cover(
    cell_indices: tuple[int, ...],
    cover: DiskSweptCoverCertificate,
    grid: RectangularGrid,
) -> tuple[int, ...]:
    centers = grid.centers()
    witnesses: list[int] = []
    for index in cell_indices:
        center = centers[index]
        disjoint = True
        for segment in cover.segment_covers:
            threshold = _add_up(segment.radius, grid.cell_radius)
            if (
                _exact_squared_distance(
                    float(center[0]),
                    float(center[1]),
                    segment.midpoint_x,
                    segment.midpoint_y,
                )
                <= _fraction(threshold) ** 2
            ):
                disjoint = False
                break
        if disjoint:
            witnesses.append(index)
    return tuple(witnesses)


def _cell_area_lower(count: int, resolution: float) -> float:
    exact = count * _fraction(resolution) ** 2
    result = float(exact)
    if _fraction(result) > exact:
        result = math.nextafter(result, -math.inf)
    return result


def certify_disk_noninclusion_from_covers(
    challenger_outer: DiskSweptCoverCertificate,
    challenger_inner: SweptInnerCellCertificate,
    native_outer: DiskSweptCoverCertificate,
    native_inner: SweptInnerCellCertificate,
    grid: RectangularGrid,
) -> BidirectionalNonInclusionCertificate:
    """Compose precomputed sound inner/outer certificates into area bounds."""

    challenger_witnesses = _cells_disjoint_from_disk_cover(
        challenger_inner.cell_indices,
        native_outer,
        grid,
    )
    native_witnesses = _cells_disjoint_from_disk_cover(
        native_inner.cell_indices,
        challenger_outer,
        grid,
    )
    return BidirectionalNonInclusionCertificate(
        challenger_minus_native_area_lower_m2=_cell_area_lower(
            len(challenger_witnesses), grid.resolution
        ),
        native_minus_challenger_area_lower_m2=_cell_area_lower(
            len(native_witnesses), grid.resolution
        ),
        challenger_witness_cells=challenger_witnesses,
        native_witness_cells=native_witnesses,
    )


def certify_bidirectional_disk_noninclusion(
    body: VehicleRectangle,
    challenger_poses: tuple[Pose2D, ...],
    native_poses: tuple[Pose2D, ...],
    grid: RectangularGrid,
    *,
    challenger_pose_error: float = 0.0,
    native_pose_error: float = 0.0,
) -> BidirectionalNonInclusionCertificate:
    """Lower-bound both continuous swept-footprint difference areas.

    For either direction, a witness cell must be wholly inside one trajectory's
    physical footprint at a frozen pose and wholly outside every disk in the
    other trajectory's continuous outer cover.  Distinct grid cells have
    disjoint interiors, so their total cell area is a valid lower bound on the
    corresponding continuous set difference.
    """

    challenger_outer = certify_swept_disk_cover(
        body,
        challenger_poses,
        grid,
        pose_error=challenger_pose_error,
    )
    native_outer = certify_swept_disk_cover(
        body,
        native_poses,
        grid,
        pose_error=native_pose_error,
    )
    challenger_inner = certify_swept_inner_disk_cells(
        body,
        challenger_poses,
        grid,
        pose_error=challenger_pose_error,
    )
    native_inner = certify_swept_inner_disk_cells(
        body,
        native_poses,
        grid,
        pose_error=native_pose_error,
    )
    return certify_disk_noninclusion_from_covers(
        challenger_outer,
        challenger_inner,
        native_outer,
        native_inner,
        grid,
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
