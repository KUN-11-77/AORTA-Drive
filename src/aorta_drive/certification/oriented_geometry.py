"""Interval-safe orientation-aware swept-rectangle certificates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from aorta_drive.certification.geometry import (
    Pose2D,
    RectangularGrid,
    VehicleRectangle,
)
from aorta_drive.certification.transform import (
    RationalInterval,
    certify_ego_to_global,
    certify_global_to_ego,
)


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(value)


def _float_up(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("outward conversion exceeded finite binary64 range")
    if _fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _sqrt_fraction_upper(value: Fraction) -> float:
    if value < 0:
        raise ValueError("cannot take the square root of a negative value")
    if value == 0:
        return 0.0
    result = math.sqrt(float(value))
    if not math.isfinite(result):
        raise ValueError("square root exceeded finite binary64 range")
    while _fraction(result) ** 2 < value:
        result = math.nextafter(result, math.inf)
    return result


def _exact_grid_coordinate(value: Fraction) -> float:
    stored = float(value)
    if not math.isfinite(stored) or _fraction(stored) != value:
        raise ValueError("grid cell boundary is not exactly representable in binary64")
    return stored


@dataclass(frozen=True)
class OrientedSegmentCover:
    """Stored midpoint body dilated by a certified Euclidean radius."""

    midpoint: Pose2D
    inflation_m: float
    translation_term_m: float
    rotation_term_m: float
    midpoint_storage_term_m: float
    pose_error_m: float


@dataclass(frozen=True)
class OrientedSweptCoverCertificate:
    """Fixed-grid h-cover of an orientation-aware continuous body sweep."""

    node_indices: tuple[int, ...]
    h: float
    segment_covers: tuple[OrientedSegmentCover, ...]
    interpolation: str = "linear-translation-linear-unwrapped-yaw"


@dataclass(frozen=True)
class OrientedInnerCellCertificate:
    """Complete grid cells certified inside at least one oriented body pose."""

    cell_indices: tuple[int, ...]


@dataclass(frozen=True)
class OrientedBidirectionalNonInclusion:
    """Conservative area lower bounds from oriented outer and inner contracts."""

    challenger_minus_native_area_lower_m2: float
    native_minus_challenger_area_lower_m2: float
    challenger_witness_cells: tuple[int, ...]
    native_witness_cells: tuple[int, ...]


def _segment_cover(
    body: VehicleRectangle,
    start: Pose2D,
    end: Pose2D,
    *,
    pose_error: float,
) -> OrientedSegmentCover:
    if not np.isfinite(pose_error) or pose_error < 0:
        raise ValueError("pose_error must be finite and non-negative")
    yaw_delta = _fraction(end.yaw) - _fraction(start.yaw)
    if abs(float(yaw_delta)) > math.pi:
        raise ValueError("adjacent yaws must be unwrapped with absolute change at most pi")

    exact_mid_x = (_fraction(start.x) + _fraction(end.x)) / 2
    exact_mid_y = (_fraction(start.y) + _fraction(end.y)) / 2
    exact_mid_yaw = (_fraction(start.yaw) + _fraction(end.yaw)) / 2
    mid_x = float(exact_mid_x)
    mid_y = float(exact_mid_y)
    mid_yaw = float(exact_mid_yaw)
    if not all(np.isfinite(value) for value in (mid_x, mid_y, mid_yaw)):
        raise ValueError("segment midpoint exceeded finite binary64 range")

    midpoint_translation_error = _sqrt_fraction_upper(
        (_fraction(mid_x) - exact_mid_x) ** 2
        + (_fraction(mid_y) - exact_mid_y) ** 2
    )
    midpoint_yaw_error = abs(_fraction(mid_yaw) - exact_mid_yaw)
    dx = _fraction(end.x) - _fraction(start.x)
    dy = _fraction(end.y) - _fraction(start.y)
    translation = _sqrt_fraction_upper((dx * dx + dy * dy) / 4)

    # sin(x) <= x for x >= 0. This is slightly looser than the exact chord
    # term but avoids trusting libm in the proof path.
    rotation_exact = _fraction(body.radius) * abs(yaw_delta) / 2
    rotation = _float_up(rotation_exact)
    storage_exact = (
        _fraction(midpoint_translation_error)
        + _fraction(body.radius) * midpoint_yaw_error
    )
    storage = _float_up(storage_exact)
    inflation_exact = (
        _fraction(translation)
        + _fraction(rotation)
        + _fraction(storage)
        + _fraction(pose_error)
    )
    return OrientedSegmentCover(
        midpoint=Pose2D(mid_x, mid_y, mid_yaw),
        inflation_m=_float_up(inflation_exact),
        translation_term_m=translation,
        rotation_term_m=rotation,
        midpoint_storage_term_m=storage,
        pose_error_m=pose_error,
    )


def _axis_distance_lower(
    interval: RationalInterval, low: Fraction, high: Fraction
) -> Fraction:
    if interval.upper < low:
        return low - interval.upper
    if interval.lower > high:
        return interval.lower - high
    return Fraction(0)


def _distance_squared_lower_to_body(
    global_point: tuple[float, float],
    body: VehicleRectangle,
    pose: Pose2D,
) -> Fraction:
    local = certify_global_to_ego(global_point, pose)
    dx = _axis_distance_lower(
        local.x_interval, -_fraction(body.rear_overhang), _fraction(body.front_extent)
    )
    dy = _axis_distance_lower(
        local.y_interval, -_fraction(body.half_width), _fraction(body.half_width)
    )
    return dx * dx + dy * dy


def _body_cover_inside_grid(
    body: VehicleRectangle,
    cover: OrientedSegmentCover,
    grid: RectangularGrid,
) -> bool:
    inflation = _fraction(cover.inflation_m)
    x_low, x_high = grid.x_support
    y_low, y_high = grid.y_support
    for local_point in (
        (-body.rear_overhang, -body.half_width),
        (-body.rear_overhang, body.half_width),
        (body.front_extent, -body.half_width),
        (body.front_extent, body.half_width),
    ):
        global_corner = certify_ego_to_global(local_point, cover.midpoint)
        if (
            global_corner.x_interval.lower - inflation < x_low
            or global_corner.x_interval.upper + inflation > x_high
            or global_corner.y_interval.lower - inflation < y_low
            or global_corner.y_interval.upper + inflation > y_high
        ):
            return False
    return True


def certify_oriented_swept_outer_cover(
    body: VehicleRectangle,
    poses: tuple[Pose2D, ...],
    grid: RectangularGrid,
    *,
    pose_error: float = 0.0,
) -> OrientedSweptCoverCertificate:
    """Certify an h-cover of a piecewise-linear oriented rectangle sweep."""

    if not poses:
        raise ValueError("pose sequence must be non-empty")
    pairs = tuple(zip(poses, poses[1:], strict=False)) or ((poses[0], poses[0]),)
    covers = tuple(
        _segment_cover(body, start, end, pose_error=pose_error) for start, end in pairs
    )
    centers = grid.centers()
    selected = np.zeros(centers.shape[0], dtype=bool)
    h = grid.cell_radius
    for cover in covers:
        if not _body_cover_inside_grid(body, cover, grid):
            raise ValueError("oriented swept cover leaves the certified grid ROI")
        threshold_squared = (
            _fraction(cover.inflation_m) + _fraction(h)
        ) ** 2
        for index, center in enumerate(centers):
            lower_squared = _distance_squared_lower_to_body(
                (float(center[0]), float(center[1])), body, cover.midpoint
            )
            if lower_squared <= threshold_squared:
                selected[index] = True
    indices = tuple(int(index) for index in np.flatnonzero(selected))
    if not indices:
        raise ValueError("oriented swept cover produced an empty node set")
    return OrientedSweptCoverCertificate(indices, h, covers)


def _cell_corners(grid: RectangularGrid, index: int) -> tuple[tuple[float, float], ...]:
    row, column = divmod(index, grid.columns)
    x0 = _fraction(grid.x_min) + column * _fraction(grid.resolution)
    x1 = x0 + _fraction(grid.resolution)
    y0 = _fraction(grid.y_min) + row * _fraction(grid.resolution)
    y1 = y0 + _fraction(grid.resolution)
    return (
        (_exact_grid_coordinate(x0), _exact_grid_coordinate(y0)),
        (_exact_grid_coordinate(x0), _exact_grid_coordinate(y1)),
        (_exact_grid_coordinate(x1), _exact_grid_coordinate(y0)),
        (_exact_grid_coordinate(x1), _exact_grid_coordinate(y1)),
    )


def certify_oriented_inner_cells(
    body: VehicleRectangle,
    poses: tuple[Pose2D, ...],
    grid: RectangularGrid,
    *,
    pose_error: float = 0.0,
) -> OrientedInnerCellCertificate:
    """Certify complete grid cells inside at least one sampled oriented body."""

    if not poses:
        raise ValueError("pose sequence must be non-empty")
    if not np.isfinite(pose_error) or pose_error < 0:
        raise ValueError("pose_error must be finite and non-negative")
    error = _fraction(pose_error)
    x_low = -_fraction(body.rear_overhang) + error
    x_high = _fraction(body.front_extent) - error
    y_low = -_fraction(body.half_width) + error
    y_high = _fraction(body.half_width) - error
    if x_low >= x_high or y_low >= y_high:
        return OrientedInnerCellCertificate(())
    selected: list[int] = []
    for index in range(grid.rows * grid.columns):
        corners = _cell_corners(grid, index)
        inside_any_pose = False
        for pose in poses:
            intervals = [certify_global_to_ego(corner, pose) for corner in corners]
            if all(
                local.x_interval.lower >= x_low
                and local.x_interval.upper <= x_high
                and local.y_interval.lower >= y_low
                and local.y_interval.upper <= y_high
                for local in intervals
            ):
                inside_any_pose = True
                break
        if inside_any_pose:
            selected.append(index)
    return OrientedInnerCellCertificate(tuple(selected))


def _cells_disjoint_from_oriented_cover(
    cells: tuple[int, ...],
    cover: OrientedSweptCoverCertificate,
    body: VehicleRectangle,
    grid: RectangularGrid,
) -> tuple[int, ...]:
    centers = grid.centers()
    witnesses: list[int] = []
    for index in cells:
        center = (float(centers[index, 0]), float(centers[index, 1]))
        disjoint = True
        for segment in cover.segment_covers:
            threshold_squared = (
                _fraction(segment.inflation_m) + _fraction(grid.cell_radius)
            ) ** 2
            if (
                _distance_squared_lower_to_body(center, body, segment.midpoint)
                <= threshold_squared
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


def certify_oriented_bidirectional_noninclusion(
    body: VehicleRectangle,
    challenger_poses: tuple[Pose2D, ...],
    native_poses: tuple[Pose2D, ...],
    grid: RectangularGrid,
    *,
    pose_error: float = 0.0,
) -> OrientedBidirectionalNonInclusion:
    """Lower-bound both continuous set differences with oriented certificates."""

    challenger_outer = certify_oriented_swept_outer_cover(
        body, challenger_poses, grid, pose_error=pose_error
    )
    native_outer = certify_oriented_swept_outer_cover(
        body, native_poses, grid, pose_error=pose_error
    )
    challenger_inner = certify_oriented_inner_cells(
        body, challenger_poses, grid, pose_error=pose_error
    )
    native_inner = certify_oriented_inner_cells(
        body, native_poses, grid, pose_error=pose_error
    )
    challenger_witnesses = _cells_disjoint_from_oriented_cover(
        challenger_inner.cell_indices, native_outer, body, grid
    )
    native_witnesses = _cells_disjoint_from_oriented_cover(
        native_inner.cell_indices, challenger_outer, body, grid
    )
    return OrientedBidirectionalNonInclusion(
        challenger_minus_native_area_lower_m2=_cell_area_lower(
            len(challenger_witnesses), grid.resolution
        ),
        native_minus_challenger_area_lower_m2=_cell_area_lower(
            len(native_witnesses), grid.resolution
        ),
        challenger_witness_cells=challenger_witnesses,
        native_witness_cells=native_witnesses,
    )
