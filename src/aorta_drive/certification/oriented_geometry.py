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
    x_min, x_max, y_min, y_max = _cover_global_bounds(body, cover)
    x_low, x_high = grid.x_support
    y_low, y_high = grid.y_support
    return x_min >= x_low and x_max <= x_high and y_min >= y_low and y_max <= y_high


def _cover_global_bounds(
    body: VehicleRectangle,
    cover: OrientedSegmentCover,
    *,
    extra_radius: float = 0.0,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Return exact-rational AABB bounds enclosing a dilated oriented body."""

    if not np.isfinite(extra_radius) or extra_radius < 0:
        raise ValueError("extra_radius must be finite and non-negative")
    inflation = _fraction(cover.inflation_m) + _fraction(extra_radius)
    x_lows: list[Fraction] = []
    x_highs: list[Fraction] = []
    y_lows: list[Fraction] = []
    y_highs: list[Fraction] = []
    for local_point in (
        (-body.rear_overhang, -body.half_width),
        (-body.rear_overhang, body.half_width),
        (body.front_extent, -body.half_width),
        (body.front_extent, body.half_width),
    ):
        global_corner = certify_ego_to_global(local_point, cover.midpoint)
        x_lows.append(global_corner.x_interval.lower - inflation)
        x_highs.append(global_corner.x_interval.upper + inflation)
        y_lows.append(global_corner.y_interval.lower - inflation)
        y_highs.append(global_corner.y_interval.upper + inflation)
    return min(x_lows), max(x_highs), min(y_lows), max(y_highs)


def certify_oriented_segment_covers(
    body: VehicleRectangle,
    poses: tuple[Pose2D, ...],
    grid: RectangularGrid,
    *,
    pose_error: float = 0.0,
    certification_support: RectangularGrid | None = None,
) -> tuple[OrientedSegmentCover, ...]:
    """Build continuous segment covers and reject any cover leaving the fixed ROI.

    ``grid`` defines the witness-cell partition. ``certification_support`` may
    separately freeze the physical ROI when diagnostic grid phases are shifted.
    """

    if not poses:
        raise ValueError("pose sequence must be non-empty")
    pairs = tuple(zip(poses, poses[1:], strict=False)) or ((poses[0], poses[0]),)
    covers = tuple(
        _segment_cover(body, start, end, pose_error=pose_error) for start, end in pairs
    )
    support = grid if certification_support is None else certification_support
    if any(not _body_cover_inside_grid(body, cover, support) for cover in covers):
        raise ValueError("oriented swept cover leaves the certified grid ROI")
    return covers


def certify_oriented_swept_outer_cover(
    body: VehicleRectangle,
    poses: tuple[Pose2D, ...],
    grid: RectangularGrid,
    *,
    pose_error: float = 0.0,
    certification_support: RectangularGrid | None = None,
) -> OrientedSweptCoverCertificate:
    """Certify an h-cover of a piecewise-linear oriented rectangle sweep."""

    covers = certify_oriented_segment_covers(
        body,
        poses,
        grid,
        pose_error=pose_error,
        certification_support=certification_support,
    )
    centers = grid.centers()
    selected = np.zeros(centers.shape[0], dtype=bool)
    h = grid.cell_radius
    for cover in covers:
        threshold_squared = (
            _fraction(cover.inflation_m) + _fraction(h)
        ) ** 2
        x_min, x_max, y_min, y_max = _cover_global_bounds(
            body, cover, extra_radius=h
        )
        for index, center in enumerate(centers):
            center_x = _fraction(float(center[0]))
            center_y = _fraction(float(center[1]))
            if not (x_min <= center_x <= x_max and y_min <= center_y <= y_max):
                continue
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
    selected: set[int] = set()
    column_bounds = [
        (
            _fraction(grid.x_min) + column * _fraction(grid.resolution),
            _fraction(grid.x_min) + (column + 1) * _fraction(grid.resolution),
        )
        for column in range(grid.columns)
    ]
    row_bounds = [
        (
            _fraction(grid.y_min) + row * _fraction(grid.resolution),
            _fraction(grid.y_min) + (row + 1) * _fraction(grid.resolution),
        )
        for row in range(grid.rows)
    ]
    for pose in poses:
        nominal_cover = OrientedSegmentCover(pose, 0.0, 0.0, 0.0, 0.0, 0.0)
        body_x_min, body_x_max, body_y_min, body_y_max = _cover_global_bounds(
            body, nominal_cover
        )
        candidate_columns = [
            column
            for column, (cell_low, cell_high) in enumerate(column_bounds)
            if cell_high >= body_x_min and cell_low <= body_x_max
        ]
        candidate_rows = [
            row
            for row, (cell_low, cell_high) in enumerate(row_bounds)
            if cell_high >= body_y_min and cell_low <= body_y_max
        ]
        for row in candidate_rows:
            for column in candidate_columns:
                index = row * grid.columns + column
                if index in selected:
                    continue
                corners = _cell_corners(grid, index)
                intervals = [certify_global_to_ego(corner, pose) for corner in corners]
                if all(
                    local.x_interval.lower >= x_low
                    and local.x_interval.upper <= x_high
                    and local.y_interval.lower >= y_low
                    and local.y_interval.upper <= y_high
                    for local in intervals
                ):
                    selected.add(index)
    return OrientedInnerCellCertificate(tuple(sorted(selected)))


def _cells_disjoint_from_oriented_cover(
    cells: tuple[int, ...],
    covers: tuple[OrientedSegmentCover, ...],
    body: VehicleRectangle,
    grid: RectangularGrid,
) -> tuple[int, ...]:
    centers = grid.centers()
    bounded_covers = tuple(
        (
            segment,
            _cover_global_bounds(body, segment, extra_radius=grid.cell_radius),
        )
        for segment in covers
    )
    witnesses: list[int] = []
    for index in cells:
        center = (float(centers[index, 0]), float(centers[index, 1]))
        cell_x = _fraction(center[0])
        cell_y = _fraction(center[1])
        disjoint = True
        for segment, (x_min, x_max, y_min, y_max) in bounded_covers:
            if not (x_min <= cell_x <= x_max and y_min <= cell_y <= y_max):
                continue
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
    certification_support: RectangularGrid | None = None,
) -> OrientedBidirectionalNonInclusion:
    """Lower-bound both continuous set differences with oriented certificates."""

    challenger_outer = certify_oriented_segment_covers(
        body,
        challenger_poses,
        grid,
        pose_error=pose_error,
        certification_support=certification_support,
    )
    native_outer = certify_oriented_segment_covers(
        body,
        native_poses,
        grid,
        pose_error=pose_error,
        certification_support=certification_support,
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
