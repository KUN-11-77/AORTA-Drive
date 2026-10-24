"""Independent high-resolution polygon diagnostics for oriented vehicle sweeps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from shapely.geometry import Polygon  # type: ignore[import-untyped]
from shapely.ops import unary_union  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import Pose2D, VehicleRectangle


@dataclass(frozen=True)
class PolygonReferenceLevel:
    subdivisions_per_interval: int
    challenger_minus_native_area_m2: float
    native_minus_challenger_area_m2: float


def _interpolated_poses(
    poses: Sequence[Pose2D], subdivisions: int
) -> tuple[Pose2D, ...]:
    if not poses:
        raise ValueError("pose sequence must be non-empty")
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")
    if len(poses) == 1:
        return (poses[0],)
    interpolated: list[Pose2D] = []
    for start, end in zip(poses, poses[1:], strict=False):
        delta_yaw = end.yaw - start.yaw
        if abs(delta_yaw) > math.pi:
            raise ValueError("adjacent yaws must use the frozen continuous unwrap")
        for step in range(subdivisions):
            fraction = step / subdivisions
            interpolated.append(
                Pose2D(
                    start.x + fraction * (end.x - start.x),
                    start.y + fraction * (end.y - start.y),
                    start.yaw + fraction * delta_yaw,
                )
            )
    interpolated.append(poses[-1])
    return tuple(interpolated)


def _body_polygon(body: VehicleRectangle, pose: Pose2D) -> Polygon:
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    corners = []
    for local_x, local_y in (
        (-body.rear_overhang, -body.half_width),
        (body.front_extent, -body.half_width),
        (body.front_extent, body.half_width),
        (-body.rear_overhang, body.half_width),
    ):
        corners.append(
            (
                pose.x + cosine * local_x - sine * local_y,
                pose.y + sine * local_x + cosine * local_y,
            )
        )
    polygon = Polygon(corners)
    if polygon.is_empty or not polygon.is_valid or not math.isfinite(polygon.area):
        raise ValueError("diagnostic body polygon is invalid")
    return polygon


def polygon_sweep_reference(
    body: VehicleRectangle,
    challenger_poses: Sequence[Pose2D],
    native_poses: Sequence[Pose2D],
    subdivisions: int,
) -> PolygonReferenceLevel:
    """Return sampled-polygon set differences for one dyadic mesh level.

    This is a convergence diagnostic based on ordinary double-precision GEOS,
    not a proof-carrying exact ground truth.
    """

    challenger = unary_union(
        [
            _body_polygon(body, pose)
            for pose in _interpolated_poses(challenger_poses, subdivisions)
        ]
    )
    native = unary_union(
        [
            _body_polygon(body, pose)
            for pose in _interpolated_poses(native_poses, subdivisions)
        ]
    )
    values = (
        float(challenger.difference(native).area),
        float(native.difference(challenger).area),
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("polygon reference returned an invalid area")
    return PolygonReferenceLevel(subdivisions, *values)

