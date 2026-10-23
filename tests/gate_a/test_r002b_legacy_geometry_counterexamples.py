"""R002B: constructive counterexamples to legacy geometry shortcuts."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class Rectangle:
    """Minimal closed rectangle used only to construct geometry counterexamples."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, point: tuple[float, float]) -> bool:
        return self.x_min <= point[0] <= self.x_max and self.y_min <= point[1] <= self.y_max


def _inside_union(point: tuple[float, float], regions: list[Rectangle]) -> bool:
    return any(region.contains(point) for region in regions)


def _footprint_corners(center_x: float) -> list[tuple[float, float]]:
    return [(center_x + x, y) for x, y in itertools.product((-1.0, 1.0), (-0.5, 0.5))]


def test_four_corners_do_not_certify_the_full_footprint() -> None:
    corners = _footprint_corners(center_x=0.0)
    drivable_islands = [Rectangle(x - 0.1, x + 0.1, y - 0.1, y + 0.1) for x, y in corners]
    legacy_corner_accepts = all(_inside_union(corner, drivable_islands) for corner in corners)
    footprint_center_is_drivable = _inside_union((0.0, 0.0), drivable_islands)
    assert legacy_corner_accepts
    assert not footprint_center_is_drivable


def test_discrete_endpoint_poses_do_not_certify_the_continuous_sweep() -> None:
    left = Rectangle(-3.1, -0.9, -0.6, 0.6)
    right = Rectangle(0.9, 3.1, -0.6, 0.6)
    drivable_regions = [left, right]
    endpoint_corners = [*_footprint_corners(-2.0), *_footprint_corners(2.0)]
    legacy_endpoint_accepts = all(
        _inside_union(corner, drivable_regions) for corner in endpoint_corners
    )
    interpolated_corners = _footprint_corners(-0.5)
    interpolated_accepts = all(
        _inside_union(corner, drivable_regions) for corner in interpolated_corners
    )
    assert legacy_endpoint_accepts
    assert not interpolated_accepts


def test_raster_edt_is_not_an_exact_continuous_boundary_distance() -> None:
    resolution = 1.0
    road_mask = np.zeros((3, 4), dtype=bool)
    road_mask[:, 2:] = True
    raster_inside_distance = distance_transform_edt(road_mask) * resolution
    first_inside_center_distance = float(raster_inside_distance[1, 2])
    true_boundary_distance = 0.5 * resolution
    assert first_inside_center_distance == resolution
    assert first_inside_center_distance - true_boundary_distance == 0.5 * resolution
