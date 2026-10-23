"""Canonical fingerprint of the metric-cache fields used by the road teacher."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import os
import pathlib
import pickle
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon, Polygon  # type: ignore[import-untyped]

SELECTED_LAYER_NAMES = frozenset(
    {"ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA"}
)
def _float_bits(value: float | int) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("teacher inputs must be finite")
    return struct.pack(">d", number).hex()


def _least_rotation(
    values: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Return the lexicographically least cyclic rotation using Booth's algorithm."""

    if not values:
        raise ValueError("polygon ring must be nonempty")
    doubled = tuple(values) + tuple(values)
    first = 0
    second = 1
    offset = 0
    size = len(values)
    while first < size and second < size and offset < size:
        left = doubled[first + offset]
        right = doubled[second + offset]
        if left == right:
            offset += 1
            continue
        if left > right:
            first = first + offset + 1
            if first <= second:
                first = second + 1
        else:
            second = second + offset + 1
            if second <= first:
                second = first + 1
        offset = 0
    start = min(first, second)
    return doubled[start : start + size]


def _canonical_ring(
    coordinates: Iterable[Sequence[float]],
) -> tuple[tuple[str, str], ...]:
    points = tuple(
        (_float_bits(point[0]), _float_bits(point[1])) for point in coordinates
    )
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("polygon ring must contain at least three distinct positions")
    forward = _least_rotation(points)
    reverse = _least_rotation(tuple(reversed(points)))
    return min(forward, reverse)


def _canonical_polygon(polygon: Polygon) -> dict[str, object]:
    holes = sorted(_canonical_ring(interior.coords) for interior in polygon.interiors)
    return {
        "outer": _canonical_ring(polygon.exterior.coords),
        "holes": holes,
    }


def teacher_input_payload(cache: Any) -> dict[str, object]:
    """Return only the exact fields consumed by the frozen R002G teacher."""

    state = cache.ego_state.rear_axle
    components: list[dict[str, object]] = []
    for geometry, map_type in zip(
        cache.drivable_area_map._geometries,
        cache.drivable_area_map._map_types,
        strict=True,
    ):
        layer = str(map_type.name)
        if layer not in SELECTED_LAYER_NAMES:
            continue
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("selected teacher geometry must be nonempty and valid")
        if isinstance(geometry, Polygon):
            polygons = (geometry,)
        elif isinstance(geometry, MultiPolygon):
            polygons = tuple(geometry.geoms)
        else:
            raise ValueError(f"unsupported selected geometry: {geometry.geom_type}")
        for polygon in polygons:
            components.append({"layer": layer, "polygon": _canonical_polygon(polygon)})
    components.sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    if not components:
        raise ValueError("metric cache contains no selected teacher geometry")
    return {
        "ego_rear_axle_binary64": {
            "x": _float_bits(state.x),
            "y": _float_bits(state.y),
            "heading": _float_bits(state.heading),
        },
        "selected_layers": sorted(SELECTED_LAYER_NAMES),
        "components": components,
    }


def teacher_input_sha256(cache: Any) -> str:
    payload = teacher_input_payload(cache)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def fingerprint_metric_cache(path: Path) -> str:
    """Load a trusted official NAVSIM cache and hash its teacher-relevant semantics."""

    if os.name == "nt":
        setattr(pathlib, "PosixPath", pathlib.WindowsPath)
    cache = pickle.loads(lzma.decompress(path.read_bytes()))  # noqa: S301
    return teacher_input_sha256(cache)
