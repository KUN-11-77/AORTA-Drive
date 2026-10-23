"""Read-only R002E audit of cached NAVSIM map and vehicle preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import lzma
import math
import os
import pathlib
import pickle
import platform
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_FILES = {
    "navsim.metric_cache": "planning/metric_caching/metric_cache.py",
    "navsim.pdm_occupancy_map": (
        "planning/simulation/planner/pdm_planner/observation/pdm_occupancy_map.py"
    ),
    "navsim.pdm_array_representation": (
        "planning/simulation/planner/pdm_planner/utils/pdm_array_representation.py"
    ),
    "navsim.pdm_scorer": "planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
    "navsim.pdm_geometry_utils": (
        "planning/simulation/planner/pdm_planner/utils/pdm_geometry_utils.py"
    ),
    "navsim.evaluate_pdm_score": "evaluate/pdm_score.py",
    "nuplan.vehicle_parameters": "common/actor_state/vehicle_parameters.py",
    "nuplan.geometry_convert": "common/geometry/convert.py",
}


@dataclass(frozen=True)
class CacheAudit:
    cache_id: str
    sha256: str
    map_name: str
    map_version: str
    polygon_count: int
    selected_polygon_count: int
    raw_area_sum_m2: float
    union_area_m2: float
    overlap_excess_m2: float
    corner_max_abs_error_m: float
    corner_compatibility_budget_m: float
    corner_error_to_budget_ratio: float
    vehicle_signature: tuple[float | str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pickle(path: Path) -> Any:
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    return pickle.loads(lzma.decompress(path.read_bytes()))


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _vehicle_signature(parameters: Any) -> tuple[float | str, ...]:
    return (
        str(parameters.vehicle_name),
        str(parameters.vehicle_type),
        float(parameters.width),
        float(parameters.front_length),
        float(parameters.rear_length),
        float(parameters.rear_axle_to_center),
        float(parameters.wheel_base),
    )


def _expected_corners(state: Any, parameters: Any) -> np.ndarray:
    pose = state.rear_axle
    cosine = math.cos(float(pose.heading))
    sine = math.sin(float(pose.heading))
    local = np.array(
        [
            [parameters.front_length, parameters.half_width],
            [-parameters.rear_length, parameters.half_width],
            [-parameters.rear_length, -parameters.half_width],
            [parameters.front_length, -parameters.half_width],
        ],
        dtype=np.float64,
    )
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return local @ rotation.T + np.array([pose.x, pose.y], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cache-count", type=int, default=164)
    arguments = parser.parse_args()

    cache_root = Path(os.environ["AORTA_NAVSIM_METRIC_CACHE_ROOT"]).resolve()
    navsim_root = Path(os.environ["AORTA_NAVSIM_SOURCE_ROOT"]).resolve()
    nuplan_root = Path(os.environ["AORTA_NUPLAN_SOURCE_ROOT"]).resolve()
    sys.path.insert(0, str(navsim_root.parent))

    import shapely
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        state_array_to_coords_array,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    source_hashes: dict[str, str] = {}
    source_failures: list[str] = []
    for logical_name, relative_path in SOURCE_FILES.items():
        root = navsim_root if logical_name.startswith("navsim.") else nuplan_root
        source_path = root / relative_path
        if not source_path.is_file():
            source_failures.append(f"missing source: {logical_name}")
            continue
        source_hashes[logical_name] = _sha256(source_path)

    cache_paths = sorted(cache_root.rglob("metric_cache.pkl"))
    failures = list(source_failures)
    if len(cache_paths) != arguments.expected_cache_count:
        failures.append(
            f"cache count {len(cache_paths)} != expected {arguments.expected_cache_count}"
        )

    expected_vehicle = ("pacifica", "gen1", 2.297, 4.049, 1.127, 1.461, 3.089)
    official_layers = {
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.DRIVABLE_AREA,
        SemanticMapLayer.CARPARK_AREA,
    }
    audits: list[CacheAudit] = []
    geometry_types: Counter[str] = Counter()
    map_layer_types: Counter[str] = Counter()
    map_names: Counter[str] = Counter()
    map_versions: Counter[str] = Counter()
    overlap_scenes = 0
    max_corner_error = 0.0
    max_corner_budget = 0.0
    max_corner_ratio = 0.0
    max_overlap = 0.0

    for path in cache_paths:
        cache_id = f"{path.parents[2].name}/{path.parent.name}"
        try:
            cache = _load_pickle(path)
            drivable_map = cache.drivable_area_map
            geometries = list(drivable_map._geometries)
            map_types = list(drivable_map._map_types)
            if len(geometries) != len(map_types):
                raise ValueError("geometry/type count mismatch")
            map_layer_types.update(map_type.name for map_type in map_types)

            selected = [
                geometry
                for geometry, map_type in zip(geometries, map_types, strict=True)
                if map_type in official_layers
            ]
            if not selected:
                raise ValueError("no official drivable polygons")
            for geometry in selected:
                geometry_types[geometry.geom_type] += 1
                coordinates = shapely.get_coordinates(geometry)
                if geometry.is_empty:
                    raise ValueError("empty selected geometry")
                if not geometry.is_valid:
                    raise ValueError("invalid selected geometry")
                if coordinates.size == 0 or not np.all(np.isfinite(coordinates)):
                    raise ValueError("non-finite or absent polygon coordinates")
                if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                    raise ValueError(f"unsupported geometry type: {geometry.geom_type}")

            union = shapely.union_all(np.asarray(selected, dtype=object))
            if union.is_empty or not union.is_valid:
                raise ValueError("canonical union is empty or invalid")
            raw_area = math.fsum(float(geometry.area) for geometry in selected)
            union_area = float(union.area)
            overlap = max(0.0, raw_area - union_area)
            overlap_scenes += int(overlap > 1e-9)
            max_overlap = max(max_overlap, overlap)

            parameters = cache.ego_state.car_footprint.vehicle_parameters
            signature = _vehicle_signature(parameters)
            if signature != expected_vehicle:
                raise ValueError(f"unexpected vehicle signature: {signature}")

            state_array = np.zeros((1, 1, StateIndex.size()), dtype=np.float64)
            state_array[0, 0, StateIndex.STATE_SE2] = cache.ego_state.rear_axle.serialize()
            official_corners = state_array_to_coords_array(state_array, parameters)[0, 0, :4]
            expected_corners = _expected_corners(cache.ego_state, parameters)
            corner_error = float(np.max(np.abs(official_corners - expected_corners)))
            coordinate_scale = max(
                1.0,
                abs(float(cache.ego_state.rear_axle.x)),
                abs(float(cache.ego_state.rear_axle.y)),
            )
            corner_budget = 32.0 * math.ulp(coordinate_scale)
            corner_ratio = corner_error / corner_budget
            max_corner_error = max(max_corner_error, corner_error)
            max_corner_budget = max(max_corner_budget, corner_budget)
            max_corner_ratio = max(max_corner_ratio, corner_ratio)
            if corner_error > corner_budget:
                raise ValueError(
                    "vehicle corner convention differs by more than the frozen "
                    f"32-ULP compatibility budget: {corner_error} > {corner_budget}"
                )

            map_name = str(cache.map_parameters.map_name)
            map_version = str(cache.map_parameters.map_version)
            map_names[map_name] += 1
            map_versions[map_version] += 1
            audits.append(
                CacheAudit(
                    cache_id=cache_id,
                    sha256=_sha256(path),
                    map_name=map_name,
                    map_version=map_version,
                    polygon_count=len(geometries),
                    selected_polygon_count=len(selected),
                    raw_area_sum_m2=raw_area,
                    union_area_m2=union_area,
                    overlap_excess_m2=overlap,
                    corner_max_abs_error_m=corner_error,
                    corner_compatibility_budget_m=corner_budget,
                    corner_error_to_budget_ratio=corner_ratio,
                    vehicle_signature=signature,
                )
            )
        except Exception as error:
            failures.append(f"{cache_id}: {type(error).__name__}: {error}")

    protocol_payload = {
        "source_hashes": source_hashes,
        "official_layers": sorted(layer.name for layer in official_layers),
        "teacher": "raw_polygon_cover_clearance",
        "vehicle": expected_vehicle,
        "geometry": "rotation_free_rear_axle_disk",
        "corner_compatibility_max_ulp": 32,
        "corner_coordinates_used_by_certificate": False,
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_manifest_hash = hashlib.sha256(
        json.dumps(
            [(audit.cache_id, audit.sha256) for audit in audits],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    summary = {
        "run_id": "R002E",
        "status": "completed" if not failures else "failed",
        "scope": "read-only NAVSIM preprocessing interface audit; no performance evidence",
        "cache_count": len(cache_paths),
        "cache_pass_count": len(audits),
        "cache_failure_count": len(cache_paths) - len(audits),
        "source_manifest_count": len(source_hashes),
        "source_manifest_expected": len(SOURCE_FILES),
        "protocol_sha256": protocol_hash,
        "cache_manifest_sha256": cache_manifest_hash,
        "map_names": dict(map_names),
        "map_versions": dict(map_versions),
        "geometry_types": dict(geometry_types),
        "map_layer_types": dict(map_layer_types),
        "overlap_scene_count": overlap_scenes,
        "max_overlap_excess_m2": max_overlap,
        "max_corner_abs_error_m": max_corner_error,
        "max_corner_compatibility_budget_m": max_corner_budget,
        "max_corner_error_to_budget_ratio": max_corner_ratio,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "shapely": shapely.__version__,
            "nuplan_devkit": _package_version("nuplan-devkit"),
        },
        "source_hashes": source_hashes,
        "failures": failures,
        "records": [asdict(audit) for audit in audits],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
