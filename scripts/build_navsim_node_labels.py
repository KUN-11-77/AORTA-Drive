"""Build deterministic R002G raw-cover node-label and error manifests."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import lzma
import os
import pathlib
import pickle
import sys
import time
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from aorta_drive.certification.geometry import Pose2D, RectangularGrid
from aorta_drive.certification.teacher import (
    ExactPolygonTeacher,
    MapErrorLedger,
    PolygonComponent,
)
from aorta_drive.certification.transform import certify_ego_to_global

SELECTED_LAYER_NAMES = {
    "ROADBLOCK",
    "INTERSECTION",
    "DRIVABLE_AREA",
    "CARPARK_AREA",
}
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class CacheLabelRecord:
    cache_id: str
    cache_sha256: str
    artifact_name: str
    artifact_sha256: str
    component_count: int
    segment_count: int
    node_count: int
    inside_node_count: int
    boundary_node_count: int
    outside_node_count: int
    label_min_m: float
    label_max_m: float
    max_transform_error_m: float
    max_distance_error_m: float
    max_total_error_m: float


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


def _ring(coordinates: Iterable[tuple[float, ...]]) -> tuple[tuple[float, float], ...]:
    points = tuple((float(point[0]), float(point[1])) for point in coordinates)
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    return points


def _polygon_component(polygon: Polygon) -> PolygonComponent:
    return PolygonComponent(
        outer=_ring(polygon.exterior.coords),
        holes=tuple(_ring(interior.coords) for interior in polygon.interiors),
    )


def _components(geometries: Iterable[Polygon | MultiPolygon]) -> tuple[PolygonComponent, ...]:
    components: list[PolygonComponent] = []
    for geometry in geometries:
        if isinstance(geometry, Polygon):
            components.append(_polygon_component(geometry))
        elif isinstance(geometry, MultiPolygon):
            components.extend(_polygon_component(polygon) for polygon in geometry.geoms)
        else:
            raise ValueError(f"unsupported selected geometry: {geometry.geom_type}")
    return tuple(components)


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(arrays[name]), compresslevel=9)


def _build_cache_labels(
    path: Path,
    output_directory: Path,
    grid: RectangularGrid,
) -> CacheLabelRecord:
    cache = _load_pickle(path)
    cache_id = f"{path.parents[2].name}/{path.parent.name}"
    selected = [
        geometry
        for geometry, map_type in zip(
            cache.drivable_area_map._geometries,
            cache.drivable_area_map._map_types,
            strict=True,
        )
        if map_type.name in SELECTED_LAYER_NAMES
    ]
    for geometry in selected:
        if geometry.is_empty:
            raise ValueError("selected geometry is empty")
        if not geometry.is_valid:
            raise ValueError("selected geometry is invalid")
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            raise ValueError(f"unsupported selected geometry: {geometry.geom_type}")
    components = _components(selected)
    teacher = ExactPolygonTeacher(components)
    state = cache.ego_state.rear_axle
    pose = Pose2D(float(state.x), float(state.y), float(state.heading))
    centers = grid.centers()
    labels = np.empty(centers.shape[0], dtype=np.float64)
    transform_errors = np.empty_like(labels)
    distance_errors = np.empty_like(labels)
    total_errors = np.empty_like(labels)

    for index, center in enumerate(centers):
        transform = certify_ego_to_global((float(center[0]), float(center[1])), pose)
        label = teacher.certified_node_label((transform.x, transform.y))
        ledger = MapErrorLedger(
            transform_error=transform.error_radius,
            polygon_error=0.0,
            distance_error=label.distance_error,
            rounding_error=0.0,
        )
        labels[index] = label.value
        transform_errors[index] = transform.error_radius
        distance_errors[index] = label.distance_error
        total_errors[index] = ledger.total

    arrays = {
        "distance_error_m": distance_errors.reshape(grid.rows, grid.columns),
        "label_m": labels.reshape(grid.rows, grid.columns),
        "total_error_m": total_errors.reshape(grid.rows, grid.columns),
        "transform_error_m": transform_errors.reshape(grid.rows, grid.columns),
    }
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("non-finite label artifact")
    if (
        np.any(total_errors < 0)
        or np.any(total_errors < transform_errors)
        or np.any(total_errors < distance_errors)
    ):
        raise ValueError("invalid error ledger direction")

    artifact_name = f"{hashlib.sha256(cache_id.encode()).hexdigest()[:20]}.npz"
    artifact_path = output_directory / "labels" / artifact_name
    _write_deterministic_npz(artifact_path, arrays)
    replay_path = output_directory / "labels" / f"{artifact_name}.replay.tmp"
    _write_deterministic_npz(replay_path, arrays)
    artifact_hash = _sha256(artifact_path)
    if _sha256(replay_path) != artifact_hash:
        raise ValueError("label artifact is not byte-reproducible")
    replay_path.unlink()

    return CacheLabelRecord(
        cache_id=cache_id,
        cache_sha256=_sha256(path),
        artifact_name=artifact_name,
        artifact_sha256=artifact_hash,
        component_count=teacher.component_count,
        segment_count=teacher.segment_count,
        node_count=labels.size,
        inside_node_count=int(np.count_nonzero(labels > 0)),
        boundary_node_count=int(np.count_nonzero(labels == 0)),
        outside_node_count=int(np.count_nonzero(labels < 0)),
        label_min_m=float(np.min(labels)),
        label_max_m=float(np.max(labels)),
        max_transform_error_m=float(np.max(transform_errors)),
        max_distance_error_m=float(np.max(distance_errors)),
        max_total_error_m=float(np.max(total_errors)),
    )


def _build_cache_labels_worker(
    path: Path,
    output_directory: Path,
    grid: RectangularGrid,
    navsim_parent: Path,
) -> CacheLabelRecord:
    if str(navsim_parent) not in sys.path:
        sys.path.insert(0, str(navsim_parent))
    return _build_cache_labels(path, output_directory, grid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-cache-count", type=int, default=164)
    parser.add_argument("--max-caches", type=int)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()

    cache_root = Path(os.environ["AORTA_NAVSIM_METRIC_CACHE_ROOT"]).resolve()
    navsim_root = Path(os.environ["AORTA_NAVSIM_SOURCE_ROOT"]).resolve()
    sys.path.insert(0, str(navsim_root.parent))
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[1]
    grid = RectangularGrid(
        x_min=-32.0,
        y_min=-32.0,
        columns=64,
        rows=64,
        resolution=1.0,
    )
    cache_paths = sorted(cache_root.rglob("metric_cache.pkl"))
    failures: list[str] = []
    if len(cache_paths) != arguments.expected_cache_count:
        failures.append(
            f"cache count {len(cache_paths)} != expected {arguments.expected_cache_count}"
        )
    if arguments.max_caches is not None:
        if arguments.max_caches <= 0:
            raise ValueError("max-caches must be positive")
        cache_paths = cache_paths[: arguments.max_caches]
    if arguments.workers <= 0:
        raise ValueError("workers must be positive")

    source_paths = {
        "teacher": repository_root / "src/aorta_drive/certification/teacher.py",
        "transform": repository_root / "src/aorta_drive/certification/transform.py",
        "geometry": repository_root / "src/aorta_drive/certification/geometry.py",
        "config": repository_root / "configs/experiment/r002g_node_label_manifest.yaml",
        "navsim_metric_cache": navsim_root / "planning/metric_caching/metric_cache.py",
        "navsim_cache_processor": navsim_root / "planning/metric_caching/metric_cache_processor.py",
        "navsim_occupancy_map": navsim_root
        / "planning/simulation/planner/pdm_planner/observation/pdm_occupancy_map.py",
    }
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    protocol = {
        "grid": {
            "x_support_m": [-32.0, 32.0],
            "y_support_m": [-32.0, 32.0],
            "resolution_m": 1.0,
            "rows": 64,
            "columns": 64,
            "nodes": 4096,
            "undirected_edges": 8064,
            "cell_radius_m": grid.cell_radius,
        },
        "selected_layers": sorted(SELECTED_LAYER_NAMES),
        "teacher": "raw_polygon_cover",
        "polygon_error_scope": "zero only relative to exact frozen cache object",
        "source_hashes": source_hashes,
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    started = time.perf_counter()
    records: list[CacheLabelRecord] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                _build_cache_labels_worker,
                path,
                output_directory,
                grid,
                navsim_root.parent,
            ): path
            for path in cache_paths
        }
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            cache_id = f"{path.parents[2].name}/{path.parent.name}"
            try:
                records.append(future.result())
                print(f"completed {len(records)}/{len(cache_paths)}: {cache_id}", flush=True)
            except Exception as error:
                failures.append(f"{cache_id}: {type(error).__name__}: {error}")
                print(f"failed {cache_id}: {error}", flush=True)

    records.sort(key=lambda record: record.cache_id)

    record_payload = [asdict(record) for record in records]
    logical_manifest_hash = hashlib.sha256(
        json.dumps(record_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "run_id": "R002G",
        "status": "completed" if not failures else "failed",
        "scope": "frozen-cache node-label ledger; no model or performance evidence",
        "development_limit": arguments.max_caches,
        "cache_count": len(cache_paths),
        "cache_pass_count": len(records),
        "cache_failure_count": len(cache_paths) - len(records),
        "node_count_per_cache": 4096,
        "total_node_labels": sum(record.node_count for record in records),
        "protocol_sha256": protocol_hash,
        "logical_manifest_sha256": logical_manifest_hash,
        "max_transform_error_m": max(
            (record.max_transform_error_m for record in records), default=None
        ),
        "max_distance_error_m": max(
            (record.max_distance_error_m for record in records), default=None
        ),
        "max_total_error_m": max((record.max_total_error_m for record in records), default=None),
        "inside_node_count": sum(record.inside_node_count for record in records),
        "boundary_node_count": sum(record.boundary_node_count for record in records),
        "outside_node_count": sum(record.outside_node_count for record in records),
        "duration_seconds": time.perf_counter() - started,
        "protocol": protocol,
        "failures": failures,
        "records": record_payload,
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in {"protocol", "records"}},
            indent=2,
        )
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
