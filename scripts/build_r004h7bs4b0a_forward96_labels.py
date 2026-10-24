"""Build deterministic 96x64 exact-teacher artifacts for H7BS4B0A."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aorta_drive.certification.geometry import RectangularGrid  # type: ignore[import-untyped]
from build_navsim_node_labels import (
    CacheLabelRecord,
    SELECTED_LAYER_NAMES,
    _build_cache_labels_worker,
    _sha256,
)


def forward96_grid() -> RectangularGrid:
    """Return the frozen full-horizon grid qualified by H7BS3R1."""

    return RectangularGrid(
        x_min=-32.0,
        y_min=-32.0,
        columns=96,
        rows=64,
        resolution=1.0,
    )


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    cache_root = Path(os.environ["AORTA_NAVSIM_METRIC_CACHE_ROOT"]).resolve()
    navsim_root = Path(os.environ["AORTA_NAVSIM_SOURCE_ROOT"]).resolve()
    output_directory = arguments.output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    output_directory.mkdir(parents=True)
    repository_root = Path(__file__).resolve().parents[1]
    grid = forward96_grid()
    cache_paths = sorted(cache_root.rglob("metric_cache.pkl"))
    if len(cache_paths) != arguments.expected_cache_count:
        raise ValueError(
            f"cache count {len(cache_paths)} != expected {arguments.expected_cache_count}"
        )
    if arguments.workers <= 0:
        raise ValueError("workers must be positive")

    source_paths = {
        "teacher": repository_root / "src/aorta_drive/certification/teacher.py",
        "transform": repository_root / "src/aorta_drive/certification/transform.py",
        "geometry": repository_root / "src/aorta_drive/certification/geometry.py",
        "base_builder": repository_root / "scripts/build_navsim_node_labels.py",
        "runner": Path(__file__).resolve(),
        "config": arguments.config.resolve(),
        "navsim_metric_cache": navsim_root / "planning/metric_caching/metric_cache.py",
        "navsim_cache_processor": navsim_root
        / "planning/metric_caching/metric_cache_processor.py",
        "navsim_occupancy_map": navsim_root
        / "planning/simulation/planner/pdm_planner/observation/pdm_occupancy_map.py",
    }
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    protocol = {
        "grid": {
            "x_support_m": [-32.0, 64.0],
            "y_support_m": [-32.0, 32.0],
            "resolution_m": 1.0,
            "rows": 64,
            "columns": 96,
            "nodes": 6144,
            "undirected_edges": 12128,
            "cell_radius_m": grid.cell_radius,
        },
        "selected_layers": sorted(SELECTED_LAYER_NAMES),
        "teacher": "raw_polygon_cover",
        "polygon_error_scope": "zero only relative to exact frozen cache object",
        "parent_support_gate": "R004H7BS3R1",
        "source_hashes": source_hashes,
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    started = time.perf_counter()
    records: list[CacheLabelRecord] = []
    failures: list[str] = []
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
            except Exception as error:  # preserve every source failure in the manifest
                failures.append(f"{cache_id}: {type(error).__name__}: {error}")
                print(f"failed {cache_id}: {error}", flush=True)
    records.sort(key=lambda record: record.cache_id)
    record_payload = [asdict(record) for record in records]
    logical_manifest_hash = hashlib.sha256(
        json.dumps(record_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "experiment_id": "R004H7BS4B0A-forward96-exact-teacher-v1",
        "code_commit": arguments.code_commit,
        "status": "completed" if not failures else "failed",
        "scope": "source exact-teacher qualification only; no model or decision evidence",
        "cache_count": len(cache_paths),
        "cache_pass_count": len(records),
        "cache_failure_count": len(failures),
        "node_count_per_cache": 6144,
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
        "duration_seconds": time.perf_counter() - started,
        "protocol": protocol,
        "failures": failures,
        "records": record_payload,
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-cache-count", type=int, default=164)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"protocol", "records"}},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    raise SystemExit(int(result["status"] != "completed"))
