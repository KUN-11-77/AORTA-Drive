"""Build exact forward96 teacher artifacts for the 672 frozen H7BS4A fit anchors."""

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
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import RectangularGrid  # type: ignore[import-untyped]
from build_navsim_node_labels import (
    CacheLabelRecord,
    SELECTED_LAYER_NAMES,
    _build_cache_labels_worker,
    _sha256,
)

ARRAY_NAMES = ("label_m", "transform_error_m", "distance_error_m", "total_error_m")


def forward96_grid() -> RectangularGrid:
    return RectangularGrid(x_min=-32.0, y_min=-32.0, columns=96, rows=64, resolution=1.0)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def select_fit_records(
    anchor_payload: Mapping[str, Any], metric_payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Join the frozen H7 anchors to R004E caches by exact log/token identity."""

    anchors = cast(list[dict[str, Any]], anchor_payload.get("anchor_ledgers"))
    metrics = cast(list[dict[str, Any]], metric_payload.get("records"))
    if len(anchors) != 672 or len(metrics) != 1440:
        raise ValueError("unexpected H7 or R004E record count")
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for record in metrics:
        key = (str(record["log_name"]), str(record["scene_token"]))
        if key in lookup:
            raise ValueError(f"duplicate R004E key: {key}")
        lookup[key] = record
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for anchor in anchors:
        key = (str(anchor["log_name"]), str(anchor["scene_token"]))
        if key in seen:
            raise ValueError(f"duplicate H7 anchor key: {key}")
        seen.add(key)
        if key not in lookup:
            raise ValueError(f"H7 anchor missing from R004E: {key}")
        record = lookup[key]
        if record.get("split") != "train":
            raise ValueError(f"non-train record in fit selection: {key}")
        if record.get("recording_group") != anchor.get("recording_group"):
            raise ValueError(f"recording-group mismatch: {key}")
        selected.append(record)
    if len({str(record["recording_group"]) for record in selected}) != 42:
        raise ValueError("fit selection does not contain 42 recording groups")
    return selected


def _prefix_is_exact(new_path: Path, old_path: Path) -> bool:
    with np.load(new_path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
        if set(new.files) != set(ARRAY_NAMES) or set(old.files) != set(ARRAY_NAMES):
            return False
        return all(
            new[name].shape == (64, 96)
            and old[name].shape == (64, 64)
            and np.array_equal(new[name][:, :64], old[name])
            for name in ARRAY_NAMES
        )


def build(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    contract = cast(dict[str, Any], config["H7BS4B0A2"])
    frozen_inputs = cast(Mapping[str, str], contract["frozen_input_sha256"])
    for name, path in {
        "h7bs4a_result": args.anchor_ledger,
        "r004e_metric_manifest": args.metric_manifest,
        "r004e_label_manifest": args.label_manifest,
    }.items():
        if _sha256(path) != frozen_inputs[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")

    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    anchor_payload = _read_json(args.anchor_ledger)
    metric_payload = _read_json(args.metric_manifest)
    label_payload = _read_json(args.label_manifest)
    selected = select_fit_records(anchor_payload, metric_payload)
    old_records = {
        str(record["cache_id"]): record
        for record in cast(list[dict[str, Any]], label_payload["records"])
    }
    if len(old_records) != 1440:
        raise ValueError("R004E label manifest is incomplete")

    cache_paths: list[Path] = []
    old_label_paths: dict[str, Path] = {}
    for record in selected:
        cache_id = f"{record['log_name']}/{record['scene_token']}"
        cache_path = args.r004e_root / str(record["relative_cache_path"])
        if not cache_path.is_file() or _sha256(cache_path) != record["cache_sha256"]:
            raise ValueError(f"R004E cache integrity failure: {cache_id}")
        old_record = old_records.get(cache_id)
        if old_record is None or old_record["cache_sha256"] != record["cache_sha256"]:
            raise ValueError(f"R004E label/cache join failure: {cache_id}")
        old_path = args.old_label_root / "labels" / str(old_record["artifact_name"])
        if not old_path.is_file() or _sha256(old_path) != old_record["artifact_sha256"]:
            raise ValueError(f"old label artifact integrity failure: {cache_id}")
        cache_paths.append(cache_path)
        old_label_paths[cache_id] = old_path

    repository = Path(__file__).resolve().parents[1]
    navsim_root = Path(os.environ["AORTA_NAVSIM_SOURCE_ROOT"]).resolve()
    grid = forward96_grid()
    source_paths = {
        "teacher": repository / "src/aorta_drive/certification/teacher.py",
        "transform": repository / "src/aorta_drive/certification/transform.py",
        "geometry": repository / "src/aorta_drive/certification/geometry.py",
        "base_builder": repository / "scripts/build_navsim_node_labels.py",
        "runner": Path(__file__).resolve(),
        "config": args.config.resolve(),
        "navsim_metric_cache": navsim_root / "planning/metric_caching/metric_cache.py",
        "navsim_cache_processor": navsim_root / "planning/metric_caching/metric_cache_processor.py",
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
        "selection": "exact H7BS4A log_name plus scene_token join to R004E",
        "source_hashes": source_hashes,
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    records: list[CacheLabelRecord] = []
    failures: list[str] = []
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _build_cache_labels_worker, path, output, grid, navsim_root.parent
            ): path
            for path in cache_paths
        }
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            cache_id = f"{path.parents[2].name}/{path.parent.name}"
            try:
                label_record = future.result()
                new_path = output / "labels" / label_record.artifact_name
                if not _prefix_is_exact(new_path, old_label_paths[cache_id]):
                    raise ValueError("forward96 prefix differs from frozen 64x64 artifact")
                records.append(label_record)
                print(f"completed {len(records)}/{len(cache_paths)}: {cache_id}", flush=True)
            except Exception as error:
                failures.append(f"{cache_id}: {type(error).__name__}: {error}")
                print(f"failed {cache_id}: {error}", flush=True)
    records.sort(key=lambda record: record.cache_id)
    record_payload = [asdict(record) for record in records]
    anchor_order = [f"{record['log_name']}/{record['scene_token']}" for record in selected]
    summary = {
        "experiment_id": "R004H7BS4B0A2-fit-forward96-exact-teacher-v1",
        "code_commit": args.code_commit,
        "status": "completed" if not failures else "failed",
        "scope": "672 frozen fit-anchor teacher artifacts only; no model or decision evidence",
        "anchor_count": len(selected),
        "recording_group_count": len({record["recording_group"] for record in selected}),
        "cache_pass_count": len(records),
        "cache_failure_count": len(failures),
        "node_count_per_cache": 6144,
        "total_node_labels": sum(record.node_count for record in records),
        "prefix_exact_count": len(records),
        "protocol_sha256": protocol_hash,
        "logical_manifest_sha256": hashlib.sha256(
            json.dumps(record_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "duration_seconds": time.perf_counter() - started,
        "protocol": protocol,
        "frozen_input_sha256": dict(frozen_inputs),
        "anchor_order_cache_ids": anchor_order,
        "failures": failures,
        "records": record_payload,
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--anchor-ledger", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--r004e-root", type=Path, required=True)
    parser.add_argument("--old-label-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"protocol", "records", "anchor_order_cache_ids"}
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    raise SystemExit(int(result["status"] != "completed"))
