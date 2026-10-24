"""Compare two full deterministic builds of the H7BS4B0A teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file  # type: ignore[import-untyped]

ARRAY_NAMES = ("label_m", "transform_error_m", "distance_error_m", "total_error_m")


def _logical_hash(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_manifest(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("status") != "completed":
        raise ValueError("teacher manifest is not complete")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("teacher manifest records are missing")
    records = cast(list[dict[str, Any]], raw_records)
    if len(records) != 164 or len({record["cache_id"] for record in records}) != 164:
        raise ValueError("teacher manifest does not contain 164 unique caches")
    if any(int(record["node_count"]) != 6144 for record in records):
        raise ValueError("teacher record node count is not 6144")
    grid = cast(Mapping[str, Any], cast(Mapping[str, Any], payload["protocol"])["grid"])
    expected = {
        "x_support_m": [-32.0, 64.0],
        "y_support_m": [-32.0, 32.0],
        "resolution_m": 1.0,
        "rows": 64,
        "columns": 96,
        "nodes": 6144,
        "undirected_edges": 12128,
    }
    if any(grid.get(key) != value for key, value in expected.items()):
        raise ValueError("teacher grid protocol mismatch")
    if payload.get("logical_manifest_sha256") != _logical_hash(records):
        raise ValueError("logical manifest hash mismatch")
    return records


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    frozen = cast(Mapping[str, str], config["H7BS4B0A"]["implementation_sha256"])
    repository = Path(__file__).resolve().parents[1]
    for name, path in {
        "builder_source": repository / "scripts/build_r004h7bs4b0a_forward96_labels.py",
        "audit_source": Path(__file__).resolve(),
        "teacher_source": repository / "src/aorta_drive/certification/teacher.py",
        "transform_source": repository / "src/aorta_drive/certification/transform.py",
        "geometry_source": repository / "src/aorta_drive/certification/geometry.py",
        "base_builder_source": repository / "scripts/build_navsim_node_labels.py",
    }.items():
        if sha256_file(path) != frozen[name]:
            raise ValueError(f"implementation hash mismatch: {name}")
    primary = cast(dict[str, Any], json.loads(args.primary_manifest.read_text(encoding="utf-8")))
    replay = cast(dict[str, Any], json.loads(args.replay_manifest.read_text(encoding="utf-8")))
    primary_records = _validate_manifest(primary)
    replay_records = _validate_manifest(replay)
    replay_by_id = {str(record["cache_id"]): record for record in replay_records}
    artifact_hash_failures = 0
    array_failures = 0
    for record in primary_records:
        cache_id = str(record["cache_id"])
        other = replay_by_id[cache_id]
        if record != other:
            artifact_hash_failures += 1
            continue
        primary_path = args.primary_root / "labels" / str(record["artifact_name"])
        replay_path = args.replay_root / "labels" / str(other["artifact_name"])
        if (
            sha256_file(primary_path) != record["artifact_sha256"]
            or sha256_file(replay_path) != other["artifact_sha256"]
        ):
            artifact_hash_failures += 1
            continue
        with (
            np.load(primary_path, allow_pickle=False) as left,
            np.load(replay_path, allow_pickle=False) as right,
        ):
            if set(left.files) != set(ARRAY_NAMES) or set(right.files) != set(ARRAY_NAMES):
                array_failures += 1
                continue
            for name in ARRAY_NAMES:
                if (
                    left[name].shape != (64, 96)
                    or not np.array_equal(left[name], right[name])
                    or not np.all(np.isfinite(left[name]))
                ):
                    array_failures += 1
                    break
            if (
                np.any(left["transform_error_m"] < 0)
                or np.any(left["distance_error_m"] < 0)
                or np.any(left["total_error_m"] < left["transform_error_m"])
                or np.any(left["total_error_m"] < left["distance_error_m"])
            ):
                array_failures += 1
    exact = primary_records == replay_records
    passed = exact and artifact_hash_failures == 0 and array_failures == 0
    output = {
        "experiment_id": config["experiment_id"],
        "primary_manifest_sha256": sha256_file(args.primary_manifest),
        "replay_manifest_sha256": sha256_file(args.replay_manifest),
        "record_count": len(primary_records),
        "total_node_count": 164 * 6144,
        "records_exact": exact,
        "artifact_hash_failure_count": artifact_hash_failures,
        "array_failure_count": array_failures,
        "mechanical_audit": "PASS" if passed else "FAIL",
        "external_scientific_review": "REVIEW_UNAVAILABLE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(int(result["mechanical_audit"] != "PASS"))
