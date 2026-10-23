"""Audit exact R004E metric-cache to node-label alignment and artifact integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aorta_drive.protocol.training_join import (
    load_training_anchors,
    select_preflight_anchors,
)

REQUIRED_ARRAYS = (
    "distance_error_m",
    "label_m",
    "total_error_m",
    "transform_error_m",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--label-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--max-anchors", type=int)
    parser.add_argument("--max-recording-groups", type=int)
    args = parser.parse_args()

    anchors = select_preflight_anchors(
        load_training_anchors(args.time_set),
        maximum_anchors=args.max_anchors,
        maximum_recording_groups=args.max_recording_groups,
    )
    expected = {f"{anchor.log_name}/{anchor.scene_token}": anchor for anchor in anchors}
    metric = json.loads(args.metric_manifest.read_text(encoding="utf-8"))
    label_manifest_path = args.label_directory / "manifest.json"
    labels = json.loads(label_manifest_path.read_text(encoding="utf-8"))

    metric_records = metric.get("records", [])
    label_records = labels.get("records", [])
    metric_ids = [f"{record['log_name']}/{record['scene_token']}" for record in metric_records]
    label_ids = [str(record["cache_id"]) for record in label_records]
    duplicate_metric_ids = sum(count - 1 for count in Counter(metric_ids).values() if count > 1)
    duplicate_label_ids = sum(count - 1 for count in Counter(label_ids).values() if count > 1)
    metric_by_id = dict(zip(metric_ids, metric_records, strict=True))
    label_by_id = dict(zip(label_ids, label_records, strict=True))
    expected_ids = set(expected)
    metric_id_set = set(metric_ids)
    label_id_set = set(label_ids)

    failures: list[str] = []
    if metric_id_set != expected_ids:
        failures.append("metric manifest does not exactly match selected frozen anchors")
    if label_id_set != expected_ids:
        failures.append("label manifest does not exactly match selected frozen anchors")
    if duplicate_metric_ids or duplicate_label_ids:
        failures.append("duplicate metric or label IDs")
    group_counts = Counter(record["recording_group"] for record in metric_records)
    if args.max_anchors is None and any(count != 16 for count in group_counts.values()):
        failures.append("a complete-group audit contains a group without exactly 16 anchors")
    if set(metric.get("allowed_splits", [])) - {"train", "development"}:
        failures.append("metric manifest contains a prohibited split")

    checked_artifacts = 0
    cache_hash_mismatches = 0
    artifact_hash_mismatches = 0
    array_failures = 0
    maximum_total_error_m = 0.0
    for cache_id in sorted(expected_ids & metric_id_set & label_id_set):
        metric_record = metric_by_id[cache_id]
        label_record = label_by_id[cache_id]
        if label_record["cache_sha256"] != metric_record["cache_sha256"]:
            cache_hash_mismatches += 1
            continue
        artifact = args.label_directory / "labels" / label_record["artifact_name"]
        if not artifact.is_file() or _sha256(artifact) != label_record["artifact_sha256"]:
            artifact_hash_mismatches += 1
            continue
        try:
            with np.load(artifact, allow_pickle=False) as arrays:
                if tuple(sorted(arrays.files)) != REQUIRED_ARRAYS:
                    raise ValueError("unexpected arrays")
                for name in REQUIRED_ARRAYS:
                    values = arrays[name]
                    if values.shape != (64, 64) or not np.all(np.isfinite(values)):
                        raise ValueError("invalid array shape or finiteness")
                for name in ("distance_error_m", "total_error_m", "transform_error_m"):
                    if np.any(arrays[name] < 0.0):
                        raise ValueError("negative error ledger")
                maximum_total_error_m = max(
                    maximum_total_error_m, float(np.max(arrays["total_error_m"]))
                )
        except (OSError, ValueError, KeyError):
            array_failures += 1
            continue
        checked_artifacts += 1
    if cache_hash_mismatches:
        failures.append("label provenance cache SHA disagrees with metric manifest")
    if artifact_hash_mismatches:
        failures.append("label artifact is missing or fails SHA-256")
    if array_failures:
        failures.append("label artifact arrays fail shape, finiteness, or ledger checks")
    if checked_artifacts != len(expected_ids):
        failures.append("not every expected label artifact passed independent checks")

    result = {
        "experiment_id": "R004E-A-cache-label-join-audit",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "status": "PASS" if not failures else "FAIL",
        "expected_anchor_count": len(expected_ids),
        "recording_group_count": len(group_counts),
        "metric_record_count": len(metric_records),
        "label_record_count": len(label_records),
        "independently_checked_label_artifacts": checked_artifacts,
        "duplicate_metric_ids": duplicate_metric_ids,
        "duplicate_label_ids": duplicate_label_ids,
        "cache_hash_mismatches": cache_hash_mismatches,
        "artifact_hash_mismatches": artifact_hash_mismatches,
        "array_failures": array_failures,
        "maximum_total_error_m": maximum_total_error_m,
        "calibration_test_or_target_accessed": False,
        "metric_manifest_sha256": _sha256(args.metric_manifest),
        "label_manifest_sha256": _sha256(label_manifest_path),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
