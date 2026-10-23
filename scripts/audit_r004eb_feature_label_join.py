"""Independently audit the frozen R004E-B feature/projection/label join."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from aorta_drive.protocol.training_join import (
    TrainingAnchor,
    load_training_anchors,
    select_preflight_anchors,
)
from aorta_drive.vision.feature_cache import (
    bfloat16_bits_to_float32,
    bilinear_sample_dense_numpy,
    load_dense_bfloat16_bits,
    sha256_file,
    validate_projection_arrays,
)

LABEL_MANIFEST_SHA256 = "1ad3283ff42133c3c04d848b54514a29d5983c2b57f4f7fca38bd69fa781c537"
MAX_ARTIFACT_BYTES = 40 * 1024**3


def _key(log_name: str, scene_token: str) -> tuple[str, str]:
    if not log_name or not scene_token:
        raise ValueError("join keys must be nonempty")
    return log_name, scene_token


def _label_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if sha256_file(path) != LABEL_MANIFEST_SHA256:
        raise RuntimeError("frozen R004E-A label manifest SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("label manifest must contain a records list")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, dict) or not isinstance(raw.get("cache_id"), str):
            raise ValueError("invalid label record")
        parts = raw["cache_id"].split("/")
        if len(parts) != 2:
            raise ValueError("label cache_id must have log_name/scene_token form")
        record_key = _key(parts[0], parts[1])
        if record_key in result:
            raise ValueError(f"duplicate label key: {record_key}")
        result[record_key] = raw
    return result


def _feature_records(
    payload: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    if payload.get("status") != "PASS":
        raise ValueError("feature manifest must have PASS status")
    if payload.get("calibration_test_or_target_accessed") is not False:
        raise ValueError("feature manifest does not preserve the source-only boundary")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("feature manifest must contain a records list")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("invalid feature record")
        record_key = _key(str(raw.get("log_name", "")), str(raw.get("scene_token", "")))
        if record_key in result:
            raise ValueError(f"duplicate feature key: {record_key}")
        result[record_key] = raw
    artifact_bytes = int(payload.get("artifact_bytes", -1))
    if artifact_bytes < 0 or artifact_bytes > MAX_ARTIFACT_BYTES:
        raise ValueError(f"feature artifact budget violated: {artifact_bytes}")
    return result, artifact_bytes


def _torch_sample(
    dense: npt.NDArray[np.float32],
    patch_xy: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float32]:
    import torch
    import torch.nn.functional as functional

    slot_count, channels, height, width = dense.shape
    output = torch.zeros(
        (patch_xy.shape[0], slot_count, channels), dtype=torch.float32
    )
    feature_tensor = torch.from_numpy(np.ascontiguousarray(dense))
    for slot in range(slot_count):
        coordinates = torch.from_numpy(
            np.ascontiguousarray(patch_xy[:, slot], dtype=np.float32)
        )
        normalized = torch.empty_like(coordinates)
        normalized[:, 0] = 2.0 * coordinates[:, 0] / (width - 1) - 1.0
        normalized[:, 1] = 2.0 * coordinates[:, 1] / (height - 1) - 1.0
        grid = normalized.view(1, -1, 1, 2)
        sampled = functional.grid_sample(
            feature_tensor[slot].unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, :, :, 0].T
        output[:, slot] = sampled
    output = torch.where(torch.from_numpy(mask).unsqueeze(-1), output, 0.0)
    return output.numpy()


def audit(args: argparse.Namespace) -> dict[str, object]:
    anchors = select_preflight_anchors(
        load_training_anchors(args.time_set),
        maximum_anchors=args.max_anchors,
        maximum_recording_groups=args.max_recording_groups,
    )
    expected = {_key(anchor.log_name, anchor.scene_token): anchor for anchor in anchors}
    label_records = _label_records(args.label_manifest)
    feature_payload = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    feature_records, artifact_bytes = _feature_records(feature_payload)
    if set(feature_records) != set(expected):
        missing = sorted(set(expected) - set(feature_records))
        extra = sorted(set(feature_records) - set(expected))
        raise RuntimeError(f"feature key mismatch: missing={missing[:3]} extra={extra[:3]}")
    if not set(expected) <= set(label_records):
        missing_labels = sorted(set(expected) - set(label_records))
        raise RuntimeError(f"missing frozen labels: {missing_labels[:3]}")

    group_counts = Counter(anchor.recording_group for anchor in expected.values())
    if args.max_anchors is None and any(count != 16 for count in group_counts.values()):
        raise RuntimeError("complete-group audit requires exactly 16 anchors per group")

    feature_files: dict[str, tuple[str, str]] = {}
    projection_hash_mismatches = 0
    projection_failures = 0
    supported_node_slot_count = 0
    all_missing_node_count = 0
    records_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record_key in sorted(expected):
        anchor = expected[record_key]
        record = feature_records[record_key]
        if (
            record.get("recording_group") != anchor.recording_group
            or int(record.get("anchor_index", -1)) != anchor.anchor_index
            or record.get("split") != anchor.split
        ):
            raise RuntimeError(f"feature metadata mismatch for {record_key}")
        xy_path = args.feature_root / str(record["projection_xy_relative_path"])
        mask_path = args.feature_root / str(record["projection_mask_relative_path"])
        if sha256_file(xy_path) != record["projection_xy_sha256"]:
            projection_hash_mismatches += 1
        if sha256_file(mask_path) != record["projection_mask_sha256"]:
            projection_hash_mismatches += 1
        try:
            with xy_path.open("rb") as stream:
                patch_xy = np.lib.format.read_array(  # type: ignore[no-untyped-call]
                    stream, allow_pickle=False
                )
            with mask_path.open("rb") as stream:
                mask = np.lib.format.read_array(  # type: ignore[no-untyped-call]
                    stream, allow_pickle=False
                )
            patch_xy, mask = validate_projection_arrays(patch_xy, mask)
        except (OSError, TypeError, ValueError):
            projection_failures += 1
            continue
        supported_node_slot_count += int(np.sum(mask))
        all_missing_node_count += int(np.sum(~np.any(mask, axis=1)))
        slots = record.get("slots")
        if not isinstance(slots, list) or len(slots) != 12:
            raise RuntimeError(f"invalid slot list for {record_key}")
        slot_indices = {int(slot["slot_index"]) for slot in slots}
        if slot_indices != set(range(12)):
            raise RuntimeError(f"invalid slot order for {record_key}")
        for slot in slots:
            relative = str(slot["feature_relative_path"])
            image_sha256 = str(slot["image_sha256"])
            feature_sha256 = str(slot["feature_sha256"])
            prior = feature_files.get(relative)
            current = (image_sha256, feature_sha256)
            if prior is not None and prior != current:
                raise RuntimeError(f"inconsistent feature reference: {relative}")
            feature_files[relative] = current
        records_by_group[anchor.recording_group].append(record)

    feature_hash_mismatches = 0
    feature_failures = 0
    dense_cache: dict[str, npt.NDArray[np.float32]] = {}
    for index, (relative, (_image_sha256, expected_sha256)) in enumerate(
        sorted(feature_files.items()), start=1
    ):
        path = args.feature_root / relative
        if sha256_file(path) != expected_sha256:
            feature_hash_mismatches += 1
            continue
        try:
            bits = load_dense_bfloat16_bits(path)
            if args.retain_dense_cache:
                dense_cache[relative] = bfloat16_bits_to_float32(bits)
        except (OSError, TypeError, ValueError):
            feature_failures += 1
        if index % 256 == 0:
            print(f"checked {index}/{len(feature_files)} dense feature files", flush=True)

    sampled_group_count = 0
    sampler_max_absolute_error = 0.0
    sampled_finite = True
    invalid_storage_exact_zero = True
    node_indices = np.unique(np.linspace(0, 4095, 64, dtype=np.int64))
    for group in sorted(records_by_group):
        record = sorted(records_by_group[group], key=lambda item: int(item["anchor_index"]))[0]
        xy_path = args.feature_root / str(record["projection_xy_relative_path"])
        mask_path = args.feature_root / str(record["projection_mask_relative_path"])
        with xy_path.open("rb") as stream:
            patch_xy = np.lib.format.read_array(  # type: ignore[no-untyped-call]
                stream, allow_pickle=False
            )[node_indices]
        with mask_path.open("rb") as stream:
            mask = np.lib.format.read_array(  # type: ignore[no-untyped-call]
                stream, allow_pickle=False
            )[node_indices]
        slot_features: list[npt.NDArray[np.float32]] = []
        for slot in sorted(record["slots"], key=lambda item: int(item["slot_index"])):
            relative = str(slot["feature_relative_path"])
            dense = dense_cache.get(relative)
            if dense is None:
                dense = bfloat16_bits_to_float32(
                    load_dense_bfloat16_bits(args.feature_root / relative)
                )
            slot_features.append(dense)
        dense_array = np.stack(slot_features, axis=0)
        numpy_sample = bilinear_sample_dense_numpy(dense_array, patch_xy, mask)
        torch_sample = _torch_sample(dense_array, patch_xy, mask)
        sampler_max_absolute_error = max(
            sampler_max_absolute_error,
            float(np.max(np.abs(numpy_sample - torch_sample))),
        )
        sampled_finite &= bool(np.all(np.isfinite(numpy_sample)))
        invalid_storage_exact_zero &= bool(np.all(numpy_sample[~mask] == 0.0))
        sampled_group_count += 1

    failures = []
    if projection_hash_mismatches:
        failures.append(f"projection_hash_mismatches={projection_hash_mismatches}")
    if projection_failures:
        failures.append(f"projection_failures={projection_failures}")
    if feature_hash_mismatches:
        failures.append(f"feature_hash_mismatches={feature_hash_mismatches}")
    if feature_failures:
        failures.append(f"feature_failures={feature_failures}")
    if not sampled_finite:
        failures.append("sampled_nonfinite")
    if not invalid_storage_exact_zero:
        failures.append("invalid_sample_storage_nonzero")
    if sampler_max_absolute_error > args.sampler_tolerance:
        failures.append(
            f"sampler_error={sampler_max_absolute_error} > {args.sampler_tolerance}"
        )
    status = "PASS" if not failures else "FAIL"
    result: dict[str, object] = {
        "experiment_id": "R004E-B-feature-label-join-audit",
        "implementation_revision": args.implementation_revision,
        "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_anchor_count": len(expected),
        "recording_group_count": len(group_counts),
        "feature_record_count": len(feature_records),
        "matched_label_record_count": len(expected),
        "unique_dense_feature_count": len(feature_files),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "label_manifest_sha256": sha256_file(args.label_manifest),
        "artifact_bytes": artifact_bytes,
        "projection_hash_mismatches": projection_hash_mismatches,
        "projection_failures": projection_failures,
        "feature_hash_mismatches": feature_hash_mismatches,
        "feature_failures": feature_failures,
        "supported_node_slot_count": supported_node_slot_count,
        "all_missing_node_count": all_missing_node_count,
        "sampled_group_count": sampled_group_count,
        "sampled_nodes_per_group": int(len(node_indices)),
        "sampler_max_absolute_error": sampler_max_absolute_error,
        "sampler_tolerance": args.sampler_tolerance,
        "sampled_finite": sampled_finite,
        "invalid_storage_exact_zero": invalid_storage_exact_zero,
        "calibration_test_or_target_accessed": False,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--sampler-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-anchors", type=int)
    parser.add_argument("--max-recording-groups", type=int)
    parser.add_argument(
        "--retain-dense-cache",
        action="store_true",
        help="retain all decoded dense features in RAM during the audit",
    )
    args = parser.parse_args()
    result = audit(args)
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
