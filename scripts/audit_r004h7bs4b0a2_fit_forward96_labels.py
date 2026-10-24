"""Mechanically audit the 672-token forward96 fit teacher artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file  # type: ignore[import-untyped]
from build_r004h7bs4b0a2_fit_forward96_labels import ARRAY_NAMES, select_fit_records


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    contract = cast(dict[str, Any], config["H7BS4B0A2"])
    frozen_impl = cast(Mapping[str, str], contract["implementation_sha256"])
    repository = Path(__file__).resolve().parents[1]
    for name, path in {
        "builder_source": repository / "scripts/build_r004h7bs4b0a2_fit_forward96_labels.py",
        "audit_source": Path(__file__).resolve(),
        "teacher_source": repository / "src/aorta_drive/certification/teacher.py",
        "transform_source": repository / "src/aorta_drive/certification/transform.py",
        "geometry_source": repository / "src/aorta_drive/certification/geometry.py",
        "base_builder_source": repository / "scripts/build_navsim_node_labels.py",
    }.items():
        if sha256_file(path) != frozen_impl[name]:
            raise ValueError(f"implementation hash mismatch: {name}")

    frozen_inputs = cast(Mapping[str, str], contract["frozen_input_sha256"])
    for name, path in {
        "h7bs4a_result": args.anchor_ledger,
        "r004e_metric_manifest": args.metric_manifest,
        "r004e_label_manifest": args.label_manifest,
    }.items():
        if sha256_file(path) != frozen_inputs[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")

    anchors = _read_json(args.anchor_ledger)
    metrics = _read_json(args.metric_manifest)
    labels = _read_json(args.label_manifest)
    selected = select_fit_records(anchors, metrics)
    selected_ids = {f"{r['log_name']}/{r['scene_token']}" for r in selected}
    old_by_id = {
        str(record["cache_id"]): record
        for record in cast(list[dict[str, Any]], labels["records"])
    }
    manifest = _read_json(args.manifest)
    records = cast(list[dict[str, Any]], manifest.get("records"))
    failures: list[str] = []
    if manifest.get("status") != "completed" or manifest.get("failures"):
        failures.append("manifest_not_complete")
    if len(records) != 672 or {str(record["cache_id"]) for record in records} != selected_ids:
        failures.append("record_identity_mismatch")
    if (
        manifest.get("recording_group_count") != 42
        or manifest.get("total_node_labels") != 4_128_768
    ):
        failures.append("scope_count_mismatch")
    logical = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("logical_manifest_sha256") != logical:
        failures.append("logical_manifest_hash_mismatch")

    checked = 0
    for record in records:
        cache_id = str(record["cache_id"])
        old_record = old_by_id.get(cache_id)
        if old_record is None:
            failures.append(f"old_record_missing:{cache_id}")
            continue
        new_path = args.artifact_root / "labels" / str(record["artifact_name"])
        old_path = args.old_label_root / "labels" / str(old_record["artifact_name"])
        if not new_path.is_file() or sha256_file(new_path) != record["artifact_sha256"]:
            failures.append(f"artifact_hash:{cache_id}")
            continue
        if not old_path.is_file() or sha256_file(old_path) != old_record["artifact_sha256"]:
            failures.append(f"old_artifact_hash:{cache_id}")
            continue
        with (
            np.load(new_path, allow_pickle=False) as new,
            np.load(old_path, allow_pickle=False) as old,
        ):
            if set(new.files) != set(ARRAY_NAMES) or set(old.files) != set(ARRAY_NAMES):
                failures.append(f"array_names:{cache_id}")
                continue
            valid = True
            for name in ARRAY_NAMES:
                if (
                    new[name].shape != (64, 96)
                    or old[name].shape != (64, 64)
                    or not np.array_equal(new[name][:, :64], old[name])
                    or not np.all(np.isfinite(new[name]))
                ):
                    valid = False
                    break
            if valid and (
                np.any(new["transform_error_m"] < 0)
                or np.any(new["distance_error_m"] < 0)
                or np.any(new["total_error_m"] < new["transform_error_m"])
                or np.any(new["total_error_m"] < new["distance_error_m"])
            ):
                valid = False
            if not valid:
                failures.append(f"array_or_prefix:{cache_id}")
                continue
        checked += 1

    passed = not failures and checked == 672
    result = {
        "experiment_id": config["experiment_id"],
        "manifest_sha256": sha256_file(args.manifest),
        "anchor_count": len(selected),
        "recording_group_count": 42,
        "artifact_check_count": checked,
        "total_node_count": 672 * 6144,
        "prefix_exact_count": checked,
        "failure_count": len(failures),
        "failures": failures,
        "mechanical_audit": "PASS" if passed else "FAIL",
        "external_scientific_review": "REVIEW_UNAVAILABLE",
        "claim_boundary": "fit teacher artifact qualification only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--anchor-ledger", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--old-label-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(int(result["mechanical_audit"] != "PASS"))
