"""Audit cached PDM-native terminal-speed semantics on the frozen H4P2 scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from aorta_drive.evaluation.h4 import (
    sample_cached_native_trajectory,
    terminal_segment_average_speed,
)

from diagnose_r004h4p2_candidate_supply import (
    _metric_records,
    _selected_anchors,
    _sha256,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for name, path in (
        ("time_set", args.time_set),
        ("role_manifest", args.role_manifest),
        ("metric_manifest", args.metric_manifest),
    ):
        if _sha256(path) != config["inputs"][f"{name}_sha256"]:
            raise ValueError(f"{name} hash disagrees with registration")

    anchors = _selected_anchors(args.time_set, args.role_manifest)
    if len(anchors) != 128:
        raise ValueError("audit requires exactly eight groups times sixteen anchors")
    metrics = _metric_records(json.loads(args.metric_manifest.read_text(encoding="utf-8")))
    source_hashes: dict[str, str] = {}
    for relative in config["source_files"]:
        path = args.navsim_root / relative
        if not path.is_file():
            raise ValueError(f"registered NAVSIM source file is missing: {relative}")
        source_hashes[relative] = _sha256(path)

    interval_s = float(config["quantities"]["reference_interval_s"])
    moving_min = float(config["decision"]["moving_reference_minimum_mps"])
    disagreement_min = float(config["decision"]["material_disagreement_mps"])
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        key = (anchor["log_name"], anchor["scene_token"])
        record = metrics.get(key)
        if record is None:
            raise ValueError("metric manifest lacks a registered anchor")
        cache_path = args.metric_root / str(record["relative_cache_path"])
        if _sha256(cache_path) != record["cache_sha256"]:
            raise ValueError("metric cache raw hash mismatch")
        with lzma.open(cache_path, "rb") as stream:
            cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM input
        native = sample_cached_native_trajectory(cache)
        reference = terminal_segment_average_speed(native.poses, interval_s=interval_s)
        suspect = native.terminal_speed_mps
        disagreement = abs(reference - suspect)
        material = reference >= moving_min and disagreement > disagreement_min
        rows.append(
            {
                "recording_group": anchor["recording_group"],
                "anchor_index": int(anchor["anchor_index"]),
                "log_name": anchor["log_name"],
                "scene_token": anchor["scene_token"],
                "cached_dynamic_terminal_speed_mps": suspect,
                "position_derived_final_segment_average_speed_mps": reference,
                "absolute_disagreement_mps": disagreement,
                "material_moving_disagreement": material,
                "native_sample_sha256": native.canonical_sample_sha256,
            }
        )

    material_count = sum(int(row["material_moving_disagreement"]) for row in rows)
    zero_suspect_count = sum(
        int(float(row["cached_dynamic_terminal_speed_mps"]) == 0.0) for row in rows
    )
    moving_count = sum(
        int(float(row["position_derived_final_segment_average_speed_mps"]) >= moving_min)
        for row in rows
    )
    reject = material_count > 0
    result = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": (
            "REJECT_CACHED_DYNAMIC_SPEED_SEMANTICS"
            if reject
            else "RETAIN_CACHED_DYNAMIC_SPEED_SEMANTICS"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": _sha256(args.config),
        "elapsed_seconds": time.perf_counter() - started,
        "data_scope": {
            "role": "fit",
            "split": "train",
            "recording_group_count": 8,
            "anchor_count": 128,
            "development_calibration_test_or_target_accessed": False,
        },
        "source_sha256": source_hashes,
        "thresholds": {
            "moving_reference_minimum_mps": moving_min,
            "material_disagreement_mps": disagreement_min,
        },
        "summary": {
            "moving_reference_anchor_count": moving_count,
            "zero_cached_dynamic_speed_anchor_count": zero_suspect_count,
            "material_moving_disagreement_anchor_count": material_count,
            "maximum_absolute_disagreement_mps": max(
                float(row["absolute_disagreement_mps"]) for row in rows
            ),
        },
        "anchor_ledger": rows,
        "claims": {
            "cached_dynamic_terminal_speed_is_valid_for_h4_intent_matching": not reject,
            "position_derived_quantity_is_instantaneous_speed": False,
            "candidate_provider_changed": False,
            "intent_threshold_changed": False,
            "geometry_evaluated": False,
            "teacher_or_prediction_read": False,
            "development_access_authorized": False,
            "h4_registered": False,
        },
    }
    if not math.isfinite(result["summary"]["maximum_absolute_disagreement_mps"]):
        raise ValueError("non-finite speed disagreement")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--navsim-root", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "anchor_ledger"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
