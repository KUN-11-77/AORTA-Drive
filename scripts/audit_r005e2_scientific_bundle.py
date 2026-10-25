"""Independent mechanical audit for an R005E2 scientific tensor bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from aorta_drive.training.r005e2_science import (
    validate_bundle_arrays,
)
from aorta_drive.vision.feature_cache import sha256_file


def audit(args: argparse.Namespace) -> dict[str, Any]:
    primary = cast(
        dict[str, Any], json.loads((args.primary / "manifest.json").read_text("utf-8"))
    )
    replay = cast(
        dict[str, Any], json.loads((args.replay / "manifest.json").read_text("utf-8"))
    )
    for payload in (primary, replay):
        if payload.get("status") != "PASS" or payload.get("anchor_count") != 672:
            raise ValueError("scientific bundle did not complete")
        if (
            payload.get("development_calibration_test_navdream_target_accessed")
            is not False
        ):
            raise ValueError("scientific bundle accessed a sealed role")
        if payload.get("previous_context_contract") != {
            "source": "R005E1B_hash_verified_previous_native_official_simulated_states",
            "coordinate_frame": "current_anchor_ego_rear_axle",
            "temporal_support": "previous_states[round(gap/0.1):]",
            "shape": [36, 3],
            "sample_interval_s": 0.1,
            "shared_current_candidate_horizon_s": 3.5,
            "expired_previous_prefix_retained": False,
            "interpolation_or_padding_used": False,
        }:
            raise ValueError(
                "previous-context contract differs from the preregistration"
            )
    validate_bundle_arrays(args.primary)
    validate_bundle_arrays(args.replay)
    artifacts = [
        "candidate_poses.npy",
        "candidate_mask.npy",
        "previous_poses.npy",
        "target_margin.npy",
        *(f"removed_slot_mean_fold_{fold}.npy" for fold in range(6)),
    ]
    hashes = {
        name: {
            "primary": sha256_file(args.primary / name),
            "replay": sha256_file(args.replay / name),
        }
        for name in artifacts
    }
    if any(row["primary"] != row["replay"] for row in hashes.values()):
        raise ValueError("scientific bundle primary/replay mismatch")
    primary_logical = dict(primary)
    replay_logical = dict(replay)
    primary_logical.pop("completed_at_utc", None)
    replay_logical.pop("completed_at_utc", None)
    if primary_logical != replay_logical:
        raise ValueError("scientific bundle logical manifests differ")
    target = cast(
        dict[str, Any], json.loads(args.target_result.read_text(encoding="utf-8"))
    )
    target_by_key = {
        (str(row["recording_group"]), int(row["anchor_index"])): row
        for row in cast(list[dict[str, Any]], target["anchor_ledgers"])
    }
    if len(target_by_key) != 672:
        raise ValueError("target ledger does not contain 672 unique anchors")
    for row in cast(list[dict[str, Any]], primary["anchors"]):
        key = (str(row["recording_group"]), int(row["anchor_index"]))
        expected = target_by_key[key]
        history = cast(dict[str, Any], row["history_provenance"])
        if (
            str(history["previous_simulated_states_sha256"])
            != str(expected["previous_native_simulated_states_sha256"])
            or int(history["overlap_start"]) != 5
            or abs(
                float(history["observed_gap_s"])
                - float(expected["observation_interval_s"])
            )
            > 1.0e-6
            or not np.isfinite(float(history["maximum_transform_error_m"]))
            or float(history["maximum_transform_error_m"]) < 0.0
        ):
            raise ValueError("previous-context provenance differs from R005E1B")
    for fold in range(6):
        value = np.load(args.primary / f"removed_slot_mean_fold_{fold}.npy")
        if value.shape != (12, 1, 1, 1280) or not np.isfinite(value).all():
            raise ValueError("removed slot mean contract failed")
    return {
        "schema": "aorta.r005e2.scientific_bundle_audit.v1",
        "status": "PASS",
        "primary_replay_artifacts_exact": True,
        "logical_manifest_exact": True,
        "finite_values": True,
        "previous_context_target_provenance_exact": True,
        "previous_context_coordinate_and_overlap_contract": True,
        "source_only": True,
        "development_calibration_test_navdream_target_accessed": False,
        "artifact_sha256": hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--target-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = audit(args)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
