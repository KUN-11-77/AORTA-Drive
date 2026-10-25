"""Independent mechanical audit for an R005E2 scientific tensor bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from aorta_drive.training.r005e2_science import (  # type: ignore[import-untyped]
    validate_bundle_arrays,
)
from aorta_drive.vision.feature_cache import sha256_file  # type: ignore[import-untyped]


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
        "source_only": True,
        "development_calibration_test_navdream_target_accessed": False,
        "artifact_sha256": hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = audit(args)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
