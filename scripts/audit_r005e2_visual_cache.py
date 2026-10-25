"""Independent replay and artifact audit for the R005E2 frozen visual cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from aorta_drive.protocol.r005e2_visual_cache import (
    anchors_from_panel_roster,
    exact_slot_order,
)
from aorta_drive.vision.feature_cache import (
    load_dense_bfloat16_bits,
    sha256_file,
    validate_projection_arrays,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _load_npy(path: Path) -> npt.NDArray[np.generic]:
    with path.open("rb") as stream:
        return np.lib.format.read_array(stream, allow_pickle=False)  # type: ignore[no-any-return,no-untyped-call]


def _record_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    records = cast(list[dict[str, Any]], payload.get("records"))
    index = {(str(row["log_name"]), str(row["scene_token"])): row for row in records}
    if len(index) != len(records):
        raise ValueError("visual cache contains duplicate log/token identities")
    return index


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.panel_identity_roster) != args.expected_roster_sha256:
        raise ValueError("panel identity roster file hash mismatch")
    roster_payload = cast(
        dict[str, Any],
        json.loads(args.panel_identity_roster.read_text(encoding="utf-8")),
    )
    anchors, roster_audit = anchors_from_panel_roster(roster_payload)
    primary = cast(dict[str, Any], json.loads(args.primary_manifest.read_text(encoding="utf-8")))
    replay = cast(dict[str, Any], json.loads(args.replay_manifest.read_text(encoding="utf-8")))
    for name, payload in (("primary", primary), ("replay", replay)):
        if payload.get("status") != "PASS":
            raise ValueError(f"{name} cache did not finish with PASS")
        if payload.get("schema") != "aorta.r005e2.frozen_visual_cache.v1":
            raise ValueError(f"{name} cache schema mismatch")
        if payload.get("development_calibration_test_navdream_target_accessed") is not False:
            raise ValueError(f"{name} cache accessed a sealed data role")
        if payload.get("utility_candidate_scores_or_labels_accessed") is not False:
            raise ValueError(f"{name} cache accessed scientific outcomes")
        if payload.get("allowed_splits") != ["train"]:
            raise ValueError(f"{name} cache split contract mismatch")
        if int(payload.get("anchor_count", -1)) != roster_audit.anchor_count:
            raise ValueError(f"{name} cache anchor count mismatch")
        if int(payload.get("recording_group_count", -1)) != roster_audit.recording_group_count:
            raise ValueError(f"{name} cache group count mismatch")
        if int(payload.get("slot_count", -1)) != 12:
            raise ValueError(f"{name} cache slot count mismatch")
        if payload["input_hashes"]["panel_identity_roster_sha256"] != args.expected_roster_sha256:
            raise ValueError(f"{name} cache roster provenance mismatch")

    stable_keys = (
        "allowed_splits",
        "anchor_count",
        "recording_group_count",
        "slot_count",
        "image_reference_count",
        "unique_image_count",
        "feature_contract",
        "projection_contract",
        "logical_manifest_sha256",
    )
    if any(primary[key] != replay[key] for key in stable_keys):
        raise ValueError("primary/replay stable manifest contracts differ")
    if primary["worker_contract"]["batch_cardinalities"] != replay["worker_contract"][
        "batch_cardinalities"
    ]:
        raise ValueError("primary/replay batch cardinalities differ")

    primary_index = _record_index(primary)
    replay_index = _record_index(replay)
    expected_keys = {(anchor.log_name, anchor.scene_token) for anchor in anchors}
    if set(primary_index) != expected_keys or set(replay_index) != expected_keys:
        raise ValueError("cache identities do not exactly equal the frozen roster")

    feature_hash_by_path: dict[str, str] = {}
    projection_count = 0
    for key in sorted(expected_keys):
        left = primary_index[key]
        right = replay_index[key]
        identity_fields = ("recording_group", "anchor_index", "split", "log_name", "scene_token")
        if any(left[field] != right[field] for field in identity_fields):
            raise ValueError("primary/replay record identity mismatch")
        left_slots = cast(list[dict[str, Any]], left["slots"])
        right_slots = cast(list[dict[str, Any]], right["slots"])
        if not exact_slot_order(left_slots) or not exact_slot_order(right_slots):
            raise ValueError("frozen visual slot order mismatch")
        if left_slots != right_slots:
            raise ValueError("primary/replay visual slot ledgers differ")

        for row, root in ((left, args.primary_root), (right, args.replay_root)):
            xy_path = root / str(row["projection_xy_relative_path"])
            mask_path = root / str(row["projection_mask_relative_path"])
            if sha256_file(xy_path) != str(row["projection_xy_sha256"]):
                raise ValueError("projection-coordinate artifact hash mismatch")
            if sha256_file(mask_path) != str(row["projection_mask_sha256"]):
                raise ValueError("projection-mask artifact hash mismatch")
            validate_projection_arrays(_load_npy(xy_path), _load_npy(mask_path))
            projection_count += 1
        if left["projection_xy_sha256"] != right["projection_xy_sha256"]:
            raise ValueError("primary/replay projection-coordinate replay failed")
        if left["projection_mask_sha256"] != right["projection_mask_sha256"]:
            raise ValueError("primary/replay projection-mask replay failed")

        for slot in left_slots:
            relative = str(slot["feature_relative_path"])
            if relative != f"dense_features/{slot['image_sha256'][:2]}/{slot['image_sha256']}.npy":
                raise ValueError("feature path is not content-addressed by raw-image hash")
            claimed = str(slot["feature_sha256"])
            prior = feature_hash_by_path.setdefault(relative, claimed)
            if prior != claimed:
                raise ValueError("one content-addressed feature has conflicting hashes")

    for relative, expected_hash in sorted(feature_hash_by_path.items()):
        primary_path = args.primary_root / relative
        replay_path = args.replay_root / relative
        load_dense_bfloat16_bits(primary_path)
        load_dense_bfloat16_bits(replay_path)
        if sha256_file(primary_path) != expected_hash:
            raise ValueError("primary dense-feature hash mismatch")
        if sha256_file(replay_path) != expected_hash:
            raise ValueError("replay dense-feature hash mismatch")

    return {
        "schema": "aorta.r005e2.frozen_visual_cache.audit.v1",
        "status": "PASS",
        "panel_identity_roster_sha256": args.expected_roster_sha256,
        "primary_manifest_sha256": sha256_file(args.primary_manifest),
        "replay_manifest_sha256": sha256_file(args.replay_manifest),
        "primary_replay_logical_manifest_exact": True,
        "primary_replay_feature_artifacts_byte_exact": True,
        "primary_replay_projection_artifacts_byte_exact": True,
        "all_artifacts_reopened_and_finite": True,
        "identity_join_exact": True,
        "slot_order_exact": True,
        "source_fit_only": True,
        "utility_candidate_scores_or_labels_accessed": False,
        "development_calibration_test_navdream_target_accessed": False,
        "anchor_count": len(expected_keys),
        "recording_group_count": roster_audit.recording_group_count,
        "unique_feature_artifact_count": len(feature_hash_by_path),
        "reopened_projection_artifact_count": projection_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-identity-roster", type=Path, required=True)
    parser.add_argument("--expected-roster-sha256", required=True)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = audit(arguments)
    _write_json(arguments.output, output)
    print(json.dumps(output, indent=2))
