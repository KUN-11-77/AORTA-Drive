"""Fail-closed input preflight for the preregistered R004H6 panel."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from aorta_drive.evaluation.heldout_visual import deterministic_group_derangement
from aorta_drive.vision.feature_cache import sha256_file


def _groups(path: Path, *, column: str, value: str) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = tuple(str(row["recording_group"]) for row in rows if row[column] == value)
    if len(groups) != len(set(groups)):
        raise ValueError("recording-group manifest contains duplicates")
    return groups


def audit(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    observed = {
        "source_split_manifest_sha256": sha256_file(args.source_split_manifest),
        "role_manifest_sha256": sha256_file(args.role_manifest),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
    }
    for key, actual in observed.items():
        if actual != frozen[key]:
            raise ValueError(f"{key} disagrees with the frozen R004H6 config")
    development = _groups(args.source_split_manifest, column="split", value="development")
    fit = _groups(args.role_manifest, column="role", value="fit")
    if len(development) != 24 or len(fit) != 42 or set(development) & set(fit):
        raise ValueError("R004H6 split cardinality or independence failed")
    donor_map = deterministic_group_derangement(development)

    payload = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("feature manifest lacks records")
    selected = [row for row in records if isinstance(row, dict) and row.get("recording_group") in set(development)]
    counts = {group: 0 for group in development}
    anchors: set[tuple[str, int]] = set()
    images: dict[str, str] = {}
    for row in selected:
        group = str(row["recording_group"])
        anchor = int(row["anchor_index"])
        anchor_key = (group, anchor)
        if anchor_key in anchors:
            raise ValueError("duplicate development group/anchor")
        anchors.add(anchor_key)
        counts[group] += 1
        slots = row.get("slots")
        if not isinstance(slots, list) or sorted(int(slot["slot_index"]) for slot in slots) != list(range(12)):
            raise ValueError("development anchor does not have slots 0..11")
        for slot in slots:
            relative = str(slot["image_relative_path"])
            expected = str(slot["image_sha256"])
            prior = images.get(relative)
            if prior is not None and prior != expected:
                raise ValueError("raw-image path has inconsistent hashes")
            images[relative] = expected
    if len(selected) != 384 or set(counts.values()) != {16}:
        raise ValueError("development panel is incomplete")
    if any((donor_map[group], anchor) not in anchors for group, anchor in anchors):
        raise ValueError("donor mapping lacks same-index anchors")

    missing: list[str] = []
    mismatched: list[str] = []
    total_bytes = 0
    for relative, expected in sorted(images.items()):
        path = (args.sensor_root / relative).resolve()
        if not path.is_relative_to(args.sensor_root.resolve()):
            raise ValueError("raw-image path escapes sensor root")
        if not path.is_file():
            missing.append(relative)
            continue
        total_bytes += path.stat().st_size
        if sha256_file(path) != expected:
            mismatched.append(relative)
    if missing or mismatched:
        raise RuntimeError(f"raw replay failed: missing={len(missing)} mismatched={len(mismatched)}")
    return {
        "experiment_id": "R004H6-input-replay-preflight-v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        **observed,
        "development_group_count": 24,
        "development_anchor_count": 384,
        "unique_raw_image_count": len(images),
        "raw_image_bytes": total_bytes,
        "missing_raw_image_count": 0,
        "hash_mismatch_count": 0,
        "donor_derangement": donor_map,
        "model_or_label_payload_accessed": False,
        "calibration_test_or_target_accessed": False,
        "claim_boundary": "Identity, completeness, raw hashes, and donor mapping only; no scientific metric.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
