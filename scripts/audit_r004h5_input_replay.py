"""Fail-closed raw-image and pairing preflight for R004H5."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file


def _fit_groups(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"recording_group", "role", "historical_exposure"}
    if not rows or set(rows[0]) != required:
        raise ValueError("role manifest columns disagree with R004H5")
    groups = tuple(row["recording_group"] for row in rows if row["role"] == "fit")
    if len(groups) != 42 or len(set(groups)) != 42:
        raise ValueError("R004H5 requires exactly 42 unique fit groups")
    return groups


def audit(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    observed_hashes = {
        "role_manifest_sha256": sha256_file(args.role_manifest),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
    }
    for key, actual in observed_hashes.items():
        if actual != frozen[key]:
            raise ValueError(f"{key} disagrees with the frozen R004H5 config")

    fit_groups = frozenset(_fit_groups(args.role_manifest))
    manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("feature manifest lacks records")
    selected = [
        record
        for record in records
        if isinstance(record, dict) and record.get("recording_group") in fit_groups
    ]
    if len(selected) != 672:
        raise ValueError("feature manifest does not contain 42 complete fit groups")
    group_counts = {group: 0 for group in fit_groups}
    scene_tokens: set[str] = set()
    image_records: dict[str, tuple[str, str]] = {}
    slot_count = 0
    for record in selected:
        group = str(record["recording_group"])
        group_counts[group] += 1
        token = str(record["scene_token"])
        if token in scene_tokens:
            raise ValueError("duplicate fit scene token")
        scene_tokens.add(token)
        slots = record.get("slots")
        if not isinstance(slots, list) or len(slots) != 12:
            raise ValueError("every fit anchor must have 12 image slots")
        indices = sorted(int(slot["slot_index"]) for slot in slots)
        if indices != list(range(12)):
            raise ValueError("slot indices must be exactly 0..11")
        slot_count += len(slots)
        for slot in slots:
            relative = str(slot["image_relative_path"])
            expected = str(slot["image_sha256"])
            prior = image_records.get(relative)
            if prior is not None and prior != (expected, str(slot["feature_relative_path"])):
                raise ValueError("one raw image path has inconsistent frozen metadata")
            image_records[relative] = (expected, str(slot["feature_relative_path"]))
    if set(group_counts.values()) != {16}:
        raise ValueError("each fit recording group must contain 16 anchors")

    missing: list[str] = []
    mismatched: list[str] = []
    total_bytes = 0
    for relative, (expected, _feature_relative) in sorted(image_records.items()):
        path = args.sensor_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        total_bytes += path.stat().st_size
        if sha256_file(path) != expected:
            mismatched.append(relative)
    if missing or mismatched:
        raise RuntimeError(
            f"raw image replay failed: missing={len(missing)} mismatched={len(mismatched)}"
        )

    return {
        "experiment_id": "R004H5-input-replay-preflight-v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        **observed_hashes,
        "fit_group_count": len(fit_groups),
        "fit_anchor_count": len(selected),
        "slot_reference_count": slot_count,
        "unique_raw_image_count": len(image_records),
        "raw_image_bytes": total_bytes,
        "missing_raw_image_count": 0,
        "hash_mismatch_count": 0,
        "prohibited_payload_accessed": False,
        "claim_boundary": (
            "Raw-image identity and exact fit pairing only. No model inference, appearance "
            "stability, task accuracy, held-out generalization, OOD, or driving claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

