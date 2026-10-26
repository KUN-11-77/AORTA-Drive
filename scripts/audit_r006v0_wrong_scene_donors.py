#!/usr/bin/env python
"""Independently audit the formal R006V0 wrong-scene donor map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aorta_drive.vision.r006_cache import sha256_file
from aorta_drive.vision.r006_donors import audit_wrong_scene_donor_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--group-roles", type=Path, required=True)
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--nested-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    nested = json.loads(args.nested_manifest.read_text(encoding="utf-8"))
    with args.group_roles.open(newline="", encoding="utf-8") as stream:
        fit_groups = {
            row["recording_group"]
            for row in csv.DictReader(stream)
            if row["role"] == "fit"
        }
    with args.time_set.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["recording_group"] in fit_groups
        ]
    audit = audit_wrong_scene_donor_manifest(
        manifest,
        rows=rows,
        nested_manifest=nested,
    )
    expected_provenance = {
        "group_roles_sha256": sha256_file(args.group_roles),
        "time_set_sha256": sha256_file(args.time_set),
        "nested_manifest_sha256": sha256_file(args.nested_manifest),
    }
    if manifest.get("provenance") != expected_provenance:
        raise ValueError("wrong-scene donor provenance mismatch")
    result = {
        "artifact": "R006V0_WRONG_SCENE_DONOR_AUDIT_V1",
        "status": "PASS",
        "manifest_sha256_without_self": manifest["manifest_sha256_without_self"],
        "checks": audit,
        "provenance_exact": True,
        "scientific_metrics_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
