#!/usr/bin/env python
"""Build the outcome-free wrong-scene donor map for formal R006V0."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aorta_drive.protocol.visual_recoverability import canonical_sha256
from aorta_drive.vision.r006_cache import sha256_file
from aorta_drive.vision.r006_donors import build_wrong_scene_donor_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-roles", type=Path, required=True)
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--nested-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", default="aorta-r006v0-wrong-scene-donor-v1")
    args = parser.parse_args()

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
    nested = json.loads(args.nested_manifest.read_text(encoding="utf-8"))
    manifest = build_wrong_scene_donor_manifest(
        rows,
        nested,
        salt=args.salt,
    )
    manifest["provenance"] = {
        "group_roles_sha256": sha256_file(args.group_roles),
        "time_set_sha256": sha256_file(args.time_set),
        "nested_manifest_sha256": sha256_file(args.nested_manifest),
    }
    manifest.pop("manifest_sha256_without_self")
    manifest["manifest_sha256_without_self"] = canonical_sha256(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "assignments": manifest["assignment_count"],
                "manifest_sha256_without_self": manifest[
                    "manifest_sha256_without_self"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
