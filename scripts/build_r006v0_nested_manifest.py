#!/usr/bin/env python
"""Build the outcome-free nested group manifest for formal R006V0."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aorta_drive.vision.r006_protocol import (
    audit_r006_nested_manifest,
    build_r006_nested_manifest,
)
from aorta_drive.protocol.visual_recoverability import canonical_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-salt", default="aorta-r006v0-outer-v1")
    parser.add_argument("--inner-salt", default="aorta-r006v0-inner-selection-v1")
    args = parser.parse_args()

    with args.group_roles.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = sorted(row["recording_group"] for row in rows if row["role"] == "fit")
    if len(groups) != 42 or len(groups) != len(set(groups)):
        raise RuntimeError("R006V0 requires exactly 42 unique source-fit groups")
    manifest = build_r006_nested_manifest(
        groups,
        outer_salt=args.outer_salt,
        inner_salt=args.inner_salt,
    )
    manifest["audit"] = audit_r006_nested_manifest(
        manifest,
        expected_groups=groups,
    )
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
                "groups": len(groups),
                "outer_folds": manifest["outer_fold_count"],
                "manifest_sha256_without_self": manifest[
                    "manifest_sha256_without_self"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
