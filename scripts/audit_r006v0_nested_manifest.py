#!/usr/bin/env python
"""Independently audit the frozen R006V0 nested group manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aorta_drive.vision.r006_protocol import audit_r006_nested_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--group-roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with args.group_roles.open(newline="", encoding="utf-8") as stream:
        groups = sorted(
            row["recording_group"]
            for row in csv.DictReader(stream)
            if row["role"] == "fit"
        )
    audit = audit_r006_nested_manifest(manifest, expected_groups=groups)
    result = {
        "artifact": "R006V0_NESTED_MANIFEST_AUDIT_V1",
        "status": "PASS",
        "manifest_sha256_without_self": manifest["manifest_sha256_without_self"],
        "checks": {
            "source_fit_groups_exact": len(groups) == 42,
            "outer_confirmation_exact_once": audit[
                "each_group_outer_test_exactly_once"
            ],
            "outcome_free_split": audit["outcomes_or_features_used"] is False,
            "outer_folds": audit["outer_fold_count"],
            "outer_confirmation_groups_per_fold": audit["outer_test_size_min"],
            "basis_discovery_groups_per_fold": audit["discovery_group_count"],
            "layer_rank_selection_groups_per_fold": audit["selection_group_count"],
        },
        "scientific_metrics_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
    }
    if not all(
        value is True
        for key, value in result["checks"].items()
        if key
        in {
            "source_fit_groups_exact",
            "outer_confirmation_exact_once",
            "outcome_free_split",
        }
    ):
        raise RuntimeError("R006V0 nested manifest audit failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
