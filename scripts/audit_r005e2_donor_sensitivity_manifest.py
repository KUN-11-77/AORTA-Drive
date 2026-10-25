"""Independent structural audit of frozen R005E2 S2/S3 donor mappings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from aorta_drive.protocol.visual_mismatch import anchor_key
from aorta_drive.protocol.visual_recoverability import canonical_sha256
from aorta_drive.vision.feature_cache import sha256_file


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = cast(dict[str, Any], json.loads(args.manifest.read_text("utf-8")))
    primary = cast(dict[str, Any], json.loads(args.primary_manifest.read_text("utf-8")))
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text("utf-8"))
    )
    roster = cast(
        dict[str, Any], json.loads(args.panel_identity_roster.read_text("utf-8"))
    )
    if manifest.get("scientific_or_visual_content_used") is not False:
        raise ValueError("sensitivity manifest is not content free")
    if (
        manifest.get("development_calibration_test_navdream_target_accessed")
        is not False
    ):
        raise ValueError("sealed role accessed")
    if manifest["source_primary_manifest_sha256"] != sha256_file(args.primary_manifest):
        raise ValueError("primary manifest hash mismatch")
    if manifest["source_crossfit_manifest_sha256"] != sha256_file(
        args.crossfit_manifest
    ):
        raise ValueError("crossfit hash mismatch")
    if manifest["source_panel_identity_roster_sha256"] != sha256_file(
        args.panel_identity_roster
    ):
        raise ValueError("roster hash mismatch")
    selfless = dict(manifest)
    claimed = str(selfless.pop("manifest_sha256_without_self"))
    if canonical_sha256(selfless) != claimed:
        raise ValueError("canonical self-hash mismatch")

    keys_by_group: dict[str, list[str]] = {
        str(group): [] for group in cast(list[str], crossfit["groups"])
    }
    group_by_key: dict[str, str] = {}
    for row in cast(list[dict[str, Any]], roster["anchors"]):
        group = str(row["recording_group"])
        key = anchor_key(group, int(row["anchor_index"]))
        keys_by_group[group].append(key)
        group_by_key[key] = group
    primary_folds = {
        int(row["outer_fold"]): row
        for row in cast(list[dict[str, Any]], primary["folds"])
    }
    manifest_folds = {
        int(row["outer_fold"]): row
        for row in cast(list[dict[str, Any]], manifest["folds"])
    }
    total = 0
    for fold_any in cast(list[dict[str, Any]], crossfit["outer_folds"]):
        fold = dict(fold_any)
        fold_index = int(fold["outer_fold"])
        specs = (
            (
                "selection_fit",
                cast(list[str], fold["inner_fit_groups"]),
                cast(list[str], fold["inner_fit_groups"]),
                True,
            ),
            (
                "selection_validation",
                cast(list[str], fold["inner_validation_groups"]),
                cast(list[str], fold["inner_fit_groups"]),
                False,
            ),
            (
                "refit_train",
                cast(list[str], fold["outer_train_groups"]),
                cast(list[str], fold["outer_train_groups"]),
                True,
            ),
            (
                "outer_test",
                cast(list[str], fold["outer_test_groups"]),
                cast(list[str], fold["outer_train_groups"]),
                False,
            ),
        )
        schemes = cast(dict[str, dict[str, Any]], manifest_folds[fold_index]["schemes"])
        if set(schemes) != {"S2_disjoint_geometry", "S3_hash_random"}:
            raise ValueError("sensitivity scheme roster mismatch")
        for stage_name, recipient_groups, donor_groups, require_permutation in specs:
            expected_recipients = {
                key for group in recipient_groups for key in keys_by_group[group]
            }
            allowed_donors = {
                key for group in donor_groups for key in keys_by_group[group]
            }
            primary_edges = {
                (str(row["recipient"]), str(row["donor"]))
                for row in cast(
                    list[dict[str, Any]],
                    primary_folds[fold_index]["stages"][stage_name]["rows"],
                )
            }
            for scheme_name, stages in schemes.items():
                rows = cast(list[dict[str, Any]], stages[stage_name]["rows"])
                recipients = [str(row["recipient"]) for row in rows]
                donors = [str(row["donor"]) for row in rows]
                if set(recipients) != expected_recipients or len(recipients) != len(
                    set(recipients)
                ):
                    raise ValueError("recipient coverage mismatch")
                if len(donors) != len(set(donors)) or not set(donors) <= allowed_donors:
                    raise ValueError("donor pool or replacement mismatch")
                if require_permutation and set(donors) != allowed_donors:
                    raise ValueError("training visual marginal is not a permutation")
                for recipient, donor in zip(recipients, donors):
                    if group_by_key[recipient] == group_by_key[donor]:
                        raise ValueError("same-group donor detected")
                    if (
                        scheme_name == "S2_disjoint_geometry"
                        and (recipient, donor) in primary_edges
                    ):
                        raise ValueError("S2 reuses a forbidden primary edge")
                total += len(rows)
    if total != int(manifest["total_assignment_count"]):
        raise ValueError("total assignment count mismatch")
    return {
        "schema": "aorta.r005e2.donor_sensitivity.audit.v1",
        "status": "PASS",
        "manifest_sha256": sha256_file(args.manifest),
        "primary_edges_disjoint_for_S2": True,
        "hash_random_rule_frozen_for_S3": True,
        "all_donors_cross_group_fold_isolated_without_replacement": True,
        "scientific_or_visual_content_used": False,
        "development_calibration_test_navdream_target_accessed": False,
        "total_assignment_count": total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--panel-identity-roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = audit(parsed)
    parsed.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
