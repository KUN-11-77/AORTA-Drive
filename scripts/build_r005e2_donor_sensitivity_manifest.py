"""Materialize preregistered S2/S3 R005E2 donor sensitivity mappings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from aorta_drive.protocol.visual_mismatch import (
    anchor_key,
    audit_geometry_matching,
    descriptor_sha256,
    globally_match_sensitivity,
    native_geometry_descriptor,
)
from aorta_drive.protocol.visual_recoverability import canonical_sha256
from aorta_drive.vision.feature_cache import sha256_file
from build_r005e2_mismatch_manifest import (
    _enrich_rows,
    _keys_for_groups,
    _resolve_panel,
)


RANDOM_SALT = "aorta-r005e2-donor-sensitivity-random-v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text("utf-8"))
    )
    roster = cast(
        dict[str, Any], json.loads(args.panel_identity_roster.read_text("utf-8"))
    )
    primary = cast(dict[str, Any], json.loads(args.primary_manifest.read_text("utf-8")))
    if (
        primary.get("utility_candidate_scores_images_dino_or_model_outputs_used")
        is not False
    ):
        raise ValueError("primary donor manifest is not outcome/content free")

    identity_by_key: dict[str, dict[str, Any]] = {}
    keys_by_group: dict[str, list[str]] = {
        str(group): [] for group in cast(list[str], crossfit["groups"])
    }
    group_by_key: dict[str, str] = {}
    descriptors: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    descriptor_hashes: dict[str, str] = {}
    for row in cast(list[dict[str, Any]], roster["anchors"]):
        group = str(row["recording_group"])
        key = anchor_key(group, int(row["anchor_index"]))
        panel_path, _ = _resolve_panel(
            cast(list[Path], args.panel_root),
            scene_token=str(row["scene_token"]),
            expected_sha256=str(row["panel_artifact_sha256"]),
        )
        with np.load(panel_path, allow_pickle=False) as payload:
            descriptor = native_geometry_descriptor(
                np.asarray(payload["realized_local_poses"], dtype=np.float64),
                np.asarray(payload["challenger_eligible"], dtype=np.bool_),
            )
        identity_by_key[key] = dict(row)
        keys_by_group[group].append(key)
        group_by_key[key] = group
        descriptors[key] = descriptor
        descriptor_hashes[key] = descriptor_sha256(descriptor)
    primary_descriptor_hashes = {
        str(row["anchor"]): str(row["geometry_descriptor_sha256"])
        for row in cast(list[dict[str, Any]], primary["descriptor_ledger"])
    }
    if descriptor_hashes != primary_descriptor_hashes:
        raise ValueError(
            "recomputed descriptors differ from the frozen primary manifest"
        )

    primary_folds = {
        int(row["outer_fold"]): row
        for row in cast(list[dict[str, Any]], primary["folds"])
    }
    output_folds: list[dict[str, Any]] = []
    total_assignments = 0
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
        schemes: dict[str, Any] = {"S2_disjoint_geometry": {}, "S3_hash_random": {}}
        for stage_name, recipient_groups, donor_groups, require_permutation in specs:
            recipients = _keys_for_groups(recipient_groups, keys_by_group)
            donors = _keys_for_groups(donor_groups, keys_by_group)
            primary_rows = cast(
                list[dict[str, Any]],
                primary_folds[fold_index]["stages"][stage_name]["rows"],
            )
            forbidden = [
                (str(row["recipient"]), str(row["donor"])) for row in primary_rows
            ]
            for scheme_name, mode, salt in (
                (
                    "S2_disjoint_geometry",
                    "disjoint_geometry",
                    f"aorta-r005e2-donor-sensitivity-disjoint-v1:fold={fold_index}:stage={stage_name}",
                ),
                (
                    "S3_hash_random",
                    "hash_random",
                    f"{RANDOM_SALT}:fold={fold_index}:stage={stage_name}",
                ),
            ):
                assignment = globally_match_sensitivity(
                    recipients,
                    donors,
                    descriptor_by_key=descriptors,
                    group_by_key=group_by_key,
                    mode=mode,
                    salt=salt,
                    require_donor_permutation=require_permutation,
                    forbidden_edges=forbidden if mode == "disjoint_geometry" else (),
                )
                audit = audit_geometry_matching(
                    assignment,
                    expected_recipients=recipients,
                    allowed_donors=donors,
                    group_by_key=group_by_key,
                    require_donor_permutation=require_permutation,
                )
                assignment["rows"] = _enrich_rows(
                    cast(list[dict[str, Any]], assignment["rows"]), identity_by_key
                )
                assignment["audit"] = audit
                schemes[scheme_name][stage_name] = assignment
                total_assignments += int(assignment["recipient_count"])
        output_folds.append({"outer_fold": fold_index, "schemes": schemes})

    manifest: dict[str, Any] = {
        "schema": "aorta.r005e2.donor_sensitivity.v1",
        "status": "MATERIALIZED_BEFORE_SCIENTIFIC_OUTPUT",
        "scientific_or_visual_content_used": False,
        "development_calibration_test_navdream_target_accessed": False,
        "random_salt": RANDOM_SALT,
        "source_crossfit_manifest_sha256": sha256_file(args.crossfit_manifest),
        "source_panel_identity_roster_sha256": sha256_file(args.panel_identity_roster),
        "source_primary_manifest_sha256": sha256_file(args.primary_manifest),
        "outer_fold_count": len(output_folds),
        "scheme_count": 2,
        "total_assignment_count": total_assignments,
        "folds": output_folds,
    }
    manifest["manifest_sha256_without_self"] = canonical_sha256(manifest)
    _write_json(args.output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--panel-identity-roster", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2))
