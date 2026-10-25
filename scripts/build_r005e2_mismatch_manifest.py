"""Build the label-free, fold-isolated visual mismatch donor manifest."""

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
    geometry_descriptor_names,
    globally_match_geometry,
    native_geometry_descriptor,
)
from aorta_drive.protocol.visual_recoverability import canonical_sha256
from aorta_drive.vision.feature_cache import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _keys_for_groups(groups: list[str], keys_by_group: dict[str, list[str]]) -> list[str]:
    return sorted(key for group in groups for key in keys_by_group[group])


def _enrich_rows(
    rows: list[dict[str, Any]], identity_by_key: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        recipient = identity_by_key[str(row["recipient"])]
        donor = identity_by_key[str(row["donor"])]
        enriched.append(
            {
                **row,
                "recipient_recording_group": recipient["recording_group"],
                "recipient_anchor_index": recipient["anchor_index"],
                "recipient_scene_token": recipient["scene_token"],
                "donor_recording_group": donor["recording_group"],
                "donor_anchor_index": donor["anchor_index"],
                "donor_scene_token": donor["scene_token"],
            }
        )
    return enriched


def run(args: argparse.Namespace) -> dict[str, Any]:
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text(encoding="utf-8"))
    )
    roster = cast(
        dict[str, Any], json.loads(args.panel_identity_roster.read_text(encoding="utf-8"))
    )
    if roster.get("utility_candidate_or_visual_fields_used") is not False:
        raise ValueError("panel identity roster is not outcome-free")
    if args.expected_crossfit_sha256 and sha256_file(args.crossfit_manifest) != args.expected_crossfit_sha256:
        raise ValueError("crossfit manifest hash mismatch")
    if args.expected_roster_sha256 and sha256_file(args.panel_identity_roster) != args.expected_roster_sha256:
        raise ValueError("panel identity roster hash mismatch")

    identity_by_key: dict[str, dict[str, Any]] = {}
    keys_by_group: dict[str, list[str]] = {
        str(group): [] for group in cast(list[str], crossfit["groups"])
    }
    group_by_key: dict[str, str] = {}
    descriptors: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    descriptor_ledger: list[dict[str, Any]] = []
    for row in cast(list[dict[str, Any]], roster["anchors"]):
        group = str(row["recording_group"])
        index = int(row["anchor_index"])
        key = anchor_key(group, index)
        if key in identity_by_key or group not in keys_by_group:
            raise ValueError("panel identity roster contains an invalid or duplicate anchor")
        panel_path = args.panel_root / f"{row['scene_token']}.npz"
        panel_hash = sha256_file(panel_path)
        if panel_hash != str(row["panel_artifact_sha256"]):
            raise ValueError("panel artifact hash mismatch")
        with np.load(panel_path, allow_pickle=False) as panel_payload:
            descriptor = native_geometry_descriptor(
                np.asarray(panel_payload["realized_local_poses"], dtype=np.float64),
                np.asarray(panel_payload["challenger_eligible"], dtype=np.bool_),
            )
        identity_by_key[key] = dict(row)
        keys_by_group[group].append(key)
        group_by_key[key] = group
        descriptors[key] = descriptor
        descriptor_ledger.append(
            {
                "anchor": key,
                "scene_token": str(row["scene_token"]),
                "panel_artifact_sha256": panel_hash,
                "geometry_descriptor_sha256": descriptor_sha256(descriptor),
            }
        )
    if len(identity_by_key) != 672 or any(len(values) != 16 for values in keys_by_group.values()):
        raise ValueError("panel identity roster must contain 42 groups by 16 anchors")

    fold_rows: list[dict[str, Any]] = []
    total_assignments = 0
    for fold_any in cast(list[dict[str, Any]], crossfit["outer_folds"]):
        fold = dict(fold_any)
        fold_index = int(fold["outer_fold"])
        stage_specs = (
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
        stages: dict[str, Any] = {}
        for stage_name, recipient_groups, donor_groups, require_permutation in stage_specs:
            recipients = _keys_for_groups(recipient_groups, keys_by_group)
            donors = _keys_for_groups(donor_groups, keys_by_group)
            assignment = globally_match_geometry(
                recipients,
                donors,
                descriptor_by_key=descriptors,
                group_by_key=group_by_key,
                tie_salt=f"aorta-r005e2-mismatch-v1:fold={fold_index}:stage={stage_name}",
                require_donor_permutation=require_permutation,
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
            stages[stage_name] = assignment
            total_assignments += int(assignment["recipient_count"])
        fold_rows.append({"outer_fold": fold_index, "stages": stages})

    manifest: dict[str, Any] = {
        "schema": "aorta.r005e2.geometry_matched_visual_donors.v1",
        "matching_objective": (
            "minimum_total_standardized_native_geometry_distance_subject_to_"
            "cross_group_without_replacement_constraints"
        ),
        "matching_inputs": "native_realized_geometry_and_legal_domain_cardinality_only",
        "utility_candidate_scores_images_dino_or_model_outputs_used": False,
        "development_calibration_test_navdream_target_accessed": False,
        "descriptor_names": list(geometry_descriptor_names()),
        "descriptor_dimension": len(geometry_descriptor_names()),
        "source_crossfit_manifest_sha256": sha256_file(args.crossfit_manifest),
        "source_panel_identity_roster_sha256": sha256_file(args.panel_identity_roster),
        "panel_artifact_count": len(descriptor_ledger),
        "descriptor_ledger": sorted(descriptor_ledger, key=lambda row: row["anchor"]),
        "outer_fold_count": len(fold_rows),
        "total_assignment_count": total_assignments,
        "folds": fold_rows,
        "audit": {
            "status": "PASS",
            "all_stages_fold_isolated": True,
            "all_donors_cross_group": True,
            "all_donors_without_replacement_within_stage": True,
            "training_visual_marginals_exact_permutations": True,
            "labels_or_visual_content_used": False,
        },
    }
    manifest["manifest_sha256_without_self"] = canonical_sha256(manifest)
    _write_json(args.output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--panel-identity-roster", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--expected-crossfit-sha256")
    parser.add_argument("--expected-roster-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(
        json.dumps(
            {key: value for key, value in output.items() if key not in {"descriptor_ledger", "folds"}},
            indent=2,
        )
    )
