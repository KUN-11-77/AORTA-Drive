"""Independent mechanical audit of the R005E2 visual donor manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from aorta_drive.protocol.visual_mismatch import anchor_key
from aorta_drive.protocol.visual_recoverability import canonical_sha256
from aorta_drive.vision.feature_cache import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _keys(groups: list[str], keys_by_group: dict[str, list[str]]) -> set[str]:
    return {key for group in groups for key in keys_by_group[group]}


def _audit_stage(
    stage: dict[str, Any],
    *,
    expected_recipients: set[str],
    allowed_donors: set[str],
    group_by_key: dict[str, str],
    require_permutation: bool,
) -> dict[str, Any]:
    rows = cast(list[dict[str, Any]], stage["rows"])
    recipients = [str(row["recipient"]) for row in rows]
    donors = [str(row["donor"]) for row in rows]
    if set(recipients) != expected_recipients or len(recipients) != len(expected_recipients):
        raise ValueError("stage recipient roster mismatch")
    if len(recipients) != len(set(recipients)):
        raise ValueError("duplicate stage recipient")
    if len(donors) != len(set(donors)):
        raise ValueError("stage donor reused")
    if not set(donors) <= allowed_donors:
        raise ValueError("stage donor is outside the frozen training pool")
    if require_permutation and set(donors) != allowed_donors:
        raise ValueError("training visual donor marginal is not an exact permutation")
    for row in rows:
        recipient = str(row["recipient"])
        donor = str(row["donor"])
        if group_by_key[recipient] == group_by_key[donor]:
            raise ValueError("same-group visual donor detected")
        if str(row["recipient_recording_group"]) != group_by_key[recipient]:
            raise ValueError("recipient expanded identity mismatch")
        if str(row["donor_recording_group"]) != group_by_key[donor]:
            raise ValueError("donor expanded identity mismatch")
    distances = np.asarray(
        [float(row["standardized_squared_distance"]) for row in rows], dtype=np.float64
    )
    if not np.isfinite(distances).all() or bool(np.any(distances < 0.0)):
        raise ValueError("stage distance ledger is invalid")
    return {
        "recipient_count": len(recipients),
        "selected_donor_count": len(donors),
        "same_group_count": 0,
        "outside_allowed_pool_count": 0,
        "donor_reuse_count": 0,
        "donor_marginal_exact_permutation": require_permutation,
        "distance_mean": float(distances.mean()),
        "distance_max": float(distances.max()),
    }


def _verify_panel(
    roots: list[Path], *, scene_token: str, expected_sha256: str
) -> int:
    matching_indices: list[int] = []
    for index, root in enumerate(roots):
        path = root / f"{scene_token}.npz"
        if path.is_file() and sha256_file(path) == expected_sha256:
            matching_indices.append(index)
    if not matching_indices:
        raise ValueError("registered panel artifact is absent from all frozen roots")
    return matching_indices[0]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_hash = sha256_file(args.manifest)
    if args.expected_manifest_sha256 and manifest_hash != args.expected_manifest_sha256:
        raise ValueError("mismatch manifest file hash differs from the registered value")
    manifest = cast(dict[str, Any], json.loads(args.manifest.read_text(encoding="utf-8")))
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text(encoding="utf-8"))
    )
    roster = cast(
        dict[str, Any], json.loads(args.panel_identity_roster.read_text(encoding="utf-8"))
    )
    if manifest.get("utility_candidate_scores_images_dino_or_model_outputs_used") is not False:
        raise ValueError("manifest does not assert outcome/content-free matching")
    if manifest.get("development_calibration_test_navdream_target_accessed") is not False:
        raise ValueError("a sealed data role was accessed")
    if manifest["source_crossfit_manifest_sha256"] != sha256_file(args.crossfit_manifest):
        raise ValueError("crossfit source hash mismatch")
    if manifest["source_panel_identity_roster_sha256"] != sha256_file(
        args.panel_identity_roster
    ):
        raise ValueError("panel identity source hash mismatch")
    selfless = dict(manifest)
    claimed_self_hash = str(selfless.pop("manifest_sha256_without_self"))
    if canonical_sha256(selfless) != claimed_self_hash:
        raise ValueError("manifest canonical self-hash mismatch")

    identity_by_key: dict[str, dict[str, Any]] = {}
    keys_by_group: dict[str, list[str]] = {
        str(group): [] for group in cast(list[str], crossfit["groups"])
    }
    group_by_key: dict[str, str] = {}
    for row in cast(list[dict[str, Any]], roster["anchors"]):
        group = str(row["recording_group"])
        key = anchor_key(group, int(row["anchor_index"]))
        identity_by_key[key] = row
        keys_by_group[group].append(key)
        group_by_key[key] = group
    if len(identity_by_key) != args.expected_anchor_count:
        raise ValueError("panel identity anchor count mismatch")

    descriptor_rows = cast(list[dict[str, Any]], manifest["descriptor_ledger"])
    descriptor_by_key = {str(row["anchor"]): row for row in descriptor_rows}
    if set(descriptor_by_key) != set(identity_by_key):
        raise ValueError("descriptor ledger identity mismatch")
    panel_root_counts = [0 for _ in args.panel_root]
    for key, identity in identity_by_key.items():
        descriptor_row = descriptor_by_key[key]
        if descriptor_row["scene_token"] != identity["scene_token"]:
            raise ValueError("descriptor scene-token mismatch")
        if descriptor_row["panel_artifact_sha256"] != identity["panel_artifact_sha256"]:
            raise ValueError("descriptor panel hash mismatch")
        root_index = _verify_panel(
            cast(list[Path], args.panel_root),
            scene_token=str(identity["scene_token"]),
            expected_sha256=str(identity["panel_artifact_sha256"]),
        )
        if int(descriptor_row["panel_source_root_index"]) != root_index:
            raise ValueError("descriptor panel source-root index mismatch")
        panel_root_counts[root_index] += 1

    crossfit_folds = {
        int(row["outer_fold"]): row
        for row in cast(list[dict[str, Any]], crossfit["outer_folds"])
    }
    manifest_folds = {
        int(row["outer_fold"]): row
        for row in cast(list[dict[str, Any]], manifest["folds"])
    }
    if set(crossfit_folds) != set(manifest_folds):
        raise ValueError("outer-fold roster mismatch")
    stage_audits: list[dict[str, Any]] = []
    total_assignments = 0
    for fold_index in sorted(crossfit_folds):
        fold = crossfit_folds[fold_index]
        stages = cast(dict[str, dict[str, Any]], manifest_folds[fold_index]["stages"])
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
        if set(stages) != {spec[0] for spec in specs}:
            raise ValueError("stage names differ from the frozen protocol")
        for name, recipient_groups, donor_groups, require_permutation in specs:
            stage_audit = _audit_stage(
                stages[name],
                expected_recipients=_keys(recipient_groups, keys_by_group),
                allowed_donors=_keys(donor_groups, keys_by_group),
                group_by_key=group_by_key,
                require_permutation=require_permutation,
            )
            stage_audits.append(
                {"outer_fold": fold_index, "stage": name, **stage_audit}
            )
            total_assignments += int(stage_audit["recipient_count"])
    if total_assignments != int(manifest["total_assignment_count"]):
        raise ValueError("total assignment count mismatch")
    return {
        "schema": "aorta.r005e2.geometry_matched_visual_donors.audit.v1",
        "status": "PASS",
        "manifest_sha256": manifest_hash,
        "manifest_canonical_self_hash_exact": True,
        "crossfit_and_roster_source_hashes_exact": True,
        "panel_artifact_count": len(descriptor_rows),
        "panel_source_root_counts": panel_root_counts,
        "outer_fold_count": len(manifest_folds),
        "stage_count": len(stage_audits),
        "total_assignment_count": total_assignments,
        "all_recipients_exactly_covered_per_stage": True,
        "all_donors_from_registered_training_pool": True,
        "all_donors_cross_group": True,
        "all_donors_without_replacement_within_stage": True,
        "training_visual_marginals_exact_permutations": True,
        "utility_candidate_scores_images_dino_or_model_outputs_used": False,
        "development_calibration_test_navdream_target_accessed": False,
        "stage_audits": stage_audits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--panel-identity-roster", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, action="append", required=True)
    parser.add_argument("--expected-anchor-count", type=int, default=672)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = audit(arguments)
    _write_json(arguments.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "stage_audits"}, indent=2))
