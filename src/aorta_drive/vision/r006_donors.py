"""Outcome-free wrong-scene donor construction for formal R006V0."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from aorta_drive.protocol.visual_recoverability import canonical_sha256


def _tie_digest(salt: str, recipient: str, donor: str) -> str:
    if not salt:
        raise ValueError("donor salt must be nonempty")
    payload = f"{salt}\0{recipient}\0{donor}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    normalized: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tokens: set[str] = set()
    for source in rows:
        row = {
            "recording_group": str(source["recording_group"]),
            "anchor_index": int(source["anchor_index"]),
            "session_position": int(source["session_position"]),
            "scene_token": str(source["scene_token"]),
        }
        if not row["recording_group"] or not row["scene_token"]:
            raise ValueError("row identifiers must be nonempty")
        if row["scene_token"] in tokens:
            raise ValueError("scene tokens must be unique")
        tokens.add(row["scene_token"])
        normalized.append(row)
        by_group[row["recording_group"]].append(row)
    for group, group_rows in by_group.items():
        group_rows.sort(key=lambda row: (row["anchor_index"], row["scene_token"]))
        indices = [row["anchor_index"] for row in group_rows]
        if indices != list(range(len(group_rows))):
            raise ValueError(f"anchor indices are not contiguous in {group}")
        positions = [row["session_position"] for row in group_rows]
        lower, upper = min(positions), max(positions)
        if upper <= lower:
            raise ValueError(f"session positions do not span a range in {group}")
        for row in group_rows:
            row["normalized_session_position"] = (row["session_position"] - lower) / (
                upper - lower
            )
    normalized.sort(
        key=lambda row: (
            row["recording_group"],
            row["anchor_index"],
            row["scene_token"],
        )
    )
    return normalized, by_group


def _fold_roles(nested_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    for fold in nested_manifest["outer_folds"]:
        outer_fold = int(fold["outer_fold"])
        train = set(str(group) for group in fold["outer_train_groups"])
        for group in fold["outer_confirmation_groups"]:
            group_name = str(group)
            if group_name in roles:
                raise ValueError("confirmation group occurs in multiple folds")
            roles[group_name] = {"outer_fold": outer_fold, "outer_train": train}
    return roles


def build_wrong_scene_donor_manifest(
    rows: Sequence[Mapping[str, Any]],
    nested_manifest: Mapping[str, Any],
    *,
    salt: str,
) -> dict[str, Any]:
    """Choose one same-index, nearest-progress outer-train donor per anchor."""

    normalized, by_group = _normalized_rows(rows)
    roles = _fold_roles(nested_manifest)
    if set(by_group) != set(roles):
        raise ValueError("time-set groups disagree with nested confirmation roster")
    assignments = []
    for recipient in normalized:
        recipient_group = recipient["recording_group"]
        role = roles[recipient_group]
        candidates = [
            donor
            for donor_group in sorted(role["outer_train"])
            for donor in by_group[donor_group]
            if donor["anchor_index"] == recipient["anchor_index"]
        ]
        if not candidates:
            raise ValueError("no legal wrong-scene donor candidate")
        donor = min(
            candidates,
            key=lambda current: (
                abs(
                    current["normalized_session_position"]
                    - recipient["normalized_session_position"]
                ),
                _tie_digest(
                    salt,
                    recipient["scene_token"],
                    current["scene_token"],
                ),
                current["scene_token"],
            ),
        )
        assignments.append(
            {
                "outer_fold": role["outer_fold"],
                "recipient_group": recipient_group,
                "recipient_anchor_index": recipient["anchor_index"],
                "recipient_scene_token": recipient["scene_token"],
                "recipient_normalized_session_position": recipient[
                    "normalized_session_position"
                ],
                "donor_group": donor["recording_group"],
                "donor_anchor_index": donor["anchor_index"],
                "donor_scene_token": donor["scene_token"],
                "donor_normalized_session_position": donor[
                    "normalized_session_position"
                ],
                "normalized_session_position_gap": abs(
                    donor["normalized_session_position"]
                    - recipient["normalized_session_position"]
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema": "aorta.r006v0.wrong_scene_donors.v1",
        "outcomes_features_trajectories_used": False,
        "salt": salt,
        "matching_order": [
            "same_anchor_index",
            "nearest_normalized_session_position",
            "salted_hash",
            "scene_token",
        ],
        "runtime_appearance_binding": "same_condition_as_recipient",
        "runtime_patch_semantics": "donor_own_clean_minus_shift_delta",
        "runtime_energy_match": "recipient_A1_frobenius_per_anchor",
        "assignment_count": len(assignments),
        "assignments": assignments,
    }
    audit_wrong_scene_donor_manifest(
        manifest,
        rows=rows,
        nested_manifest=nested_manifest,
    )
    manifest["manifest_sha256_without_self"] = canonical_sha256(manifest)
    return manifest


def audit_wrong_scene_donor_manifest(
    manifest: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    nested_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless every assignment is the deterministic legal optimum."""

    if manifest.get("schema") != "aorta.r006v0.wrong_scene_donors.v1":
        raise ValueError("unexpected donor manifest schema")
    if manifest.get("outcomes_features_trajectories_used") is not False:
        raise ValueError("wrong-scene donor matching is not outcome-free")
    if manifest.get("runtime_appearance_binding") != "same_condition_as_recipient":
        raise ValueError("wrong-scene appearance binding changed")
    if manifest.get("runtime_patch_semantics") != "donor_own_clean_minus_shift_delta":
        raise ValueError("wrong-scene patch semantics changed")
    if manifest.get("runtime_energy_match") != "recipient_A1_frobenius_per_anchor":
        raise ValueError("wrong-scene energy matching changed")
    salt = str(manifest.get("salt", ""))
    normalized, by_group = _normalized_rows(rows)
    roles = _fold_roles(nested_manifest)
    row_by_token = {row["scene_token"]: row for row in normalized}
    assignments = [dict(row) for row in manifest.get("assignments", [])]
    if int(manifest.get("assignment_count", -1)) != len(normalized):
        raise ValueError("wrong-scene donor assignment count mismatch")
    recipient_counts = Counter(row["recipient_scene_token"] for row in assignments)
    if set(recipient_counts) != set(row_by_token) or any(
        count != 1 for count in recipient_counts.values()
    ):
        raise ValueError("recipient donor coverage is not exact")
    for assignment in assignments:
        recipient = row_by_token[assignment["recipient_scene_token"]]
        donor = row_by_token.get(assignment["donor_scene_token"])
        if donor is None:
            raise ValueError("unknown donor scene token")
        role = roles[recipient["recording_group"]]
        if assignment["recipient_group"] != recipient["recording_group"]:
            raise ValueError("recipient group field disagrees with token")
        if int(assignment["recipient_anchor_index"]) != recipient["anchor_index"]:
            raise ValueError("recipient anchor field disagrees with token")
        if assignment["donor_group"] != donor["recording_group"]:
            raise ValueError("donor group field disagrees with token")
        if int(assignment["donor_anchor_index"]) != donor["anchor_index"]:
            raise ValueError("donor anchor field disagrees with token")
        if int(assignment["outer_fold"]) != role["outer_fold"]:
            raise ValueError("donor outer fold mismatch")
        if donor["recording_group"] not in role["outer_train"]:
            raise ValueError("donor is not in recipient outer-train")
        if donor["recording_group"] == recipient["recording_group"]:
            raise ValueError("wrong-scene donor uses the recipient group")
        if donor["anchor_index"] != recipient["anchor_index"]:
            raise ValueError("wrong-scene donor anchor index mismatch")
        legal = [
            candidate
            for donor_group in sorted(role["outer_train"])
            for candidate in by_group[donor_group]
            if candidate["anchor_index"] == recipient["anchor_index"]
        ]
        expected = min(
            legal,
            key=lambda current: (
                abs(
                    current["normalized_session_position"]
                    - recipient["normalized_session_position"]
                ),
                _tie_digest(
                    salt,
                    recipient["scene_token"],
                    current["scene_token"],
                ),
                current["scene_token"],
            ),
        )
        if donor["scene_token"] != expected["scene_token"]:
            raise ValueError("wrong-scene donor is not the deterministic optimum")
    registered_hash = manifest.get("manifest_sha256_without_self")
    if registered_hash is not None:
        payload = dict(manifest)
        payload.pop("manifest_sha256_without_self")
        if registered_hash != canonical_sha256(payload):
            raise ValueError("wrong-scene donor manifest self-hash mismatch")
    return {
        "status": "PASS",
        "assignment_count": len(assignments),
        "recipient_coverage_exact": True,
        "donors_outer_train_only": True,
        "same_anchor_index": True,
        "outcome_free": True,
    }
