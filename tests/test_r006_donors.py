from __future__ import annotations

import copy

import pytest

from aorta_drive.vision.r006_donors import (
    audit_wrong_scene_donor_manifest,
    build_wrong_scene_donor_manifest,
)
from aorta_drive.vision.r006_protocol import build_r006_nested_manifest


GROUPS = [f"group-{index:02d}" for index in range(12)]


def _rows() -> list[dict[str, object]]:
    rows = []
    for group_index, group in enumerate(GROUPS):
        for anchor_index in range(16):
            rows.append(
                {
                    "recording_group": group,
                    "anchor_index": anchor_index,
                    "session_position": 10 * anchor_index + group_index,
                    "scene_token": f"{group}-token-{anchor_index:02d}",
                }
            )
    return rows


def _nested() -> dict[str, object]:
    return build_r006_nested_manifest(
        GROUPS,
        outer_salt="outer",
        inner_salt="inner",
        outer_fold_count=3,
        selection_group_count=2,
    )


def test_wrong_scene_donors_are_deterministic_and_outer_train_only() -> None:
    rows = _rows()
    nested = _nested()
    first = build_wrong_scene_donor_manifest(rows, nested, salt="donor-v1")
    reordered = build_wrong_scene_donor_manifest(
        list(reversed(rows)), nested, salt="donor-v1"
    )
    assert first == reordered
    audit = audit_wrong_scene_donor_manifest(
        first,
        rows=rows,
        nested_manifest=nested,
    )
    assert audit["status"] == "PASS"
    assert audit["assignment_count"] == 12 * 16


def test_wrong_scene_donor_manifest_rejects_role_leakage_and_hash_changes() -> None:
    rows = _rows()
    nested = _nested()
    manifest = build_wrong_scene_donor_manifest(rows, nested, salt="donor-v1")
    broken = copy.deepcopy(manifest)
    assignment = broken["assignments"][0]
    assignment["donor_group"] = assignment["recipient_group"]
    assignment["donor_scene_token"] = assignment["recipient_scene_token"]
    with pytest.raises(ValueError):
        audit_wrong_scene_donor_manifest(
            broken,
            rows=rows,
            nested_manifest=nested,
        )
    bad_hash = copy.deepcopy(manifest)
    bad_hash["manifest_sha256_without_self"] = "0" * 64
    with pytest.raises(ValueError, match="self-hash"):
        audit_wrong_scene_donor_manifest(
            bad_hash,
            rows=rows,
            nested_manifest=nested,
        )
