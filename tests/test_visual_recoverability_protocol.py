from __future__ import annotations

import copy

import pytest

from aorta_drive.protocol.visual_recoverability import (
    audit_nested_group_manifest,
    balanced_group_folds,
    build_nested_group_manifest,
    canonical_sha256,
)


GROUPS = [f"group-{index:02d}" for index in range(42)]


def test_balanced_group_folds_are_deterministic_and_exact() -> None:
    first = balanced_group_folds(GROUPS, fold_count=6, salt="outer")
    second = balanced_group_folds(list(reversed(GROUPS)), fold_count=6, salt="outer")
    assert first == second
    assert sorted(first.values()).count(0) == 7
    assert {fold: list(first.values()).count(fold) for fold in range(6)} == {
        fold: 7 for fold in range(6)
    }


def test_nested_manifest_has_no_leakage_and_exact_outer_coverage() -> None:
    manifest = build_nested_group_manifest(
        GROUPS,
        outer_fold_count=6,
        inner_validation_count=7,
        outer_salt="outer",
        inner_salt="inner",
    )
    audit = audit_nested_group_manifest(manifest, expected_groups=GROUPS)
    assert audit["status"] == "PASS"
    assert audit["outer_test_size_min"] == 7
    assert audit["outer_test_size_max"] == 7
    for row in manifest["outer_folds"]:
        assert len(row["outer_train_groups"]) == 35
        assert len(row["inner_validation_groups"]) == 7
        assert len(row["inner_fit_groups"]) == 28


def test_manifest_hash_changes_with_salt_but_not_input_order() -> None:
    first = build_nested_group_manifest(
        GROUPS,
        outer_fold_count=6,
        inner_validation_count=7,
        outer_salt="outer-a",
        inner_salt="inner",
    )
    reordered = build_nested_group_manifest(
        list(reversed(GROUPS)),
        outer_fold_count=6,
        inner_validation_count=7,
        outer_salt="outer-a",
        inner_salt="inner",
    )
    changed = build_nested_group_manifest(
        GROUPS,
        outer_fold_count=6,
        inner_validation_count=7,
        outer_salt="outer-b",
        inner_salt="inner",
    )
    assert canonical_sha256(first) == canonical_sha256(reordered)
    assert first["group_outer_fold"] != changed["group_outer_fold"]


@pytest.mark.parametrize(
    "mutation",
    ["overlap", "missing", "wrong_inner_count", "outcome_flag"],
)
def test_manifest_audit_fails_closed(mutation: str) -> None:
    manifest = build_nested_group_manifest(
        GROUPS,
        outer_fold_count=6,
        inner_validation_count=7,
        outer_salt="outer",
        inner_salt="inner",
    )
    broken = copy.deepcopy(manifest)
    if mutation == "overlap":
        broken["outer_folds"][0]["outer_train_groups"].append(
            broken["outer_folds"][0]["outer_test_groups"][0]
        )
    elif mutation == "missing":
        broken["outer_folds"][0]["outer_test_groups"].pop()
    elif mutation == "wrong_inner_count":
        moved = broken["outer_folds"][0]["inner_validation_groups"].pop()
        broken["outer_folds"][0]["inner_fit_groups"].append(moved)
    else:
        broken["outcomes_or_features_used"] = True
    with pytest.raises(ValueError):
        audit_nested_group_manifest(broken, expected_groups=GROUPS)


def test_duplicate_groups_are_rejected() -> None:
    with pytest.raises(ValueError):
        balanced_group_folds(GROUPS + [GROUPS[0]], fold_count=6, salt="outer")
