"""Outcome-free split primitives for the R005E2 recoverability study.

The functions in this module accept recording-group identifiers only.  They do
not accept labels, candidate utilities, visual features, or model outcomes, so
the registered outer and inner roles cannot be stratified on scientific results.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def _digest(salt: str, value: str) -> str:
    if not salt:
        raise ValueError("salt must be nonempty")
    if not value:
        raise ValueError("recording-group identifier must be nonempty")
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 for a JSON-compatible manifest."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def balanced_group_folds(
    groups: Sequence[str],
    *,
    fold_count: int,
    salt: str,
) -> dict[str, int]:
    """Assign unique group identifiers to balanced folds using salted hashes."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    normalized = [str(group) for group in groups]
    if len(normalized) != len(set(normalized)):
        raise ValueError("recording groups must be unique")
    if len(normalized) < fold_count:
        raise ValueError("there must be at least one group per fold")
    ordered = sorted(normalized, key=lambda group: (_digest(salt, group), group))
    return {group: index % fold_count for index, group in enumerate(ordered)}


def build_nested_group_manifest(
    groups: Sequence[str],
    *,
    outer_fold_count: int,
    inner_validation_count: int,
    outer_salt: str,
    inner_salt: str,
) -> dict[str, Any]:
    """Build a deterministic outer-cross-fit and inner-validation manifest."""

    normalized = sorted(str(group) for group in groups)
    outer_assignment = balanced_group_folds(
        normalized,
        fold_count=outer_fold_count,
        salt=outer_salt,
    )
    outer_rows: list[dict[str, Any]] = []
    for outer_fold in range(outer_fold_count):
        test_groups = sorted(
            group for group in normalized if outer_assignment[group] == outer_fold
        )
        train_groups = sorted(set(normalized) - set(test_groups))
        if not 0 < inner_validation_count < len(train_groups):
            raise ValueError("inner_validation_count must split outer-train groups")
        inner_order = sorted(
            train_groups,
            key=lambda group: (
                _digest(f"{inner_salt}:outer={outer_fold}", group),
                group,
            ),
        )
        inner_validation_groups = sorted(inner_order[:inner_validation_count])
        inner_fit_groups = sorted(inner_order[inner_validation_count:])
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "outer_test_groups": test_groups,
                "outer_train_groups": train_groups,
                "inner_validation_groups": inner_validation_groups,
                "inner_fit_groups": inner_fit_groups,
            }
        )

    manifest: dict[str, Any] = {
        "schema": "aorta.r005e2.nested_group_crossfit.v1",
        "hash_algorithm": "sha256",
        "outcomes_or_features_used": False,
        "outer_fold_count": outer_fold_count,
        "inner_validation_count": inner_validation_count,
        "outer_salt": outer_salt,
        "inner_salt": inner_salt,
        "group_count": len(normalized),
        "groups": normalized,
        "group_outer_fold": {
            group: outer_assignment[group] for group in normalized
        },
        "outer_folds": outer_rows,
    }
    audit_nested_group_manifest(manifest, expected_groups=normalized)
    manifest["manifest_sha256_without_self"] = canonical_sha256(manifest)
    return manifest


def audit_nested_group_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_groups: Sequence[str],
) -> dict[str, Any]:
    """Fail closed on overlap, omission, imbalance, or role inconsistency."""

    expected = sorted(str(group) for group in expected_groups)
    if len(expected) != len(set(expected)):
        raise ValueError("expected_groups contains duplicates")
    if manifest.get("outcomes_or_features_used") is not False:
        raise ValueError("manifest is not outcome-free")
    groups = sorted(str(group) for group in manifest.get("groups", []))
    if groups != expected:
        raise ValueError("manifest group roster differs from expected roster")
    fold_count = int(manifest["outer_fold_count"])
    rows = list(manifest.get("outer_folds", []))
    if len(rows) != fold_count:
        raise ValueError("outer fold row count mismatch")

    seen_test: list[str] = []
    test_sizes: list[int] = []
    for expected_fold, row_any in enumerate(rows):
        row = dict(row_any)
        if int(row["outer_fold"]) != expected_fold:
            raise ValueError("outer fold rows are not canonical")
        test = set(str(group) for group in row["outer_test_groups"])
        train = set(str(group) for group in row["outer_train_groups"])
        inner_validation = set(
            str(group) for group in row["inner_validation_groups"]
        )
        inner_fit = set(str(group) for group in row["inner_fit_groups"])
        if test & train:
            raise ValueError("outer train/test overlap")
        if sorted(test | train) != expected:
            raise ValueError("outer train/test do not cover the roster")
        if inner_validation & inner_fit:
            raise ValueError("inner fit/validation overlap")
        if inner_validation | inner_fit != train:
            raise ValueError("inner roles do not partition outer train")
        if len(inner_validation) != int(manifest["inner_validation_count"]):
            raise ValueError("inner validation size mismatch")
        seen_test.extend(test)
        test_sizes.append(len(test))

    counts = Counter(seen_test)
    if set(counts) != set(expected) or any(count != 1 for count in counts.values()):
        raise ValueError("each group must be outer-test exactly once")
    if max(test_sizes) - min(test_sizes) > 1:
        raise ValueError("outer folds are not balanced")
    return {
        "status": "PASS",
        "group_count": len(expected),
        "outer_fold_count": fold_count,
        "outer_test_size_min": min(test_sizes),
        "outer_test_size_max": max(test_sizes),
        "each_group_outer_test_exactly_once": True,
        "outcomes_or_features_used": False,
    }
