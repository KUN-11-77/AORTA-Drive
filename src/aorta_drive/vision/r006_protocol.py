"""Leakage-resistant nested selection primitives for formal R006V0.

This module only receives recording-group identifiers and already-computed
coverage values.  It cannot inspect RGB, activations, trajectories, utilities,
or outer-confirmation endpoints while constructing roles or selecting the
registered interface.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from aorta_drive.protocol.visual_recoverability import (
    audit_nested_group_manifest,
    build_nested_group_manifest,
    canonical_sha256,
)


REGISTERED_LAYER_ORDER = (
    "planner_memory",
    "trajectory_query_secondary",
)


@dataclass(frozen=True)
class CoverageSummary:
    rank: int
    group_count: int
    group_equal_mean: float
    worst_group_mean: float
    qualifies: bool
    group_means: dict[str, float]


@dataclass(frozen=True)
class InterfaceSelection:
    layer: str
    rank: int
    group_equal_mean: float
    worst_group_mean: float
    rule: str


@dataclass(frozen=True)
class CrossFoldInterfaceDecision:
    status: str
    layer: str | None
    rank: int | None
    layer_counts: dict[str, int]
    rank_counts_for_consensus_layer: dict[int, int]
    selected_fold_count: int


def build_r006_nested_manifest(
    groups: Sequence[str],
    *,
    outer_salt: str,
    inner_salt: str,
    outer_fold_count: int = 6,
    selection_group_count: int = 7,
) -> dict[str, Any]:
    """Build the frozen 7-confirmation/28-discovery/7-selection roles."""

    base = build_nested_group_manifest(
        groups,
        outer_fold_count=outer_fold_count,
        inner_validation_count=selection_group_count,
        outer_salt=outer_salt,
        inner_salt=inner_salt,
    )
    folds = []
    for row in base["outer_folds"]:
        folds.append(
            {
                "outer_fold": row["outer_fold"],
                "outer_confirmation_groups": row["outer_test_groups"],
                "outer_train_groups": row["outer_train_groups"],
                "basis_discovery_groups": row["inner_fit_groups"],
                "layer_rank_selection_groups": row["inner_validation_groups"],
            }
        )
    manifest: dict[str, Any] = {
        "schema": "aorta.r006v0.nested_mediation.v1",
        "hash_algorithm": "sha256",
        "outcomes_or_features_used": False,
        "group_count": len(base["groups"]),
        "groups": base["groups"],
        "outer_fold_count": outer_fold_count,
        "outer_confirmation_count": len(folds[0]["outer_confirmation_groups"]),
        "selection_group_count": selection_group_count,
        "discovery_group_count": len(folds[0]["basis_discovery_groups"]),
        "outer_salt": outer_salt,
        "inner_salt": inner_salt,
        "group_outer_fold": base["group_outer_fold"],
        "outer_folds": folds,
    }
    audit_r006_nested_manifest(manifest, expected_groups=groups)
    manifest["manifest_sha256_without_self"] = canonical_sha256(manifest)
    return manifest


def audit_r006_nested_manifest(
    manifest: Mapping[str, Any], *, expected_groups: Sequence[str]
) -> dict[str, Any]:
    """Reject omissions, overlaps, imbalanced folds, or role contamination."""

    if manifest.get("schema") != "aorta.r006v0.nested_mediation.v1":
        raise ValueError("unexpected R006 nested manifest schema")
    if not manifest.get("outer_salt") or not manifest.get("inner_salt"):
        raise ValueError("R006 nested salts must be nonempty")
    expected = sorted(str(group) for group in expected_groups)
    if int(manifest.get("group_count", -1)) != len(expected):
        raise ValueError("R006 nested group count mismatch")
    translated = {
        "outcomes_or_features_used": manifest.get("outcomes_or_features_used"),
        "groups": manifest.get("groups"),
        "outer_fold_count": manifest.get("outer_fold_count"),
        "inner_validation_count": manifest.get("selection_group_count"),
        "outer_folds": [
            {
                "outer_fold": row["outer_fold"],
                "outer_test_groups": row["outer_confirmation_groups"],
                "outer_train_groups": row["outer_train_groups"],
                "inner_validation_groups": row["layer_rank_selection_groups"],
                "inner_fit_groups": row["basis_discovery_groups"],
            }
            for row in manifest.get("outer_folds", [])
        ],
    }
    base_audit = audit_nested_group_manifest(
        translated,
        expected_groups=expected_groups,
    )
    rows = list(manifest["outer_folds"])
    observed_assignment: dict[str, int] = {}
    for row in rows:
        if len(row["outer_confirmation_groups"]) != int(
            manifest["outer_confirmation_count"]
        ):
            raise ValueError("outer confirmation count mismatch")
        if len(row["basis_discovery_groups"]) != int(manifest["discovery_group_count"]):
            raise ValueError("basis discovery count mismatch")
        for group in row["outer_confirmation_groups"]:
            observed_assignment[str(group)] = int(row["outer_fold"])
    registered_assignment = {
        str(group): int(fold)
        for group, fold in dict(manifest.get("group_outer_fold", {})).items()
    }
    if observed_assignment != registered_assignment:
        raise ValueError("group_outer_fold disagrees with confirmation roles")
    registered_hash = manifest.get("manifest_sha256_without_self")
    if registered_hash is not None:
        payload = dict(manifest)
        payload.pop("manifest_sha256_without_self")
        if registered_hash != canonical_sha256(payload):
            raise ValueError("R006 nested manifest self-hash mismatch")
    return {
        **base_audit,
        "schema": "aorta.r006v0.nested_mediation.audit.v1",
        "selection_group_count": int(manifest["selection_group_count"]),
        "discovery_group_count": int(manifest["discovery_group_count"]),
    }


def summarize_basis_coverage(
    total_task_drift_sq: torch.Tensor,
    uncovered_task_drift_sq: torch.Tensor,
    group_ids: Sequence[str],
    *,
    rank: int,
    epsilon: float = 1e-12,
    mean_threshold: float = 0.90,
    worst_group_threshold: float = 0.50,
) -> CoverageSummary:
    """Compute unclipped group-equal coverage and its qualification decision."""

    total = torch.as_tensor(total_task_drift_sq, dtype=torch.float64).flatten()
    uncovered = torch.as_tensor(uncovered_task_drift_sq, dtype=torch.float64).flatten()
    if total.shape != uncovered.shape or total.numel() != len(group_ids):
        raise ValueError("coverage arrays and group_ids must have equal length")
    if total.numel() == 0:
        raise ValueError("coverage input must be nonempty")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not torch.isfinite(total).all() or not torch.isfinite(uncovered).all():
        raise ValueError("coverage inputs must be finite")
    if torch.any(total < 0) or torch.any(uncovered < 0):
        raise ValueError("squared drift inputs must be nonnegative")
    normalized_groups = [str(group) for group in group_ids]
    if any(not group for group in normalized_groups):
        raise ValueError("group identifiers must be nonempty")

    coverage = 1.0 - uncovered / (total + float(epsilon))
    group_means: dict[str, float] = {}
    for group in sorted(set(normalized_groups)):
        indices = [
            index for index, current in enumerate(normalized_groups) if current == group
        ]
        group_means[group] = float(coverage[indices].mean())
    group_values = list(group_means.values())
    group_equal_mean = float(sum(group_values) / len(group_values))
    worst_group_mean = float(min(group_values))
    return CoverageSummary(
        rank=rank,
        group_count=len(group_means),
        group_equal_mean=group_equal_mean,
        worst_group_mean=worst_group_mean,
        qualifies=(
            group_equal_mean >= mean_threshold
            and worst_group_mean >= worst_group_threshold
        ),
        group_means=group_means,
    )


def select_registered_interface(
    summaries: Mapping[str, Mapping[int, CoverageSummary]],
) -> InterfaceSelection:
    """Select one layer/rank without consulting outer-confirmation endpoints.

    Within each layer choose the smallest qualifying rank.  Across layers use
    the smallest rank, then the strongest worst-group coverage, then the
    strongest group-equal mean, and finally the preregistered layer order.
    """

    candidates: list[tuple[str, CoverageSummary]] = []
    unknown_layers = set(summaries) - set(REGISTERED_LAYER_ORDER)
    if unknown_layers:
        raise ValueError(f"unregistered layers: {sorted(unknown_layers)}")
    for layer in REGISTERED_LAYER_ORDER:
        layer_summaries = summaries.get(layer, {})
        qualified = [
            summary
            for rank, summary in layer_summaries.items()
            if rank == summary.rank and summary.qualifies
        ]
        if qualified:
            candidates.append((layer, min(qualified, key=lambda item: item.rank)))
    if not candidates:
        raise ValueError("no registered layer/rank satisfies coverage gates")
    order = {layer: index for index, layer in enumerate(REGISTERED_LAYER_ORDER)}
    layer, summary = min(
        candidates,
        key=lambda item: (
            item[1].rank,
            -item[1].worst_group_mean,
            -item[1].group_equal_mean,
            order[item[0]],
        ),
    )
    return InterfaceSelection(
        layer=layer,
        rank=summary.rank,
        group_equal_mean=summary.group_equal_mean,
        worst_group_mean=summary.worst_group_mean,
        rule=(
            "smallest_qualifying_rank_then_worst_group_then_group_equal_mean_"
            "then_registered_layer_order"
        ),
    )


def selection_as_dict(selection: InterfaceSelection) -> dict[str, Any]:
    """Return a JSON-compatible representation for frozen manifests."""

    return asdict(selection)


def aggregate_cross_fold_interfaces(
    selections: Sequence[InterfaceSelection | None],
    *,
    expected_fold_count: int = 6,
    minimum_layer_count: int = 5,
    minimum_rank_count: int = 4,
) -> CrossFoldInterfaceDecision:
    """Apply the preregistered six-fold deployment-interface consensus rule.

    Every fold must first produce a qualified interface.  The layer must have a
    unique mode present in at least five folds.  Only folds selecting that layer
    contribute to the rank mode; its count is still measured against all six
    folds and must be at least four.  No outcome or outer-confirmation endpoint
    enters this aggregation.
    """

    if len(selections) != expected_fold_count:
        raise ValueError("cross-fold interface roster must contain every outer fold")
    selected = [item for item in selections if item is not None]
    layer_counts = Counter(item.layer for item in selected)
    if len(selected) != expected_fold_count:
        return CrossFoldInterfaceDecision(
            status="STOP_GLOBAL_SHARED_INTERFACE",
            layer=None,
            rank=None,
            layer_counts=dict(sorted(layer_counts.items())),
            rank_counts_for_consensus_layer={},
            selected_fold_count=len(selected),
        )

    highest_layer_count = max(layer_counts.values(), default=0)
    layer_modes = sorted(
        layer for layer, count in layer_counts.items() if count == highest_layer_count
    )
    if highest_layer_count < minimum_layer_count or len(layer_modes) != 1:
        return CrossFoldInterfaceDecision(
            status="STOP_GLOBAL_SHARED_INTERFACE",
            layer=None,
            rank=None,
            layer_counts=dict(sorted(layer_counts.items())),
            rank_counts_for_consensus_layer={},
            selected_fold_count=len(selected),
        )

    consensus_layer = layer_modes[0]
    rank_counts = Counter(
        item.rank for item in selected if item.layer == consensus_layer
    )
    highest_rank_count = max(rank_counts.values(), default=0)
    rank_modes = sorted(
        rank for rank, count in rank_counts.items() if count == highest_rank_count
    )
    if highest_rank_count < minimum_rank_count or len(rank_modes) != 1:
        return CrossFoldInterfaceDecision(
            status="STOP_GLOBAL_SHARED_INTERFACE",
            layer=consensus_layer,
            rank=None,
            layer_counts=dict(sorted(layer_counts.items())),
            rank_counts_for_consensus_layer=dict(sorted(rank_counts.items())),
            selected_fold_count=len(selected),
        )

    return CrossFoldInterfaceDecision(
        status="PASS_GLOBAL_SHARED_INTERFACE",
        layer=consensus_layer,
        rank=rank_modes[0],
        layer_counts=dict(sorted(layer_counts.items())),
        rank_counts_for_consensus_layer=dict(sorted(rank_counts.items())),
        selected_fold_count=len(selected),
    )
