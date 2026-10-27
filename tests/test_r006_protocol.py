from __future__ import annotations

import copy

import pytest
import torch

from aorta_drive.vision.r006_protocol import (
    CoverageSummary,
    InterfaceSelection,
    aggregate_cross_fold_interfaces,
    audit_r006_nested_manifest,
    build_r006_nested_manifest,
    select_registered_interface,
    summarize_basis_coverage,
)


GROUPS = [f"group-{index:02d}" for index in range(42)]


def _summary(rank: int, mean: float, worst: float, qualifies: bool) -> CoverageSummary:
    return CoverageSummary(
        rank=rank,
        group_count=7,
        group_equal_mean=mean,
        worst_group_mean=worst,
        qualifies=qualifies,
        group_means={f"group-{index}": mean for index in range(7)},
    )


def test_r006_manifest_freezes_28_7_7_roles_without_overlap() -> None:
    manifest = build_r006_nested_manifest(
        GROUPS,
        outer_salt="aorta-r006v0-outer-v1",
        inner_salt="aorta-r006v0-inner-selection-v1",
    )
    audit = audit_r006_nested_manifest(manifest, expected_groups=GROUPS)
    assert audit["status"] == "PASS"
    assert manifest["outcomes_or_features_used"] is False
    for fold in manifest["outer_folds"]:
        assert len(fold["outer_confirmation_groups"]) == 7
        assert len(fold["basis_discovery_groups"]) == 28
        assert len(fold["layer_rank_selection_groups"]) == 7
        assert not (
            set(fold["outer_confirmation_groups"]) & set(fold["outer_train_groups"])
        )
        assert not (
            set(fold["basis_discovery_groups"])
            & set(fold["layer_rank_selection_groups"])
        )


def test_r006_manifest_is_order_invariant_and_fails_on_role_leakage() -> None:
    first = build_r006_nested_manifest(
        GROUPS,
        outer_salt="aorta-r006v0-outer-v1",
        inner_salt="aorta-r006v0-inner-selection-v1",
    )
    reordered = build_r006_nested_manifest(
        list(reversed(GROUPS)),
        outer_salt="aorta-r006v0-outer-v1",
        inner_salt="aorta-r006v0-inner-selection-v1",
    )
    assert first == reordered
    broken = copy.deepcopy(first)
    leaked = broken["outer_folds"][0]["outer_confirmation_groups"][0]
    broken["outer_folds"][0]["basis_discovery_groups"].append(leaked)
    with pytest.raises(ValueError):
        audit_r006_nested_manifest(broken, expected_groups=GROUPS)
    bad_hash = copy.deepcopy(first)
    bad_hash["manifest_sha256_without_self"] = "0" * 64
    with pytest.raises(ValueError, match="self-hash"):
        audit_r006_nested_manifest(bad_hash, expected_groups=GROUPS)


def test_coverage_is_group_equal_and_not_clipped() -> None:
    summary = summarize_basis_coverage(
        total_task_drift_sq=torch.tensor([1.0, 1.0, 1.0]),
        uncovered_task_drift_sq=torch.tensor([0.0, 0.0, 2.0]),
        group_ids=["large", "large", "small"],
        rank=4,
    )
    assert summary.group_means["large"] == pytest.approx(1.0)
    assert summary.group_means["small"] == pytest.approx(-1.0)
    assert summary.group_equal_mean == pytest.approx(0.0, abs=2e-12)
    assert summary.worst_group_mean == pytest.approx(-1.0)
    assert summary.qualifies is False


def test_interface_selection_uses_registered_complexity_and_robustness_order() -> None:
    selection = select_registered_interface(
        {
            "planner_memory": {
                4: _summary(4, 0.89, 0.70, False),
                8: _summary(8, 0.93, 0.60, True),
                16: _summary(16, 0.97, 0.80, True),
            },
            "trajectory_query_secondary": {
                4: _summary(4, 0.92, 0.55, True),
                8: _summary(8, 0.99, 0.95, True),
            },
        }
    )
    assert selection.layer == "trajectory_query_secondary"
    assert selection.rank == 4


def test_interface_selection_uses_worst_group_then_fixed_layer_order() -> None:
    robust = select_registered_interface(
        {
            "planner_memory": {4: _summary(4, 0.99, 0.51, True)},
            "trajectory_query_secondary": {4: _summary(4, 0.91, 0.70, True)},
        }
    )
    assert robust.layer == "trajectory_query_secondary"
    tied = select_registered_interface(
        {
            "planner_memory": {4: _summary(4, 0.91, 0.70, True)},
            "trajectory_query_secondary": {4: _summary(4, 0.91, 0.70, True)},
        }
    )
    assert tied.layer == "planner_memory"


def test_interface_selection_fails_closed_without_qualified_candidate() -> None:
    with pytest.raises(ValueError, match="no registered layer/rank"):
        select_registered_interface(
            {
                "planner_memory": {4: _summary(4, 0.80, 0.40, False)},
                "trajectory_query_secondary": {},
            }
        )


def _selection(layer: str, rank: int) -> InterfaceSelection:
    return InterfaceSelection(
        layer=layer,
        rank=rank,
        group_equal_mean=0.95,
        worst_group_mean=0.75,
        rule="registered",
    )


def test_cross_fold_interface_requires_unique_rank_in_at_least_four_folds() -> None:
    stopped = aggregate_cross_fold_interfaces(
        [
            _selection("trajectory_query_secondary", rank)
            for rank in (8, 32, 4, 32, 4, 32)
        ]
    )
    assert stopped.status == "STOP_GLOBAL_SHARED_INTERFACE"
    assert stopped.layer == "trajectory_query_secondary"
    assert stopped.rank is None
    assert stopped.layer_counts == {"trajectory_query_secondary": 6}
    assert stopped.rank_counts_for_consensus_layer == {4: 2, 8: 1, 32: 3}

    passed = aggregate_cross_fold_interfaces(
        [
            _selection("trajectory_query_secondary", rank)
            for rank in (8, 32, 32, 32, 4, 32)
        ]
    )
    assert passed.status == "PASS_GLOBAL_SHARED_INTERFACE"
    assert passed.layer == "trajectory_query_secondary"
    assert passed.rank == 32


def test_cross_fold_interface_fails_when_any_fold_has_no_qualified_interface() -> None:
    decision = aggregate_cross_fold_interfaces(
        [
            _selection("trajectory_query_secondary", 8),
            _selection("trajectory_query_secondary", 8),
            _selection("trajectory_query_secondary", 8),
            _selection("trajectory_query_secondary", 8),
            _selection("trajectory_query_secondary", 8),
            None,
        ]
    )
    assert decision.status == "STOP_GLOBAL_SHARED_INTERFACE"
    assert decision.selected_fold_count == 5
    assert decision.rank is None
