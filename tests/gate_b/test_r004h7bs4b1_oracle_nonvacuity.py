"""Decision-gate tests for the H7BS4B1 oracle non-vacuity experiment."""

from __future__ import annotations

from typing import Any

from run_r004h7bs4b1_oracle_nonvacuity import _summarize


def _variant(sign_identified: bool, width: float) -> dict[str, Any]:
    return {
        "lower_m": 0.1 if sign_identified else -0.1,
        "upper_m": 0.2,
        "width_m": width,
        "sign_identified": sign_identified,
        "positive_identified": sign_identified,
        "negative_identified": False,
    }


def _row(group: str, *, joint: bool, box: bool) -> dict[str, Any]:
    return {
        "recording_group": group,
        "anchor_index": 0,
        "maximum_numerical_widening": 0.0,
        "teacher_included": True,
        "pair_intervals": [
            {
                "joint_lipschitz": _variant(joint, 0.2),
                "shared_node_box": _variant(box, 0.4),
                "candidate_independent": _variant(False, 0.6),
                "all_variants_contain_teacher": True,
                "joint_subset_of_shared_box_numerically": True,
                "shared_box_subset_of_candidate_independent_numerically": True,
            }
        ],
        "selection": {
            variant: {
                "intervened": False,
                "teacher_regret_m": 0.0,
                "harmful_intervention": False,
            }
            for variant in (
                "joint_lipschitz",
                "shared_node_box",
                "candidate_independent",
            )
        },
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _summarize(
        rows,
        group_names=("g0", "g1"),
        anchors_per_group=1,
        minimum_anchors_per_group=1,
        numerical_widening_limit_m=1e-9,
        strict_improvement_lower_bound=0.0,
        bootstrap_seed=20261025,
        bootstrap_resamples=100,
    )


def test_joint_nonvacuity_requires_strict_group_level_gain() -> None:
    result = _summary([_row("g0", joint=True, box=False), _row("g1", joint=True, box=False)])
    assert result["decision"] == "JOINT_LIPSCHITZ_ORACLE_NONVACUITY_SUPPORTED"


def test_equal_box_identifiability_does_not_justify_graph_correlation() -> None:
    result = _summary([_row("g0", joint=True, box=True), _row("g1", joint=True, box=True)])
    assert result["decision"] == "SHARED_NODE_BOX_SUFFICIENT_GRAPH_CORRELATION_NOT_JUSTIFIED"


def test_vacuous_joint_set_forces_interface_redesign() -> None:
    result = _summary(
        [_row("g0", joint=False, box=False), _row("g1", joint=False, box=False)]
    )
    assert result["decision"] == "ORACLE_SUPPORT_SET_NEAR_VACUOUS_REDESIGN_INTERFACE"


def test_material_feasible_potential_widening_invalidates_run() -> None:
    rows = [_row("g0", joint=True, box=False), _row("g1", joint=True, box=False)]
    rows[0]["maximum_numerical_widening"] = 1e-3
    result = _summarize(
        rows,
        group_names=("g0", "g1"),
        anchors_per_group=1,
        minimum_anchors_per_group=1,
        numerical_widening_limit_m=1e-9,
        strict_improvement_lower_bound=0.0,
        bootstrap_seed=20261025,
        bootstrap_resamples=100,
    )
    assert result["decision"] == "INVALID_ORACLE_SET_OR_NUMERICS"
