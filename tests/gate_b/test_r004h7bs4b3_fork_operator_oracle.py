from __future__ import annotations

from typing import Any

from run_r004h7bs4b3_fork_operator_oracle import _summarize


def _pair(*, temporal_nonzero: bool, spatial_nonzero: bool) -> dict[str, Any]:
    return {
        "parent_category": "common_minimum_domination",
        "global_contrast_m": 0.0,
        "fixed_suffix_contrast_m": 1.0 if temporal_nonzero else 0.0,
        "temporal_fork": {
            "valid": True,
            "fork_index": 1,
            "nonzero_branch_contrast": temporal_nonzero,
            "shared_strictly_safe": True,
            "road_rescue_opportunity": temporal_nonzero,
            "road_induction_risk": False,
        },
        "spatial_fork": {
            "valid": True,
            "nonzero_branch_contrast": spatial_nonzero,
            "shared_strictly_safe": True,
        },
    }


def test_temporal_primary_uses_group_equal_repeated_measure_summary() -> None:
    rows = [
        {
            "recording_group": "g0",
            "pair_factors": [_pair(temporal_nonzero=True, spatial_nonzero=False)],
        },
        {
            "recording_group": "g1",
            "pair_factors": [_pair(temporal_nonzero=True, spatial_nonzero=True)],
        },
    ]
    result = _summarize(
        rows,
        group_names=("g0", "g1"),
        expected_common_min_pairs=2,
        minimum_valid_fraction=0.95,
        minimum_recovery_rate=0.5,
        minimum_affected_groups=1,
        bootstrap_seed=7,
        bootstrap_resamples=100,
    )
    assert result["temporal_primary_pass"]
    assert result["temporal_group_equal_recovery_rate"] == 1.0
    assert result["decision"] == (
        "TEMPORAL_FORK_OPERATOR_NONDEGENERATE_ON_FIT__DECISION_RELEVANCE_HOLD"
    )


def test_spatial_sensitivity_cannot_replace_failed_temporal_primary() -> None:
    rows = [
        {
            "recording_group": "g0",
            "pair_factors": [_pair(temporal_nonzero=False, spatial_nonzero=True)],
        },
        {
            "recording_group": "g1",
            "pair_factors": [_pair(temporal_nonzero=False, spatial_nonzero=True)],
        },
    ]
    result = _summarize(
        rows,
        group_names=("g0", "g1"),
        expected_common_min_pairs=2,
        minimum_valid_fraction=0.95,
        minimum_recovery_rate=0.5,
        minimum_affected_groups=1,
        bootstrap_seed=7,
        bootstrap_resamples=100,
    )
    assert not result["temporal_primary_pass"]
    assert result["spatial_sensitivity_pass"]
    assert result["decision"] == (
        "SPATIAL_ONLY_SIGNAL_REQUIRES_THEORY_REVIEW__NO_OPERATOR_SELECTED"
    )
