from __future__ import annotations

from copy import deepcopy

import pytest

from aorta_drive.evaluation.contextual_oracle import (
    contextual_utility,
    summarize_contextual_oracle,
)
from aorta_drive.evaluation.eligible_preference_oracle import (
    EligiblePreferenceOracleProtocol,
)


def _candidate() -> dict[str, object]:
    return {
        "hard_metrics": {
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "driving_direction_compliance": 1.0,
            "traffic_light_compliance": 1.0,
        },
        "weighted_metrics": {
            "ego_progress": 0.8,
            "time_to_collision_within_bound": 1.0,
            "lane_keeping": 0.5,
            "history_comfort": 1.0,
        },
    }


def test_contextual_utility_matches_frozen_weighting() -> None:
    assert contextual_utility(_candidate(), 1.0) == pytest.approx(14.0 / 16.0)
    assert contextual_utility(_candidate(), 0.0) == pytest.approx(12.0 / 16.0)


def test_contextual_utility_respects_multiplicative_gate() -> None:
    candidate = deepcopy(_candidate())
    candidate["hard_metrics"]["traffic_light_compliance"] = 0.0  # type: ignore[index]
    assert contextual_utility(candidate, 1.0) == 0.0


def test_contextual_utility_rejects_nonbinary_comfort() -> None:
    with pytest.raises(ValueError, match="binary"):
        contextual_utility(_candidate(), 0.5)


def test_contextual_summary_uses_groups_and_detects_context_reversal() -> None:
    rows = []
    for group_index in range(42):
        for anchor_index in range(16):
            native = {
                **_candidate(),
                "registered_index": 0,
                "one_frame_pdm_score": 0.8,
                "contextual_utility": 0.7,
            }
            challenger = {
                **_candidate(),
                "registered_index": 1,
                "one_frame_pdm_score": 0.7,
                "contextual_utility": 0.9,
            }
            rows.append(
                {
                    "recording_group": f"g{group_index:02d}",
                    "anchor_index": anchor_index,
                    "frozen_challenger_eligible": [True, False, False, False, False, False],
                    "panel_pose_reconstruction_bit_exact": True,
                    "membership_artifact_exact": True,
                    "ineligible_candidate_score_count": 0,
                    "sentinel_replay_exact": True,
                    "label_generation_valid": True,
                    "candidates": [native, challenger],
                }
            )
    protocol = EligiblePreferenceOracleProtocol(
        numerical_tie_tolerance=1.0e-12,
        minimum_groups_with_nonzero_pair=42,
        minimum_groups_with_strict_native_improvement=21,
        bootstrap_seed=7,
        bootstrap_resamples=100,
        expected_eligible_challenger_count=672,
        expected_zero_eligible_anchor_count=0,
        expected_nonzero_eligible_anchor_count=672,
    )
    summary = summarize_contextual_oracle(
        rows,
        group_names=[f"g{index:02d}" for index in range(42)],
        protocol=protocol,
        diagnostic_margins=(0.0,),
    )
    assert summary["decision"].startswith("GO_TO_VISUAL_RECOVERABILITY")
    assert summary["raw_context_top_set_disagreement_anchor_count"] == 672
    assert summary["filtered_context_top_set_disagreement_anchor_count"] == 672
    assert summary["group_equal_mean_oracle_gain_all_anchors"] == pytest.approx(0.2)
