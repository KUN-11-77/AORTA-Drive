from __future__ import annotations

from pathlib import Path

import pytest

from aorta_drive.evaluation.contextual_target_qualification import (
    filtered_one_stage,
    qualify_contextual_target,
)


def _candidate(index: int, *, raw: float = 1.0, dac: float = 1.0):
    return {
        "registered_index": index,
        "one_frame_pdm_score": raw,
        "hard_metrics": {
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": dac,
            "driving_direction_compliance": 1.0,
            "traffic_light_compliance": 1.0,
        },
        "weighted_metrics": {
            "ego_progress": 1.0,
            "time_to_collision_within_bound": 1.0,
            "lane_keeping": 1.0,
            "history_comfort": 1.0,
        },
    }


def _payloads():
    b4_rows = []
    context_rows = []
    mappings = []
    for group_index in range(42):
        group = f"g{group_index:02d}"
        for anchor in range(16):
            token = f"c-{group}-{anchor}"
            previous = f"p-{group}-{anchor}"
            b4_rows.append(
                {
                    "recording_group": group,
                    "anchor_index": anchor,
                    "log_name": f"log-{group}",
                    "scene_token": token,
                    "legal_registered_indices": [0, 1],
                    "label_generation_valid": True,
                    "panel_pose_reconstruction_bit_exact": True,
                    "membership_artifact_exact": True,
                    "ineligible_candidate_score_count": 0,
                    "candidates": [_candidate(0), _candidate(1, raw=0.5, dac=0.0)],
                }
            )
            context_rows.append(
                {
                    "recording_group": group,
                    "anchor_index": anchor,
                    "log_name": f"log-{group}",
                    "current_token": token,
                    "previous_token": previous,
                    "gap_s": 0.5,
                    "status": "AVAILABLE",
                }
            )
            mappings.append([token, previous, []])
    b4 = {
        "anchor_ledgers": b4_rows,
        "development_calibration_test_navdream_target_accessed": False,
        "full_official_pdms_claimed": False,
    }
    roster = {
        "anchor_ledger": context_rows,
        "development_calibration_test_navdream_target_accessed": False,
        "scientific_outcomes_accessed": False,
    }
    return b4, roster, {"reactive_all_mapping": mappings}


def test_filtered_one_stage_uses_returned_components() -> None:
    assert filtered_one_stage(_candidate(0)) == 1.0
    assert filtered_one_stage(_candidate(0, dac=0.0)) == 0.0


def test_qualification_separates_mapping_and_cache_readiness(tmp_path: Path) -> None:
    b4, roster, mapping = _payloads()
    report = qualify_contextual_target(b4, roster, [mapping])
    assert report["status"] == "PASS"
    assert report["candidate_count"] == 1344
    assert report["filter_difference_candidate_count"] == 672
    assert report["q_pi_ref_counterfactual_mapping_valid"] is True
    assert report["uctx_materialization_ready"] is False
    assert "PREVIOUS_CACHE_BUILD_REQUIRED" in report["decision"]


def test_missing_two_stage_mapping_fails_q_only() -> None:
    b4, roster, _ = _payloads()
    report = qualify_contextual_target(b4, roster, [{}])
    assert report["status"] == "PASS"
    assert report["uctx_definition_valid"] is True
    assert report["q_pi_ref_counterfactual_mapping_valid"] is False


def test_context_token_mismatch_is_invalid() -> None:
    b4, roster, mapping = _payloads()
    roster["anchor_ledger"][0]["current_token"] = "wrong"
    with pytest.raises(ValueError, match="current token mismatch"):
        qualify_contextual_target(b4, roster, [mapping])

