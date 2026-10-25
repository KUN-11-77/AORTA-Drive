from __future__ import annotations

from audit_r004h7bs4b4r1_filtered_score_semantics import audit, filtered_one_stage_score


def _candidate(score: float, *, dac: float = 1.0, comfort: float = 1.0):
    return {
        "one_frame_pdm_score": score,
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
            "history_comfort": comfort,
        },
    }


def _result(candidates):
    return {
        "experiment_id": "synthetic",
        "anchor_ledgers": [
            {"recording_group": "g0", "candidates": candidates},
        ],
    }


def test_filtered_score_matches_components() -> None:
    candidate = _candidate(1.0)
    assert filtered_one_stage_score(candidate) == 1.0
    report = audit(_result([candidate, candidate]))
    assert report["mismatch_candidate_count"] == 0
    assert report["pair_sign_disagreement_count"] == 0


def test_post_filter_component_change_is_detected_without_changing_registered_decision() -> None:
    native = _candidate(0.0, dac=1.0)
    challenger = _candidate(0.0, dac=1.0, comfort=0.0)
    report = audit(_result([native, challenger]))
    assert report["mismatch_candidate_count"] == 2
    assert report["pair_sign_disagreement_count"] == 1
    assert report["top_set_disagreement_anchor_count"] == 1
    assert report["registered_b4r1_decision_changed"] is False
    assert report["full_target_claim_supported"] is False
