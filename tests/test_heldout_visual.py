from __future__ import annotations

from typing import Any

from aorta_drive.evaluation.heldout_visual import (
    CONFIRMATORY_CONDITIONS,
    deterministic_group_derangement,
    summarize_heldout_ledgers,
)


def test_derangement_is_deterministic_and_has_no_fixed_points() -> None:
    groups = ["g2", "g0", "g3", "g1"]
    first = deterministic_group_derangement(groups)
    second = deterministic_group_derangement(reversed(groups))
    assert first == second
    assert set(first) == set(groups)
    assert set(first.values()) == set(groups)
    assert all(group != donor for group, donor in first.items())


def _row(group: str, anchor: int, *, correct: float = 0.2) -> dict[str, Any]:
    conditions = {}
    for condition in (*CONFIRMATORY_CONDITIONS, "low_exposure_twice"):
        conditions[condition] = {
            "task_mae_m": correct,
            "appearance_mae_m": 0.0 if condition == "clean" else 0.01,
            "edge_energy_mae_m": 0.30,
            "teacher_cosine": 0.8,
        }
    return {
        "recording_group": group,
        "anchor_index": anchor,
        "conditions": conditions,
        "zero_task_mae_m": 0.6,
        "fit_teacher_prior": {"task_mae_m": 0.4},
        "mismatched_clean": {"task_mae_m": 0.35},
        "image_removed": {"task_mae_m": 0.45},
    }


def test_summary_uses_group_level_conjunctive_gate() -> None:
    ledgers = [_row(group, anchor) for group in ("a", "b", "c") for anchor in range(2)]
    summary = summarize_heldout_ledgers(
        ledgers,
        all_conditions=(*CONFIRMATORY_CONDITIONS, "low_exposure_twice"),
        expected_groups=3,
        anchors_per_group=2,
        replicates=500,
    )
    assert summary["claim_1_heldout_generalization_supported"] is True
    assert summary["claim_2_scene_conditioning_supported"] is True
    assert summary["decision"] == "SUPPORTED"
