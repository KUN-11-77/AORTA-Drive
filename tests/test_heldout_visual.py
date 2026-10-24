from __future__ import annotations

from typing import Any

from aorta_drive.evaluation.heldout_visual import (
    CONFIRMATORY_CONDITIONS,
    deterministic_group_derangement,
    summarize_heldout_ledgers,
)
from aorta_drive.vision.appearance import H6_APPEARANCE_CONDITIONS


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
    for condition in H6_APPEARANCE_CONDITIONS:
        conditions[condition] = {
            "task_mae_m": correct,
            "appearance_mae_m": 0.0 if condition == "clean" else 0.01,
            "edge_energy_mae_m": 0.30,
            "teacher_cosine": 0.8,
            "salient_edge_mae_m": correct + 0.1,
        }
    return {
        "recording_group": group,
        "anchor_index": anchor,
        "conditions": conditions,
        "zero_task_mae_m": 0.6,
        "fit_teacher_prior": {"task_mae_m": 0.4, "salient_edge_mae_m": 0.55},
        "mismatched_clean": {"task_mae_m": 0.35, "salient_edge_mae_m": 0.5},
        "image_removed": {"task_mae_m": 0.45, "salient_edge_mae_m": 0.6},
        "spatial_pool": {
            condition: {"task_mae_m": 0.5}
            for condition in H6_APPEARANCE_CONDITIONS
        },
        "uniform_fit_mean_DINO": {"task_mae_m": 0.52},
        "perturbation_ledger": {
            **{
                condition: {
                    "pixel_srgb_mae": 0.0 if condition == "clean" else 0.1,
                    "dense_feature_rms": 0.0 if condition == "clean" else 0.1,
                    "dense_feature_one_minus_cosine": 0.0 if condition == "clean" else 0.01,
                }
                for condition in H6_APPEARANCE_CONDITIONS
            },
            "mismatched_clean": {
                "pixel_srgb_mae": 0.2,
                "dense_feature_rms": 0.2,
                "dense_feature_one_minus_cosine": 0.02,
            },
            "image_removed": {
                "pixel_srgb_mae": 0.3,
                "dense_feature_rms": 0.3,
                "dense_feature_one_minus_cosine": 0.03,
            },
        },
    }


def test_summary_uses_group_level_conjunctive_gate() -> None:
    ledgers = [_row(group, anchor) for group in ("a", "b", "c") for anchor in range(2)]
    summary = summarize_heldout_ledgers(
        ledgers,
        all_conditions=H6_APPEARANCE_CONDITIONS,
        expected_groups=3,
        anchors_per_group=2,
        replicates=500,
    )
    assert summary["claim_1_heldout_generalization_supported"] is True
    assert summary["claim_2_scene_conditioning_supported"] is True
    assert summary["decision"] == "SUPPORTED"
