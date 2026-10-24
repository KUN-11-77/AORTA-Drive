from __future__ import annotations

from typing import Any

import pytest

from aorta_drive.evaluation.preference_oracle import (
    PreferenceOracleProtocol,
    summarize_preference_oracle,
)


def _rows(*, positive_gain: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_index in range(42):
        group = f"group-{group_index:02d}"
        for anchor in range(16):
            native = 0.5
            scores = [native, 0.6 if positive_gain else 0.4, 0.3, 0.2, 0.1, 0.0, 0.45]
            candidates = []
            for index, score in enumerate(scores):
                candidates.append(
                    {
                        "registered_index": index,
                        "one_frame_pdm_score": score,
                        "hard_metrics": {
                            "no_at_fault_collisions": 1.0,
                            "drivable_area_compliance": 1.0,
                            "driving_direction_compliance": 1.0,
                            "traffic_light_compliance": 1.0,
                        },
                    }
                )
            rows.append(
                {
                    "recording_group": group,
                    "anchor_index": anchor,
                    "candidates": candidates,
                    "label_generation_valid": True,
                    "panel_reconstruction_bit_exact": True,
                    "sentinel_replay_exact": True,
                }
            )
    return rows


def _protocol() -> PreferenceOracleProtocol:
    return PreferenceOracleProtocol(
        numerical_tie_tolerance=1.0e-12,
        minimum_groups_with_nonzero_pair=42,
        minimum_groups_with_strict_native_improvement=21,
        bootstrap_seed=20261025,
        bootstrap_resamples=200,
    )


def test_positive_oracle_signal_passes_without_training_claim() -> None:
    result = summarize_preference_oracle(
        _rows(),
        group_names=[f"group-{index:02d}" for index in range(42)],
        protocol=_protocol(),
        diagnostic_margins=[0.0, 0.01, 0.05],
    )
    assert result["decision"] == (
        "GO_TO_MATCHED_STRUCTURE_PREREGISTRATION__NO_TRAINING_YET"
    )
    assert result["group_bootstrap_oracle_gain_ci95"][0] > 0.0
    assert result["candidate_count"] == 4704
    assert result["pair_count"] == 14112


def test_no_positive_native_improvement_stops_current_panel() -> None:
    result = summarize_preference_oracle(
        _rows(positive_gain=False),
        group_names=[f"group-{index:02d}" for index in range(42)],
        protocol=_protocol(),
        diagnostic_margins=[0.0, 0.01],
    )
    assert result["groups_with_strict_native_improvement"] == 0
    assert result["decision"] == "STOP_CURRENT_PANEL_NO_ORACLE_IMPROVEMENT"


def test_interface_failure_dominates_scientific_signal() -> None:
    rows = _rows()
    rows[0]["panel_reconstruction_bit_exact"] = False
    result = summarize_preference_oracle(
        rows,
        group_names=[f"group-{index:02d}" for index in range(42)],
        protocol=_protocol(),
        diagnostic_margins=[0.0],
    )
    assert result["decision"] == "INVALID_PREFERENCE_LABEL_INTERFACE"


def test_rejects_nonfinite_or_out_of_range_score() -> None:
    rows = _rows()
    rows[0]["candidates"][0]["one_frame_pdm_score"] = 1.1
    with pytest.raises(ValueError, match="scores must be finite"):
        summarize_preference_oracle(
            rows,
            group_names=[f"group-{index:02d}" for index in range(42)],
            protocol=_protocol(),
            diagnostic_margins=[0.0],
        )
