from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from aorta_drive.evaluation.eligible_preference_oracle import (
    EligiblePreferenceOracleProtocol,
    summarize_eligible_preference_oracle,
)
from aorta_drive.vision.feature_cache import sha256_file


def _candidate(index: int, score: float, *, hard: float = 1.0) -> dict[str, Any]:
    return {
        "registered_index": index,
        "one_frame_pdm_score": score,
        "hard_metrics": {
            "no_at_fault_collisions": hard,
            "drivable_area_compliance": 1.0,
            "driving_direction_compliance": 1.0,
            "traffic_light_compliance": 1.0,
        },
    }


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_index in range(42):
        group = f"group-{group_index:02d}"
        for anchor in range(16):
            if anchor == 0:
                mask = [False] * 6
                candidates = [_candidate(0, 0.5)]
            else:
                mask = [True, False, True, False, False, False]
                candidates = [
                    _candidate(0, 0.5, hard=0.0),
                    _candidate(1, 0.6),
                    _candidate(3, 0.4),
                ]
            rows.append(
                {
                    "recording_group": group,
                    "anchor_index": anchor,
                    "frozen_challenger_eligible": mask,
                    "candidates": candidates,
                    "label_generation_valid": True,
                    "panel_pose_reconstruction_bit_exact": True,
                    "membership_artifact_exact": True,
                    "ineligible_candidate_score_count": 0,
                    "sentinel_replay_exact": True,
                }
            )
    return rows


def _protocol() -> EligiblePreferenceOracleProtocol:
    return EligiblePreferenceOracleProtocol(
        numerical_tie_tolerance=1.0e-12,
        minimum_groups_with_nonzero_pair=42,
        minimum_groups_with_strict_native_improvement=21,
        bootstrap_seed=20261025,
        bootstrap_resamples=200,
        expected_eligible_challenger_count=42 * 15 * 2,
        expected_zero_eligible_anchor_count=42,
        expected_nonzero_eligible_anchor_count=42 * 15,
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_eligible_preference_oracle(
        rows,
        group_names=[f"group-{index:02d}" for index in range(42)],
        protocol=_protocol(),
        diagnostic_margins=[0.0, 0.01, 0.05],
    )


def test_variable_legal_domain_passes_and_retains_zero_candidate_anchors() -> None:
    result = _summarize(_rows())
    assert result["decision"] == (
        "GO_TO_MATCHED_STRUCTURE_PREREGISTRATION__NO_TRAINING_YET"
    )
    assert result["zero_eligible_anchor_count"] == 42
    assert result["candidate_count"] == 42 * (1 + 15 * 3)
    assert result["pair_count"] == 42 * 15 * 3
    assert result["group_equal_mean_oracle_gain_all_anchors"] == pytest.approx(
        15.0 / 16.0 * 0.1
    )


def test_scoring_ineligible_registered_candidate_is_rejected() -> None:
    rows = _rows()
    rows[0]["candidates"].append(_candidate(1, 0.9))
    with pytest.raises(ValueError, match="frozen legal domain"):
        _summarize(rows)


def test_domain_count_drift_invalidates_interface_before_signal() -> None:
    rows = _rows()
    rows[0]["frozen_challenger_eligible"] = [True, False, False, False, False, False]
    rows[0]["candidates"] = [_candidate(0, 0.5), _candidate(1, 0.9)]
    result = _summarize(rows)
    assert result["domain_counts_exact"] is False
    assert result["decision"] == "INVALID_PREFERENCE_LABEL_INTERFACE"


@pytest.mark.parametrize(
    "field,value",
    [
        ("panel_pose_reconstruction_bit_exact", False),
        ("membership_artifact_exact", False),
        ("ineligible_candidate_score_count", 1),
        ("sentinel_replay_exact", False),
    ],
)
def test_each_interface_gate_fails_closed(field: str, value: Any) -> None:
    rows = _rows()
    rows[0][field] = value
    result = _summarize(rows)
    assert result["decision"] == "INVALID_PREFERENCE_LABEL_INTERFACE"


def test_no_legal_challengers_anywhere_stops_without_nan() -> None:
    rows = _rows()
    for row in rows:
        row["frozen_challenger_eligible"] = [False] * 6
        row["candidates"] = [row["candidates"][0]]
    protocol = EligiblePreferenceOracleProtocol(
        numerical_tie_tolerance=1.0e-12,
        minimum_groups_with_nonzero_pair=42,
        minimum_groups_with_strict_native_improvement=21,
        bootstrap_seed=20261025,
        bootstrap_resamples=200,
        expected_eligible_challenger_count=0,
        expected_zero_eligible_anchor_count=672,
        expected_nonzero_eligible_anchor_count=0,
    )
    result = summarize_eligible_preference_oracle(
        rows,
        group_names=[f"group-{index:02d}" for index in range(42)],
        protocol=protocol,
        diagnostic_margins=[0.0],
    )
    assert result["pair_count"] == 0
    assert result["group_equal_mean_nonzero_pair_fraction_all_anchors"] == 0.0
    assert result["decision"] == "STOP_CURRENT_LEGAL_PANEL_NO_ORACLE_IMPROVEMENT"


def test_r1_contract_freezes_implementation_and_retains_b4_thresholds() -> None:
    repository = Path(__file__).resolve().parents[2]
    r1 = yaml.safe_load(
        (repository / "configs/experiment/r004h7bs4b4r1_eligible_preference_oracle.yaml").read_text(
            encoding="utf-8"
        )
    )["H7BS4B4R1"]
    b4 = yaml.safe_load(
        (repository / "configs/experiment/r004h7bs4b4_preference_oracle.yaml").read_text(
            encoding="utf-8"
        )
    )["H7BS4B4"]
    frozen = r1["implementation_sha256"]
    paths = {
        "eligible_preference_oracle_source": repository
        / "src/aorta_drive/evaluation/eligible_preference_oracle.py",
        "navsim_execution_source": repository
        / "src/aorta_drive/evaluation/navsim_execution.py",
        "native_offset_source": repository / "src/aorta_drive/candidates/native_offset.py",
        "h4_source": repository / "src/aorta_drive/evaluation/h4.py",
        "b4_runner_dependency": repository
        / "scripts/run_r004h7bs4b4_preference_oracle.py",
        "runner_source": repository
        / "scripts/run_r004h7bs4b4r1_eligible_preference_oracle.py",
        "audit_source": repository
        / "scripts/audit_r004h7bs4b4r1_eligible_preference_oracle.py",
    }
    assert {name: sha256_file(path) for name, path in paths.items()} == {
        name: frozen[name] for name in paths
    }
    assert r1["decision_protocol"] == {
        **b4["decision_protocol"],
        "all_sixteen_anchors_remain_in_every_group_mean": True,
        "thresholds_identical_to_b4": True,
        "if_zero_groups_have_strict_native_improvement": (
            "STOP_CURRENT_LEGAL_PANEL_NO_ORACLE_IMPROVEMENT"
        ),
        "otherwise": "HOLD_CURRENT_LEGAL_PANEL_PREFERENCE_SIGNAL_INSUFFICIENT",
    }
    domain = r1["candidate_domain"]
    assert domain["expected_total_scored_candidates"] == (
        672 + domain["expected_eligible_challengers"]
    )
    assert domain["expected_zero_eligible_anchors"] + domain[
        "expected_nonzero_eligible_anchors"
    ] == 672
