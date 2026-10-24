"""Adversarial tests for the R004H4P5I payload-free result audit."""

from __future__ import annotations

import copy

from audit_r004h4p5i_result import audit


REVISION = "a" * 40
CONFIG_HASH = "b" * 64


def _candidate(amplitude: float) -> dict[str, object]:
    return {
        "amplitude_m": amplitude,
        "intent_eligible": True,
        "intent_reason": "eligible",
        "geometry_valid": True,
        "meaningful_noninclusion": True,
        "teacher_bound_computed": True,
        "teacher_opportunity": True,
        "challenger_minus_native_area_lower_m2": 1.0,
        "native_minus_challenger_area_lower_m2": 1.0,
        "teacher_improvement_lower_m": 0.5,
    }


def _valid_result() -> dict[str, object]:
    rows = []
    for index in range(8):
        rows.append(
            {
                "recording_group": f"group-{index}",
                "anchor_index": 0,
                "metric_cache_label_cache_bound": True,
                "official_vehicle_parameters_exact": True,
                "corner_compatibility_verified": True,
                "maximum_corner_coordinate_disagreement_m": 1e-10,
                "corner_tolerance_m": 1e-9,
                "execution_deterministic_replay_bit_exact": True,
                "maximum_realized_transform_error_m": 1e-12,
                "registered_candidate_count": 6,
                "emitted_candidate_count": 6,
                "realized_intent_eligible_count": 6,
                "has_realized_intent_supply": True,
                "has_geometry_valid_candidate": True,
                "has_meaningful_noninclusion": True,
                "has_teacher_bound": True,
                "has_teacher_opportunity": True,
                "candidate_ledger": [
                    _candidate(value) for value in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
                ],
            }
        )
    counts = {
        "anchors_with_realized_intent_supply": 8,
        "anchors_with_geometry_valid_candidate": 8,
        "anchors_with_meaningful_noninclusion": 8,
        "anchors_with_teacher_bound": 8,
        "anchors_with_teacher_opportunity": 8,
    }
    return {
        "experiment_id": "R004H4P5I-realized-geometry-interface-v1",
        "status": "COMPLETED",
        "decision": "PASS_TO_REGISTER_128_ANCHOR_GEOMETRY_PREFLIGHT",
        "implementation_revision": REVISION,
        "config_sha256": CONFIG_HASH,
        "elapsed_seconds": 1.0,
        "data_scope": {
            "role": "fit",
            "split": "train",
            "recording_group_count": 8,
            "anchor_count": 8,
            "sentinel_anchor_index": 0,
            "prevalence_or_significance_claim": False,
            "development_calibration_test_or_target_accessed": False,
        },
        "interface_failures": [],
        "geometry_failures": {},
        "pipeline_counts": counts,
        "sentinel_ledger": rows,
        "claims": {
            "realized_geometry_implementation_closed": True,
            "full_pipeline_exercised": True,
            "prevalence_estimated": False,
            "method_performance_evaluated": False,
            "development_access_authorized": False,
            "coverage_established": False,
            "gate_b_passed": False,
        },
    }


def test_p5i_audit_accepts_complete_consistent_sentinel_result() -> None:
    audited = audit(
        _valid_result(), expected_revision=REVISION, expected_config_sha256=CONFIG_HASH
    )
    assert audited["status"] == "PASS"


def test_p5i_audit_rejects_nan_corner_metric() -> None:
    result = copy.deepcopy(_valid_result())
    ledger = result["sentinel_ledger"]
    assert isinstance(ledger, list)
    ledger[0]["maximum_corner_coordinate_disagreement_m"] = float("nan")
    audited = audit(
        result, expected_revision=REVISION, expected_config_sha256=CONFIG_HASH
    )
    assert audited["status"] == "FAIL"
    assert "sentinel contains non-finite or negative interface metric" in audited[
        "failures"
    ]


def test_p5i_audit_rejects_teacher_opportunity_without_geometry() -> None:
    result = copy.deepcopy(_valid_result())
    ledger = result["sentinel_ledger"]
    assert isinstance(ledger, list)
    candidate = ledger[0]["candidate_ledger"][0]
    candidate["geometry_valid"] = False
    audited = audit(
        result, expected_revision=REVISION, expected_config_sha256=CONFIG_HASH
    )
    assert audited["status"] == "FAIL"
    assert "meaningful non-inclusion lacks valid geometry" in audited["failures"]
