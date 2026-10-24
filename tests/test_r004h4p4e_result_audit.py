"""Adversarial tests for the payload-free R004H4P4E result audit."""

from __future__ import annotations

import copy

from audit_r004h4p4e_result import audit


def _valid_result() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for group_index in range(8):
        for anchor_index in range(16):
            rows.append(
                {
                    "recording_group": f"group-{group_index}",
                    "anchor_index": anchor_index,
                    "shape_ok": True,
                    "deterministic_replay_bit_exact": True,
                    "realized_initial_state_shared_bit_exact": True,
                    "maximum_desired_roundtrip_position_error_m": 1e-10,
                    "maximum_desired_roundtrip_heading_error_rad": 1e-15,
                    "maximum_transform_error_m": 1e-15,
                    "native_desired_to_realized_terminal_position_error_m": 1.0,
                    "maximum_challenger_desired_to_realized_position_error_m": 2.0,
                    "realized_eligible_candidate_count": 2,
                    "realized_has_eligible_candidate": True,
                    "realized_has_both_signed_eligible_candidates": True,
                    "desired_final_identity_survived_execution_for_all_candidates": False,
                }
            )
    return {
        "experiment_id": "R004H4P4E-official-execution-interface-audit-v1",
        "status": "COMPLETED",
        "implementation_revision": "06284a8fd3f13763fe56ed97cfe38f01b427d106",
        "config_sha256": (
            "eb3e2ffbcca839c5161a5d246b9e7320850d67fc271cc0bd352982421e8ddd196"
        ),
        "elapsed_seconds": 1.0,
        "data_scope": {
            "role": "fit",
            "recording_group_count": 8,
            "anchor_count": 128,
            "geometry_teacher_prediction_image_or_development_read": False,
        },
        "interface_failures": [],
        "secondary_diagnostics": {
            "realized_anchors_with_eligible_candidate": 128,
            "realized_anchor_supply_fraction": 1.0,
            "realized_anchors_with_both_signed_eligible_candidates": 128,
            "desired_final_identity_survived_execution_anchor_count": 0,
        },
        "anchor_ledger": rows,
        "decision": "PASS_OFFICIAL_EXECUTION_INTERFACE",
        "claims": {
            "official_execution_interface_closed": True,
            "realized_supply_is_decision_endpoint": False,
            "geometry_evaluated": False,
            "teacher_or_prediction_read": False,
            "development_access_authorized": False,
            "coverage_established": False,
            "gate_b_passed": False,
        },
    }


def test_result_audit_accepts_complete_internally_consistent_payload() -> None:
    assert audit(_valid_result())["status"] == "PASS"


def test_result_audit_fails_closed_on_nan_roundtrip() -> None:
    result = copy.deepcopy(_valid_result())
    ledger = result["anchor_ledger"]
    assert isinstance(ledger, list)
    ledger[0]["maximum_desired_roundtrip_position_error_m"] = float("nan")
    audited = audit(result)
    assert audited["status"] == "FAIL"
    assert "anchor contains missing, negative or non-finite diagnostics" in audited[
        "failures"
    ]
