"""Payload-free mechanical result audit for R004H4P4E."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def audit(result: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if result.get("experiment_id") != "R004H4P4E-official-execution-interface-audit-v1":
        failures.append("experiment ID mismatch")
    if result.get("status") != "COMPLETED":
        failures.append("experiment status is not completed")
    if (
        result.get("implementation_revision")
        != "06284a8fd3f13763fe56ed97cfe38f01b427d106"
    ):
        failures.append("implementation revision mismatch")
    if (
        result.get("config_sha256")
        != "eb3e2ffbcca839c5161a5d246b9e7320850d67fc271cc0bd352982421e8dd196"
    ):
        failures.append("registered config hash mismatch")
    scope = _mapping(result.get("data_scope"), "data_scope")
    if (
        scope.get("role") != "fit"
        or scope.get("recording_group_count") != 8
        or scope.get("anchor_count") != 128
        or scope.get("geometry_teacher_prediction_image_or_development_read") is not False
    ):
        failures.append("registered data scope mismatch")

    ledger = result.get("anchor_ledger")
    if not isinstance(ledger, list) or len(ledger) != 128:
        raise ValueError("anchor ledger must contain 128 rows")
    keys: set[tuple[str, int]] = set()
    groups: Counter[str] = Counter()
    supplied = 0
    both_signed = 0
    final_identity = 0
    for value in ledger:
        row = _mapping(value, "anchor row")
        group = row.get("recording_group")
        anchor_index = row.get("anchor_index")
        if not isinstance(group, str) or not isinstance(anchor_index, int):
            raise ValueError("anchor row lacks group/index")
        keys.add((group, anchor_index))
        groups[group] += 1
        for gate in (
            "shape_ok",
            "deterministic_replay_bit_exact",
            "realized_initial_state_shared_bit_exact",
        ):
            if row.get(gate) is not True:
                failures.append(f"interface gate false: {gate}")
        numeric_fields = (
            "maximum_desired_roundtrip_position_error_m",
            "maximum_desired_roundtrip_heading_error_rad",
            "maximum_transform_error_m",
            "native_desired_to_realized_terminal_position_error_m",
            "maximum_challenger_desired_to_realized_position_error_m",
        )
        values = {
            key: float(row.get(key, float("nan"))) for key in numeric_fields
        }
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            failures.append("anchor contains missing, negative or non-finite diagnostics")
        if values["maximum_desired_roundtrip_position_error_m"] > 1e-8:
            failures.append("desired position roundtrip exceeds registered tolerance")
        if values["maximum_desired_roundtrip_heading_error_rad"] > 1e-10:
            failures.append("desired heading roundtrip exceeds registered tolerance")
        eligible_count = row.get("realized_eligible_candidate_count")
        if not isinstance(eligible_count, int) or not 0 <= eligible_count <= 6:
            failures.append("realized eligible-candidate count is invalid")
            eligible_count = -1
        has_supply = row.get("realized_has_eligible_candidate") is True
        has_both = row.get("realized_has_both_signed_eligible_candidates") is True
        if has_supply != (eligible_count > 0):
            failures.append("realized supply boolean disagrees with eligible count")
        if has_both and eligible_count < 2:
            failures.append("both-sign diagnostic is impossible for eligible count")
        supplied += int(has_supply)
        both_signed += int(
            has_both
        )
        final_identity += int(
            row.get("desired_final_identity_survived_execution_for_all_candidates")
            is True
        )
    if len(keys) != 128 or len(groups) != 8 or any(
        count != 16 for count in groups.values()
    ):
        failures.append("anchor ledger is not eight complete unique groups")

    diagnostics = _mapping(result.get("secondary_diagnostics"), "secondary_diagnostics")
    expected_diagnostics = {
        "realized_anchors_with_eligible_candidate": supplied,
        "realized_anchor_supply_fraction": supplied / 128.0,
        "realized_anchors_with_both_signed_eligible_candidates": both_signed,
        "desired_final_identity_survived_execution_anchor_count": final_identity,
    }
    if diagnostics != expected_diagnostics:
        failures.append("secondary diagnostics do not replay from ledger")
    if result.get("interface_failures") != []:
        failures.append("interface failure ledger is not empty")
    elapsed = float(result.get("elapsed_seconds", float("nan")))
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        failures.append("elapsed time is missing or invalid")

    interface_passed = not failures
    expected_decision = (
        "PASS_OFFICIAL_EXECUTION_INTERFACE"
        if interface_passed
        else "STOP_OFFICIAL_EXECUTION_INTERFACE"
    )
    if result.get("decision") != expected_decision:
        failures.append("decision disagrees with registered interface gates")
    claims = _mapping(result.get("claims"), "claims")
    if claims.get("official_execution_interface_closed") is not interface_passed:
        failures.append("official-interface claim mismatch")
    for key in (
        "realized_supply_is_decision_endpoint",
        "geometry_evaluated",
        "teacher_or_prediction_read",
        "development_access_authorized",
        "coverage_established",
        "gate_b_passed",
    ):
        if claims.get(key) is not False:
            failures.append(f"forbidden claim is not false: {key}")
    return {
        "experiment_id": "R004H4P4E-mechanical-result-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "payload_files_read": 0,
        "recomputed": expected_diagnostics,
        "external_cross_model_review": "REVIEW_UNAVAILABLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audited = audit(json.loads(args.result.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audited, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audited, ensure_ascii=False, indent=2, sort_keys=True))
    if audited["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
