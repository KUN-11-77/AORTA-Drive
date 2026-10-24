"""Payload-free mechanical audit for R004H4P3S synthetic qualification."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def audit(result: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if result.get("experiment_id") != "R004H4P3S-native-offset-synthetic-qualification-v1":
        failures.append("experiment ID mismatch")
    design = _mapping(result.get("design"), "design")
    if (
        design.get("analytic_case_count") != 6
        or design.get("registered_amplitude_count") != 6
        or design.get("statistical_inference_authorized") is not False
        or design.get("fit_development_calibration_test_or_target_accessed") is not False
    ):
        failures.append("registered deterministic design mismatch")
    ledger = result.get("candidate_ledger")
    if not isinstance(ledger, list) or not ledger:
        raise ValueError("candidate ledger must be a non-empty list")
    if design.get("accepted_candidate_count") != len(ledger):
        failures.append("accepted candidate count mismatch")
    cases: dict[str, set[float]] = {}
    for value in ledger:
        row = _mapping(value, "candidate row")
        case = row.get("case")
        amplitude = row.get("amplitude_m")
        if not isinstance(case, str) or not isinstance(amplitude, (int, float)):
            raise ValueError("candidate row lacks case/amplitude")
        cases.setdefault(case, set()).add(math.copysign(1.0, float(amplitude)))
        if row.get("intent_eligible") is not True:
            failures.append("accepted candidate is not intent eligible")
        if row.get("final_segment_bit_exact") is not True:
            failures.append("accepted candidate lacks exact final segment")
        if float(row.get("maximum_middle_displacement_m", -1.0)) < 0.1:
            failures.append("accepted candidate lacks nontrivial middle displacement")
        if (
            float(row.get("progress_ratio", math.nan)) != 1.0
            or float(row.get("terminal_cross_track_m", math.nan)) != 0.0
            or float(row.get("terminal_heading_difference_rad", math.nan)) != 0.0
            or float(row.get("terminal_speed_difference_mps", math.nan)) != 0.0
        ):
            failures.append("endpoint invariant arithmetic is not exact")
    if len(cases) != 6 or any(signs != {-1.0, 1.0} for signs in cases.values()):
        failures.append("analytic cases do not retain both perturbation signs")

    checks = _mapping(result.get("checks"), "checks")
    expected_checks = {
        "every_analytic_case_has_both_signed_accepted_candidates",
        "every_accepted_candidate_is_intent_eligible",
        "every_accepted_candidate_has_bit_exact_final_segment",
        "every_accepted_candidate_has_nontrivial_mid_horizon_displacement",
        "deterministic_replay_is_bit_exact",
        "excessive_amplitude_is_rejected",
        "insufficient_displacement_is_rejected",
        "full_repository_tests_pass",
    }
    if set(checks) != expected_checks or any(value is not True for value in checks.values()):
        failures.append("one or more registered qualification checks did not pass")
    if int(result.get("repository_test_count", 0)) < 209:
        failures.append("full repository test count is below registration")
    if result.get("invariant_failures") != []:
        failures.append("invariant failure ledger is not empty")
    negative = _mapping(result.get("negative_controls"), "negative_controls")
    excessive = _mapping(
        negative.get("excessive_amplitude_rejections"), "excessive rejection counts"
    )
    insufficient = _mapping(
        negative.get("insufficient_displacement_rejections"),
        "insufficient rejection counts",
    )
    if excessive.get("continuous_dynamic_envelope") != 1:
        failures.append("excessive-amplitude control was not dynamically rejected")
    if insufficient.get("native_reference_displacement") != 6:
        failures.append("insufficient-displacement control was not rejected six times")

    expected_decision = (
        "PASS_TO_REGISTER_R004H4P3_FIT_SUPPLY"
        if not failures
        else "STOP_SYNTHETIC_PROVIDER_QUALIFICATION"
    )
    if result.get("decision") != expected_decision:
        failures.append("decision disagrees with replayed qualification gates")
    claims = _mapping(result.get("claims"), "claims")
    if claims.get("synthetic_provider_interface_qualified") is not (not failures):
        failures.append("synthetic qualification claim mismatch")
    for key in (
        "continuous_dynamics_screen_is_formal_real_arithmetic_proof",
        "fit_data_read",
        "geometry_teacher_or_prediction_read",
        "candidate_supply_estimated",
        "development_access_authorized",
        "h4_registered",
        "coverage_established",
        "gate_b_passed",
    ):
        if claims.get(key) is not False:
            failures.append(f"forbidden claim is not false: {key}")
    return {
        "experiment_id": "R004H4P3S-mechanical-result-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "payload_files_read": 0,
        "recomputed": {
            "accepted_candidate_count": len(ledger),
            "analytic_case_count": len(cases),
            "both_sign_case_count": sum(signs == {-1.0, 1.0} for signs in cases.values()),
            "repository_test_count": int(result.get("repository_test_count", 0)),
        },
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
