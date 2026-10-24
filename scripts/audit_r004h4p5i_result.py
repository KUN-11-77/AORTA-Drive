"""Payload-free fail-closed audit for R004H4P5I sentinel evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


AMPLITUDES = {-1.5, -1.0, -0.5, 0.5, 1.0, 1.5}


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def audit(
    result: dict[str, Any], *, expected_revision: str, expected_config_sha256: str
) -> dict[str, Any]:
    failures: list[str] = []
    if result.get("experiment_id") != "R004H4P5I-realized-geometry-interface-v1":
        failures.append("experiment ID mismatch")
    if result.get("status") != "COMPLETED":
        failures.append("experiment status is not completed")
    if result.get("implementation_revision") != expected_revision:
        failures.append("implementation revision mismatch")
    if result.get("config_sha256") != expected_config_sha256:
        failures.append("registered config hash mismatch")
    elapsed = float(result.get("elapsed_seconds", math.nan))
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        failures.append("elapsed time is missing or invalid")

    scope = _mapping(result.get("data_scope"), "data_scope")
    if (
        scope.get("role") != "fit"
        or scope.get("split") != "train"
        or scope.get("recording_group_count") != 8
        or scope.get("anchor_count") != 8
        or scope.get("sentinel_anchor_index") != 0
        or scope.get("prevalence_or_significance_claim") is not False
        or scope.get("development_calibration_test_or_target_accessed") is not False
    ):
        failures.append("registered data scope mismatch")

    ledger = result.get("sentinel_ledger")
    if not isinstance(ledger, list) or len(ledger) != 8:
        raise ValueError("sentinel ledger must contain eight rows")
    groups: set[str] = set()
    recomputed = {
        "anchors_with_realized_intent_supply": 0,
        "anchors_with_geometry_valid_candidate": 0,
        "anchors_with_meaningful_noninclusion": 0,
        "anchors_with_teacher_bound": 0,
        "anchors_with_teacher_opportunity": 0,
    }
    for value in ledger:
        row = _mapping(value, "sentinel row")
        group = row.get("recording_group")
        if not isinstance(group, str) or row.get("anchor_index") != 0:
            failures.append("sentinel row lacks registered block identity")
        else:
            groups.add(group)
        for key in (
            "metric_cache_label_cache_bound",
            "official_vehicle_parameters_exact",
            "corner_compatibility_verified",
            "execution_deterministic_replay_bit_exact",
        ):
            if row.get(key) is not True:
                failures.append(f"implementation interface false: {key}")
        numeric = {
            key: float(row.get(key, math.nan))
            for key in (
                "maximum_corner_coordinate_disagreement_m",
                "corner_tolerance_m",
                "maximum_realized_transform_error_m",
            )
        }
        if any(not math.isfinite(number) or number < 0.0 for number in numeric.values()):
            failures.append("sentinel contains non-finite or negative interface metric")
        if (
            math.isfinite(numeric["maximum_corner_coordinate_disagreement_m"])
            and math.isfinite(numeric["corner_tolerance_m"])
            and numeric["maximum_corner_coordinate_disagreement_m"]
            > numeric["corner_tolerance_m"]
        ):
            failures.append("corner disagreement exceeds registered tolerance")
        candidates = row.get("candidate_ledger")
        if not isinstance(candidates, list) or len(candidates) != 6:
            failures.append("candidate ledger does not contain six controls")
            continue
        amplitudes = {float(_mapping(item, "candidate").get("amplitude_m", math.nan)) for item in candidates}
        if amplitudes != AMPLITUDES:
            failures.append("candidate amplitudes disagree with registration")
        intent_count = 0
        has_geometry = False
        has_meaningful = False
        has_teacher = False
        has_opportunity = False
        for item in candidates:
            candidate = _mapping(item, "candidate")
            intent = candidate.get("intent_eligible") is True
            geometry = candidate.get("geometry_valid") is True
            meaningful = candidate.get("meaningful_noninclusion") is True
            teacher = candidate.get("teacher_bound_computed") is True
            opportunity = candidate.get("teacher_opportunity") is True
            intent_count += int(intent)
            has_geometry |= geometry
            has_meaningful |= meaningful
            has_teacher |= teacher
            has_opportunity |= opportunity
            if geometry and not intent:
                failures.append("geometry-valid candidate is not intent eligible")
            if meaningful and not geometry:
                failures.append("meaningful non-inclusion lacks valid geometry")
            if teacher and not geometry:
                failures.append("teacher bound lacks valid geometry")
            if opportunity and not (meaningful and teacher):
                failures.append("teacher opportunity lacks its prerequisites")
            if geometry:
                for key in (
                    "challenger_minus_native_area_lower_m2",
                    "native_minus_challenger_area_lower_m2",
                    "teacher_improvement_lower_m",
                ):
                    number = float(candidate.get(key, math.nan))
                    if not math.isfinite(number):
                        failures.append("geometry candidate has non-finite bound")
            if meaningful and (
                float(candidate["challenger_minus_native_area_lower_m2"]) < 1.0
                or float(candidate["native_minus_challenger_area_lower_m2"]) < 1.0
            ):
                failures.append("meaningful flag violates frozen area threshold")
            if opportunity and float(candidate["teacher_improvement_lower_m"]) < 0.5:
                failures.append("opportunity flag violates frozen teacher threshold")
        if row.get("registered_candidate_count") != 6 or row.get("emitted_candidate_count") != 6:
            failures.append("provider count disagrees with registration")
        if row.get("realized_intent_eligible_count") != intent_count:
            failures.append("realized intent count does not replay")
        expected_flags = {
            "has_realized_intent_supply": intent_count > 0,
            "has_geometry_valid_candidate": has_geometry,
            "has_meaningful_noninclusion": has_meaningful,
            "has_teacher_bound": has_teacher,
            "has_teacher_opportunity": has_opportunity,
        }
        for key, expected in expected_flags.items():
            if row.get(key) is not expected:
                failures.append(f"sentinel aggregate does not replay: {key}")
        recomputed["anchors_with_realized_intent_supply"] += int(intent_count > 0)
        recomputed["anchors_with_geometry_valid_candidate"] += int(has_geometry)
        recomputed["anchors_with_meaningful_noninclusion"] += int(has_meaningful)
        recomputed["anchors_with_teacher_bound"] += int(has_teacher)
        recomputed["anchors_with_teacher_opportunity"] += int(has_opportunity)
    if len(groups) != 8:
        failures.append("sentinels do not cover eight unique recording-group blocks")
    if result.get("interface_failures") != []:
        failures.append("interface failure ledger is not empty")
    if result.get("pipeline_counts") != recomputed:
        failures.append("pipeline counts do not replay from sentinel ledger")

    implementation_passed = not failures
    pipeline_exercised = (
        recomputed["anchors_with_realized_intent_supply"] > 0
        and recomputed["anchors_with_geometry_valid_candidate"] > 0
        and recomputed["anchors_with_teacher_bound"] > 0
    )
    expected_decision = (
        "STOP_REALIZED_GEOMETRY_IMPLEMENTATION"
        if not implementation_passed
        else (
            "PASS_TO_REGISTER_128_ANCHOR_GEOMETRY_PREFLIGHT"
            if pipeline_exercised
            else "STOP_REALIZED_GEOMETRY_PIPELINE_UNEXERCISED"
        )
    )
    if result.get("decision") != expected_decision:
        failures.append("decision disagrees with registered gates")
    claims = _mapping(result.get("claims"), "claims")
    if claims.get("realized_geometry_implementation_closed") is not implementation_passed:
        failures.append("implementation claim mismatch")
    if claims.get("full_pipeline_exercised") is not pipeline_exercised:
        failures.append("pipeline-exercise claim mismatch")
    for key in (
        "prevalence_estimated",
        "method_performance_evaluated",
        "development_access_authorized",
        "coverage_established",
        "gate_b_passed",
    ):
        if claims.get(key) is not False:
            failures.append(f"forbidden claim is not false: {key}")
    return {
        "experiment_id": "R004H4P5I-mechanical-result-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "payload_files_read": 0,
        "recomputed": recomputed,
        "external_cross_model_review": "REVIEW_UNAVAILABLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-implementation-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_config_sha256 = __import__("hashlib").sha256(args.config.read_bytes()).hexdigest()
    audited = audit(
        json.loads(args.result.read_text(encoding="utf-8")),
        expected_revision=args.expected_implementation_revision,
        expected_config_sha256=expected_config_sha256,
    )
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
