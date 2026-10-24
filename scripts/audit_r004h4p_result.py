"""Mechanical integrity audit for the R004H4P single-anchor result."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def audit(result: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if result.get("experiment_id") != "R004H4P-continuous-opportunity-preflight-v1":
        failures.append("experiment ID mismatch")
    scope = _mapping(result.get("data_scope"), "data_scope")
    if scope.get("role") != "fit" or scope.get("split") != "train":
        failures.append("result did not use the registered fit/train scope")
    if scope.get("calibration_test_target_or_development_accessed") is not False:
        failures.append("prohibited data-access flag is not false")
    native = _mapping(result.get("native"), "native")
    if native.get("identity") != "cached_PDM_Closed_planner_trajectory":
        failures.append("native identity mismatch")
    if native.get("human_trajectory_used") is not False or native.get("sample_count") != 9:
        failures.append("native source or sample-count contract failed")

    counts = _mapping(result.get("candidate_counts"), "candidate_counts")
    ledger = result.get("candidate_ledger")
    if not isinstance(ledger, list):
        raise ValueError("candidate_ledger must be a list")
    if counts.get("raw") != len(ledger):
        failures.append("raw candidate count disagrees with ledger")
    observed: Counter[str] = Counter()
    for item in ledger:
        row = _mapping(item, "candidate row")
        if row.get("intent_eligible") is True:
            observed["intent_eligible"] += 1
        if row.get("geometry_valid") is True:
            observed["continuous_geometry_valid"] += 1
        if row.get("meaningful_noninclusion") is True:
            observed["meaningful_bidirectional_noninclusion"] += 1
        if row.get("teacher_opportunity") is True:
            observed["teacher_opportunity"] += 1
    for key in (
        "intent_eligible",
        "continuous_geometry_valid",
        "meaningful_bidirectional_noninclusion",
        "teacher_opportunity",
    ):
        if counts.get(key) != observed[key]:
            failures.append(f"{key} count disagrees with ledger")

    intent_count = observed["intent_eligible"]
    geometry_count = observed["continuous_geometry_valid"]
    decision = result.get("decision")
    if intent_count == 0 and decision != "STOP_NO_INTENT_ELIGIBLE_CANDIDATE":
        failures.append("zero intent eligibility was not a STOP decision")
    if intent_count > 0 and geometry_count == 0 and decision != (
        "STOP_NO_CONTINUOUS_GEOMETRY_VALID_CANDIDATE"
    ):
        failures.append("zero valid geometry was not a STOP decision")
    if decision == "PASS" and (intent_count == 0 or geometry_count == 0):
        failures.append("PASS requires exercised intent and geometry branches")

    claims = _mapping(result.get("claims"), "claims")
    if claims.get("continuous_noninclusion_interface_closed") is not (geometry_count > 0):
        failures.append("continuous-interface claim disagrees with exercised geometry")
    for key in (
        "r004h4_registered",
        "development_access_authorized",
        "coverage_established",
        "gate_b_passed",
        "opportunity_prevalence_estimated",
    ):
        if claims.get(key) is not False:
            failures.append(f"forbidden claim is not false: {key}")
    return {
        "experiment_id": "R004H4P-mechanical-result-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "payload_files_read": 0,
        "recomputed_counts": {
            key: observed[key]
            for key in (
                "intent_eligible",
                "continuous_geometry_valid",
                "meaningful_bidirectional_noninclusion",
                "teacher_opportunity",
            )
        },
        "external_cross_model_review": "REVIEW_UNAVAILABLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    audited = audit(result)
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
