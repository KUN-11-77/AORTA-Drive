"""Mechanical integrity audit for the R004H4P2 candidate-supply result."""

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
    experiment_id = result.get("experiment_id")
    valid_ids = {
        "R004H4P2-fit-candidate-supply-diagnosis-v1",
        "R004H4P2R1-comparable-speed-supply-replay-v1",
    }
    if experiment_id not in valid_ids:
        failures.append("experiment ID mismatch")
    scope = _mapping(result.get("data_scope"), "data_scope")
    if (
        scope.get("role") != "fit"
        or scope.get("split") != "train"
        or scope.get("recording_group_count") != 8
        or scope.get("anchor_count") != 128
    ):
        failures.append("registered data scope mismatch")
    if scope.get("development_calibration_test_or_target_accessed") is not False:
        failures.append("prohibited data-access flag is not false")
    if experiment_id == "R004H4P2R1-comparable-speed-supply-replay-v1" and scope.get(
        "terminal_speed_quantity"
    ) != "position_derived_final_segment_average_speed_for_both":
        failures.append("comparable positional speed quantity is not declared")

    ledger = result.get("anchor_ledger")
    if not isinstance(ledger, list) or len(ledger) != 128:
        raise ValueError("anchor_ledger must contain exactly 128 rows")
    group_counts: Counter[str] = Counter()
    group_supplied: Counter[str] = Counter()
    supplied = 0
    raw_candidates = 0
    eligible_candidates = 0
    keys: set[tuple[str, int]] = set()
    for value in ledger:
        row = _mapping(value, "anchor row")
        group = row.get("recording_group")
        index = row.get("anchor_index")
        if not isinstance(group, str) or not isinstance(index, int):
            raise ValueError("anchor row lacks group/index")
        keys.add((group, index))
        group_counts[group] += 1
        has_supply = row.get("has_eligible_candidate") is True
        supplied += int(has_supply)
        group_supplied[group] += int(has_supply)
        raw_candidates += int(row.get("raw_candidate_count", -1))
        eligible_candidates += int(row.get("eligible_candidate_count", -1))
    if len(keys) != 128 or any(count != 16 for count in group_counts.values()):
        failures.append("anchor ledger is not eight complete unique groups")

    supply = _mapping(result.get("supply"), "supply")
    expected_anchor_fraction = supplied / 128.0
    expected_group_fractions = {
        group: group_supplied[group] / 16.0 for group in sorted(group_counts)
    }
    if supply.get("anchors_with_eligible_candidate") != supplied:
        failures.append("supplied-anchor count mismatch")
    if supply.get("anchor_supply_fraction") != expected_anchor_fraction:
        failures.append("anchor supply fraction mismatch")
    if supply.get("group_anchor_supply_fractions") != expected_group_fractions:
        failures.append("group supply fractions mismatch")
    if supply.get("minimum_group_anchor_supply_fraction") != min(
        expected_group_fractions.values()
    ):
        failures.append("minimum group supply fraction mismatch")

    diagnostics = _mapping(result.get("candidate_diagnostics"), "candidate_diagnostics")
    failures_by_reason = _mapping(
        diagnostics.get("first_failure_counts"), "first_failure_counts"
    )
    if diagnostics.get("raw_candidate_evaluations") != raw_candidates:
        failures.append("raw candidate total mismatch")
    if sum(int(value) for value in failures_by_reason.values()) != raw_candidates:
        failures.append("first-failure ledger does not exhaust candidate evaluations")
    if int(failures_by_reason.get("eligible", -1)) != eligible_candidates:
        failures.append("eligible candidate total mismatch")
    independent = _mapping(
        diagnostics.get("independent_gate_pass_fractions"),
        "independent_gate_pass_fractions",
    )
    if set(independent) != {"progress", "cross_track", "heading", "speed"} or any(
        not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0
        for value in independent.values()
    ):
        failures.append("independent gate fractions are invalid")

    adequate = expected_anchor_fraction >= 0.90 and min(
        expected_group_fractions.values()
    ) >= 0.75
    expected_decision = (
        "PASS_TO_FIT_GEOMETRY_PREFLIGHT" if adequate else "STOP_CANDIDATE_SUPPLY_COVERAGE"
    )
    if result.get("decision") != expected_decision:
        failures.append("decision disagrees with registered supply thresholds")
    claims = _mapping(result.get("claims"), "claims")
    if claims.get("candidate_supply_adequate_on_registered_fit_sample") is not adequate:
        failures.append("candidate-supply claim mismatch")
    for key in (
        "geometry_evaluated",
        "teacher_or_prediction_read",
        "r004h4_registered",
        "development_access_authorized",
        "coverage_established",
        "gate_b_passed",
    ):
        if claims.get(key) is not False:
            failures.append(f"forbidden claim is not false: {key}")
    return {
        "experiment_id": (
            "R004H4P2R1-mechanical-result-audit-v1"
            if experiment_id == "R004H4P2R1-comparable-speed-supply-replay-v1"
            else "R004H4P2-mechanical-result-audit-v1"
        ),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "payload_files_read": 0,
        "recomputed": {
            "raw_candidate_evaluations": raw_candidates,
            "eligible_candidate_evaluations": eligible_candidates,
            "anchors_with_eligible_candidate": supplied,
            "anchor_supply_fraction": expected_anchor_fraction,
            "minimum_group_anchor_supply_fraction": min(expected_group_fractions.values()),
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
