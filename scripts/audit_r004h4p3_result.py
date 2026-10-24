"""Payload-free mechanical audit for R004H4P3 fit supply."""

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
    if result.get("experiment_id") != "R004H4P3-native-offset-fit-supply-v1":
        failures.append("experiment ID mismatch")
    scope = _mapping(result.get("data_scope"), "data_scope")
    if (
        scope.get("role") != "fit"
        or scope.get("split") != "train"
        or scope.get("recording_group_count") != 8
        or scope.get("anchor_count") != 128
        or scope.get("geometry_teacher_prediction_or_image_read") is not False
        or scope.get("development_calibration_test_or_target_accessed") is not False
    ):
        failures.append("registered data scope mismatch")

    ledger = result.get("anchor_ledger")
    if not isinstance(ledger, list) or len(ledger) != 128:
        raise ValueError("anchor ledger must contain 128 rows")
    keys: set[tuple[str, int]] = set()
    group_counts: Counter[str] = Counter()
    group_supplied: Counter[str] = Counter()
    supplied = 0
    both_sign = 0
    registered = 0
    accepted = 0
    rejections: Counter[str] = Counter()
    nested_candidate_count = 0
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
        both_sign += int(row.get("has_both_signed_eligible_candidates") is True)
        registered += int(row.get("registered_candidate_count", -1))
        accepted_count = int(row.get("accepted_candidate_count", -1))
        accepted += accepted_count
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != accepted_count:
            failures.append("nested candidate count mismatch")
            continue
        nested_candidate_count += len(candidates)
        if int(row.get("eligible_candidate_count", -1)) != sum(
            int(_mapping(candidate, "candidate").get("intent_eligible") is True)
            for candidate in candidates
        ):
            failures.append("eligible candidate count mismatch")
        if int(row.get("final_exact_candidate_count", -1)) != sum(
            int(_mapping(candidate, "candidate").get("endpoint_invariant_exact") is True)
            for candidate in candidates
        ):
            failures.append("endpoint-invariant count mismatch")
        if any(
            _mapping(candidate, "candidate").get("endpoint_invariant_exact") is not True
            for candidate in candidates
        ):
            failures.append("accepted candidate violates endpoint invariant")
        rejections.update(_mapping(row.get("provider_rejection_counts"), "rejections"))
    if len(keys) != 128 or len(group_counts) != 8 or any(
        count != 16 for count in group_counts.values()
    ):
        failures.append("anchor ledger is not eight complete unique groups")
    if nested_candidate_count != accepted:
        failures.append("nested candidate ledger total mismatch")

    expected_groups = {
        group: group_supplied[group] / 16.0 for group in sorted(group_counts)
    }
    anchor_fraction = supplied / 128.0
    minimum_group = min(expected_groups.values())
    supply = _mapping(result.get("supply"), "supply")
    expected_supply = {
        "anchors_with_eligible_candidate": supplied,
        "anchor_supply_fraction": anchor_fraction,
        "group_anchor_supply_fractions": expected_groups,
        "minimum_group_anchor_supply_fraction": minimum_group,
        "anchors_with_both_signed_eligible_candidates": both_sign,
        "both_signed_anchor_supply_fraction": both_sign / 128.0,
    }
    if supply != expected_supply:
        failures.append("supply summary does not replay from anchor ledger")
    provider = _mapping(result.get("provider"), "provider")
    if (
        provider.get("registered_constructions") != registered
        or provider.get("accepted_constructions") != accepted
        or provider.get("continuous_dynamic_acceptance_fraction") != accepted / registered
        or provider.get("rejection_counts") != dict(sorted(rejections.items()))
        or sum(rejections.values()) != registered - accepted
    ):
        failures.append("provider acceptance/rejection accounting mismatch")
    invariant_failures = result.get("endpoint_invariant_failures")
    if invariant_failures != []:
        failures.append("endpoint-invariant failure ledger is not empty")

    adequate = anchor_fraction >= 0.90 and minimum_group >= 0.75 and not failures
    expected_decision = (
        "PASS_TO_REGISTER_FIT_GEOMETRY_PREFLIGHT"
        if adequate
        else "STOP_CANDIDATE_SUPPLY_COVERAGE"
    )
    if result.get("decision") != expected_decision:
        failures.append("decision disagrees with registered supply gate")
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
        "experiment_id": "R004H4P3-mechanical-result-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "payload_files_read": 0,
        "recomputed": {
            "registered_constructions": registered,
            "accepted_constructions": accepted,
            "anchors_with_eligible_candidate": supplied,
            "anchor_supply_fraction": anchor_fraction,
            "minimum_group_anchor_supply_fraction": minimum_group,
            "anchors_with_both_signed_eligible_candidates": both_sign,
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
