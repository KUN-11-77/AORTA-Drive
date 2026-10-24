"""Fail-closed payload-free audit for R004H4P5D result JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from diagnose_r004h4p2_candidate_supply import _sha256


def audit(
    result: dict[str, Any],
    *,
    config_sha256: str,
    expected_implementation_revision: str,
) -> dict[str, Any]:
    failures: list[str] = []
    if result.get("experiment_id") != "R004H4P5D-disk-resolution-diagnosis-v1":
        failures.append("unexpected experiment_id")
    if result.get("status") != "COMPLETED":
        failures.append("result status is not COMPLETED")
    if result.get("implementation_revision") != expected_implementation_revision:
        failures.append("implementation revision mismatch")
    if result.get("config_sha256") != config_sha256:
        failures.append("config hash mismatch")
    elapsed = result.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        failures.append("elapsed time is not positive finite")

    rows = result.get("sentinel_ledger")
    observed = 0
    explained = 0
    maximum_separation = 0.0
    minimum_barrier = math.inf
    if not isinstance(rows, list) or len(rows) != 8:
        failures.append("sentinel ledger must contain eight rows")
        rows = []
    keys: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            failures.append("sentinel row is not a mapping")
            continue
        key = (str(row.get("log_name")), str(row.get("scene_token")))
        if key in keys:
            failures.append("sentinel rows are not unique")
        keys.add(key)
        if row.get("execution_deterministic_replay_bit_exact") is not True:
            failures.append(f"{key}: deterministic replay failed")
        candidates = row.get("candidate_ledger")
        if not isinstance(candidates, list) or len(candidates) != 6:
            failures.append(f"{key}: candidate ledger must contain six rows")
            continue
        amplitudes: list[float] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                failures.append(f"{key}: candidate row is not a mapping")
                continue
            amplitude = candidate.get("amplitude_m")
            separation = candidate.get("maximum_paired_separation_upper_m")
            barrier = candidate.get("no_witness_threshold_lower_m")
            if (
                not isinstance(amplitude, (int, float))
                or not isinstance(separation, (int, float))
                or not isinstance(barrier, (int, float))
                or not math.isfinite(amplitude)
                or not math.isfinite(separation)
                or not math.isfinite(barrier)
            ):
                failures.append(f"{key}: non-finite candidate diagnostic")
                continue
            amplitude_value = float(amplitude)
            separation_value = float(separation)
            barrier_value = float(barrier)
            amplitudes.append(amplitude_value)
            if separation_value < 0 or barrier_value < 0:
                failures.append(f"{key}: negative candidate diagnostic")
            recomputed_guarantee = separation_value <= barrier_value
            if candidate.get("barrier_guarantees_zero_witnesses") is not recomputed_guarantee:
                failures.append(f"{key}: barrier comparison mismatch")
            prior_zero = candidate.get("predecessor_bidirectional_zero") is True
            recomputed_explained = prior_zero and recomputed_guarantee
            if candidate.get("barrier_explains_predecessor_zero") is not recomputed_explained:
                failures.append(f"{key}: predecessor explanation mismatch")
            if prior_zero and candidate.get("predecessor_geometry_valid") is not True:
                failures.append(f"{key}: zero ledger lacks prior geometry validity")
            observed += int(prior_zero)
            explained += int(recomputed_explained)
            maximum_separation = max(maximum_separation, separation_value)
            minimum_barrier = min(minimum_barrier, barrier_value)
        if sorted(amplitudes) != [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
            failures.append(f"{key}: amplitude ledger mismatch")

    counts = result.get("diagnostic_counts", {})
    if counts.get("predecessor_bidirectional_zero_candidates") != observed:
        failures.append("observed-zero total mismatch")
    if counts.get("zero_candidates_explained_by_barrier") != explained:
        failures.append("explained-zero total mismatch")
    if observed != 36:
        failures.append("predecessor zero ledger is not the registered 36 candidates")
    if result.get("maximum_paired_separation_upper_m") != maximum_separation:
        failures.append("maximum separation summary mismatch")
    if result.get("minimum_no_witness_threshold_lower_m") != minimum_barrier:
        failures.append("minimum barrier summary mismatch")
    implementation_passed = not result.get("implementation_failures") and len(rows) == 8
    all_explained = observed > 0 and explained == observed
    expected_decision = (
        "RETIRE_DISK_NONINCLUSION_AND_QUALIFY_ORIENTED_RECTANGLE"
        if implementation_passed and all_explained
        else "STOP_UNEXPLAINED_DISK_ZERO"
        if implementation_passed
        else "STOP_DISK_RESOLUTION_DIAGNOSIS_IMPLEMENTATION"
    )
    if result.get("decision") != expected_decision:
        failures.append("decision mismatch")
    claims = result.get("claims", {})
    expected_verified = implementation_passed and all_explained
    if claims.get("disk_certificate_resolution_barrier_verified") is not expected_verified:
        failures.append("barrier claim mismatch")
    for forbidden in (
        "physical_opportunity_absent",
        "prevalence_estimated",
        "method_performance_evaluated",
        "coverage_established",
        "gate_b_passed",
        "development_access_authorized",
    ):
        if claims.get(forbidden) is not False:
            failures.append(f"forbidden claim is not false: {forbidden}")
    return {
        "experiment_id": "R004H4P5D-mechanical-result-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "payload_files_read": 0,
        "recomputed": {
            "predecessor_bidirectional_zero_candidates": observed,
            "zero_candidates_explained_by_barrier": explained,
            "maximum_paired_separation_upper_m": maximum_separation,
            "minimum_no_witness_threshold_lower_m": minimum_barrier,
        },
        "external_cross_model_review": "REVIEW_UNAVAILABLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-implementation-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    audited = audit(
        result,
        config_sha256=_sha256(args.config),
        expected_implementation_revision=args.expected_implementation_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audited, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audited, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if audited["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
