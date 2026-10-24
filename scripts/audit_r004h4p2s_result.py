"""Payload-free mechanical audit for the R004H4P2S result ledger."""

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
    if result.get("experiment_id") != "R004H4P2S-native-speed-semantics-audit-v1":
        failures.append("experiment ID mismatch")
    scope = _mapping(result.get("data_scope"), "data_scope")
    if (
        scope.get("role") != "fit"
        or scope.get("split") != "train"
        or scope.get("recording_group_count") != 8
        or scope.get("anchor_count") != 128
        or scope.get("development_calibration_test_or_target_accessed") is not False
    ):
        failures.append("registered data scope mismatch")
    source_hashes = _mapping(result.get("source_sha256"), "source_sha256")
    if len(source_hashes) != 2 or any(
        not isinstance(value, str) or len(value) != 64 for value in source_hashes.values()
    ):
        failures.append("NAVSIM source hashes are incomplete")

    ledger = result.get("anchor_ledger")
    if not isinstance(ledger, list) or len(ledger) != 128:
        raise ValueError("anchor_ledger must contain exactly 128 rows")
    keys: set[tuple[str, int]] = set()
    material = 0
    moving = 0
    zero_suspect = 0
    maximum = 0.0
    for value in ledger:
        row = _mapping(value, "anchor row")
        group = row.get("recording_group")
        index = row.get("anchor_index")
        if not isinstance(group, str) or not isinstance(index, int):
            raise ValueError("anchor row lacks group/index")
        keys.add((group, index))
        suspect = float(row["cached_dynamic_terminal_speed_mps"])
        reference = float(row["position_derived_final_segment_average_speed_mps"])
        disagreement = float(row["absolute_disagreement_mps"])
        if not all(math.isfinite(item) and item >= 0.0 for item in (suspect, reference, disagreement)):
            failures.append("non-finite or negative speed metric")
            continue
        if disagreement != abs(reference - suspect):
            failures.append("speed disagreement arithmetic mismatch")
        expected_material = reference >= 0.5 and disagreement > 1.0
        if row.get("material_moving_disagreement") is not expected_material:
            failures.append("material-disagreement flag mismatch")
        material += int(expected_material)
        moving += int(reference >= 0.5)
        zero_suspect += int(suspect == 0.0)
        maximum = max(maximum, disagreement)
    if len(keys) != 128:
        failures.append("anchor ledger keys are not unique")

    summary = _mapping(result.get("summary"), "summary")
    expected_summary = {
        "moving_reference_anchor_count": moving,
        "zero_cached_dynamic_speed_anchor_count": zero_suspect,
        "material_moving_disagreement_anchor_count": material,
        "maximum_absolute_disagreement_mps": maximum,
    }
    if summary != expected_summary:
        failures.append("summary does not replay from ledger")
    expected_decision = (
        "REJECT_CACHED_DYNAMIC_SPEED_SEMANTICS"
        if material > 0
        else "RETAIN_CACHED_DYNAMIC_SPEED_SEMANTICS"
    )
    if result.get("decision") != expected_decision:
        failures.append("decision disagrees with registered rule")
    claims = _mapping(result.get("claims"), "claims")
    if claims.get("cached_dynamic_terminal_speed_is_valid_for_h4_intent_matching") is not (
        material == 0
    ):
        failures.append("speed-validity claim mismatch")
    for key in (
        "position_derived_quantity_is_instantaneous_speed",
        "candidate_provider_changed",
        "intent_threshold_changed",
        "geometry_evaluated",
        "teacher_or_prediction_read",
        "development_access_authorized",
        "h4_registered",
    ):
        if claims.get(key) is not False:
            failures.append(f"forbidden claim is not false: {key}")
    return {
        "experiment_id": "R004H4P2S-mechanical-result-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "payload_files_read": 0,
        "recomputed": expected_summary,
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
