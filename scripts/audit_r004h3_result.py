"""Mechanically audit R004H3 without reading feature or label payloads."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.calibration.heterogeneous_proxy import CONTROL_NAMES
from aorta_drive.vision.feature_cache import sha256_file


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-script", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--stage-p-checkpoint", type=Path, required=True)
    parser.add_argument("--scale-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()

    config = _mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")), "config")
    result = _mapping(json.loads(args.result.read_text(encoding="utf-8")), "result")
    if result.get("experiment_id") != config.get("experiment_id"):
        raise ValueError("experiment id mismatch")
    if result.get("status") != "COMPLETED" or result.get("decision") != "PASS":
        raise ValueError("H3 did not record the expected completed PASS")
    if result.get("implementation_revision") != args.implementation_revision:
        raise ValueError("implementation revision mismatch")
    if result.get("config_sha256") != sha256_file(args.config):
        raise ValueError("config hash mismatch")
    if result.get("script_sha256") != sha256_file(args.run_script):
        raise ValueError("run script hash mismatch")
    if result.get("stage_p_checkpoint_sha256") != sha256_file(args.stage_p_checkpoint):
        raise ValueError("Stage-P checkpoint hash mismatch")
    if result.get("scale_checkpoint_sha256") != sha256_file(args.scale_checkpoint):
        raise ValueError("scale checkpoint hash mismatch")
    if result.get("proxy_group_count") != 16 or result.get("proxy_anchor_count") != 256:
        raise ValueError("proxy scope disagrees with registration")
    if result.get("payload_read_passes") != 1:
        raise ValueError("proxy payloads were not read exactly once")
    if result.get("cached_score_replay_passes") != 2:
        raise ValueError("cached session scores were not replayed twice")

    ledger = result.get("score_ledger")
    if not isinstance(ledger, list) or len(ledger) != 16:
        raise ValueError("score ledger must contain 16 complete groups")
    if len({row.get("recording_group") for row in ledger if isinstance(row, dict)}) != 16:
        raise ValueError("score ledger group IDs are not unique")
    scores_by_control: dict[str, list[float]] = {name: [] for name in CONTROL_NAMES}
    for item in ledger:
        row = _mapping(item, "score ledger row")
        if row.get("anchor_count") != 16:
            raise ValueError("every proxy group must contain all 16 anchors")
        scores = _mapping(row.get("session_max_scores"), "session scores")
        if set(scores) != set(CONTROL_NAMES):
            raise ValueError("group score row does not contain all controls")
        for name in CONTROL_NAMES:
            value = _finite(scores[name], f"{name} session score")
            if value < 0.0:
                raise ValueError("session scores must be nonnegative")
            scores_by_control[name].append(value)

    control_results = _mapping(result.get("control_results"), "control results")
    if set(control_results) != set(CONTROL_NAMES):
        raise ValueError("control results do not contain all controls")
    replayed: dict[str, dict[str, float | int | bool]] = {}
    for name in CONTROL_NAMES:
        control = _mapping(control_results[name], f"{name} result")
        if control.get("status") != "FINITE":
            raise ValueError(f"{name} did not remain finite")
        if control.get("valid_group_count") != 16 or control.get("invalid_group_count") != 0:
            raise ValueError(f"{name} group counts disagree with PASS")
        values = scores_by_control[name]
        expected_threshold = max(values)
        if control.get("rank") != 16:
            raise ValueError(f"{name} rank is not 16")
        if _finite(control.get("threshold"), f"{name} threshold") != expected_threshold:
            raise ValueError(f"{name} threshold is not the complete-session maximum")
        if control.get("threshold_equals_maximum") is not True:
            raise ValueError(f"{name} maximum flag disagrees with replay")
        expected_statistics = {
            "minimum_session_score": min(values),
            "median_session_score": median(values),
            "maximum_session_score": max(values),
        }
        for key, expected in expected_statistics.items():
            if _finite(control.get(key), f"{name} {key}") != expected:
                raise ValueError(f"{name} {key} disagrees with replay")
        budget_min = _finite(control.get("edge_budget_minimum_m"), f"{name} budget min")
        budget_median = _finite(
            control.get("edge_budget_median_m"), f"{name} budget median"
        )
        budget_max = _finite(control.get("edge_budget_maximum_m"), f"{name} budget max")
        if not 0.0 <= budget_min <= budget_median <= budget_max:
            raise ValueError(f"{name} budget ordering is invalid")
        replayed[name] = {
            "rank": 16,
            "threshold": expected_threshold,
            "threshold_equals_maximum": True,
        }

    for field in (
        "trainable_parameters",
        "parameter_updates",
    ):
        if result.get(field) != 0:
            raise ValueError(f"{field} must be zero")
    for field in ("checkpoint_written", "prohibited_payload_accessed"):
        if result.get(field) is not False:
            raise ValueError(f"{field} must be false")
    if result.get("h4_registration_authorized") is not True:
        raise ValueError("H4 registration authorization disagrees with PASS")
    if result.get("coverage_established") is not False:
        raise ValueError("proxy rehearsal cannot establish coverage")
    if result.get("official_calibration_access_authorized") is not False:
        raise ValueError("proxy rehearsal cannot authorize official calibration")
    resources = config["resources"]
    if _finite(result.get("peak_cuda_bytes"), "peak CUDA bytes") / 1024**3 > float(
        resources["maximum_peak_gpu_gib"]
    ):
        raise ValueError("GPU cap exceeded")
    if _finite(result.get("elapsed_seconds"), "elapsed seconds") / 60.0 > float(
        resources["maximum_wall_minutes"]
    ):
        raise ValueError("wall-time cap exceeded")

    audit = {
        "experiment_id": "R004H3-result-mechanical-audit-v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "raw_result_sha256": sha256_file(args.result),
        "replayed_controls": replayed,
        "replayed_group_count": 16,
        "replayed_anchor_count": 256,
        "payload_files_read": 0,
        "checks": {
            "source_and_checkpoint_hashes": True,
            "complete_session_scope": True,
            "independent_rank_16_thresholds": True,
            "finite_budget_diagnostics": True,
            "no_training_or_checkpoint_write": True,
            "claim_boundary": True,
            "resource_caps": True,
        },
        "claim_boundary": "Mechanical proxy-workflow audit only; no coverage claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
