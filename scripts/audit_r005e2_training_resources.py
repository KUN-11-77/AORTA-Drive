"""Independent fail-closed audit of the outcome-free R005E2 training rehearsal."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

from aorta_drive.vision.feature_cache import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _positive_finite(value: object, name: str) -> float:
    number = float(cast(Any, value))
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def audit(args: argparse.Namespace) -> dict[str, Any]:
    result = cast(dict[str, Any], json.loads(args.result.read_text(encoding="utf-8")))
    if result.get("schema") != "aorta.r005e2.training_resource_rehearsal.v1":
        raise ValueError("unexpected resource-rehearsal schema")
    if result.get("status") != "PASS":
        raise ValueError("resource rehearsal did not pass")
    for field in (
        "scientific_metrics_emitted",
        "scientific_labels_accessed",
        "development_calibration_test_navdream_target_accessed",
    ):
        if result.get(field) is not False:
            raise ValueError(f"forbidden evidence access flag: {field}")
    if result.get("previous_trajectory_source") != (
        "current_native_shape_surrogate_not_scientific_history"
    ):
        raise ValueError("resource rehearsal used an unregistered history source")
    if result.get("target_source") != "exact_zero_resource_surrogate":
        raise ValueError("resource rehearsal used a scientific target")
    expected_counts = {
        "inner_fit_group_count": 28,
        "anchor_count": 448,
        "completed_training_steps": 448,
        "batch_size": 1,
    }
    for field, expected in expected_counts.items():
        if int(result.get(field, -1)) != expected:
            raise ValueError(f"resource count mismatch: {field}")
    if result.get("finite_loss_and_gradients") is not True:
        raise ValueError("loss/gradient finiteness did not pass")
    if result.get("checkpoint_tensor_state_exact") is not True:
        raise ValueError("checkpoint tensor recovery did not pass")
    if result.get("checkpoint_forward_recovery_bit_exact") is not True:
        raise ValueError("checkpoint forward replay did not pass")
    if sha256_file(args.checkpoint) != result.get("checkpoint_sha256"):
        raise ValueError("checkpoint artifact hash mismatch")

    expected_hashes = {
        "panel_roster_sha256": sha256_file(args.panel_roster),
        "crossfit_manifest_sha256": sha256_file(args.crossfit_manifest),
        "visual_manifest_sha256": sha256_file(args.visual_manifest),
        "scorer_source_sha256": sha256_file(args.scorer_source),
        "runner_sha256": sha256_file(args.runner_source),
    }
    if result.get("input_hashes") != expected_hashes:
        raise ValueError("resource-rehearsal provenance hashes differ")
    _positive_finite(result.get("peak_cuda_bytes"), "peak_cuda_bytes")
    _positive_finite(result.get("total_runtime_seconds"), "total_runtime_seconds")
    _positive_finite(
        result.get("projected_three_arm_three_seed_six_fold_gpu_hours"),
        "projected resource hours",
    )
    for section in ("data_wait_ms", "train_step_ms"):
        values = cast(dict[str, Any], result.get(section))
        mean = _positive_finite(values.get("mean"), f"{section}.mean")
        p95 = _positive_finite(values.get("p95"), f"{section}.p95")
        maximum = _positive_finite(values.get("maximum"), f"{section}.maximum")
        if p95 < mean or maximum < p95:
            raise ValueError(f"{section} quantile ordering is invalid")

    forbidden = {"relation_mse", "decision_regret", "oracle_recovery", "scientific_decision"}
    if forbidden & set(result):
        raise ValueError("resource result contains a scientific endpoint")
    return {
        "schema": "aorta.r005e2.training_resource_rehearsal.audit.v1",
        "status": "PASS",
        "result_sha256": sha256_file(args.result),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "input_provenance_exact": True,
        "registered_fold_cardinality_exact": True,
        "finite_resource_measurements": True,
        "checkpoint_recovery_exact": True,
        "scientific_metrics_emitted": False,
        "scientific_labels_accessed": False,
        "development_calibration_test_navdream_target_accessed": False,
        "scientific_training_authorized_by_this_audit": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel-roster", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument("--scorer-source", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = audit(arguments)
    _write_json(arguments.output, output)
    print(json.dumps(output, indent=2))
