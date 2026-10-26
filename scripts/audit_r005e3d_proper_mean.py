"""Mechanically audit R005E3D and apply its frozen proper-mean gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file
from audit_r005e2_scientific_run import _load_npz, _state_sha256
from run_r005e3a_fold_job import _temperature_from_outer_train


TIE_TOLERANCE = 1.0e-12
EXPECTED_FOLD = 0
EXPECTED_SEED = 20261025


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _summary(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
) -> dict[str, float | int]:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask shapes differ")
    if prediction.ndim != 2 or not bool(mask[:, 0].all()):
        raise ValueError("candidate tensors or native mask are invalid")
    if not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise FloatingPointError("valid candidate values must be finite")
    row = np.arange(prediction.shape[0])
    selected = np.argmax(np.where(mask, prediction, -np.inf), axis=1)
    native = np.asarray(target[:, 0], dtype=np.float64)
    selected_utility = np.asarray(target[row, selected], dtype=np.float64)
    oracle = np.asarray(
        np.max(np.where(mask, target, -np.inf), axis=1), dtype=np.float64
    )
    gain = selected_utility - native
    oracle_gain = oracle - native
    regret = oracle - selected_utility
    nonnative = selected != 0
    beneficial = nonnative & (gain > TIE_TOLERANCE)
    harmful = nonnative & (gain < -TIE_TOLERANCE)
    neutral = nonnative & ~(beneficial | harmful)
    mean_gain = float(np.mean(gain))
    mean_oracle_gain = float(np.mean(oracle_gain))
    challenger = mask.copy()
    challenger[:, 0] = False
    challenger_error = (
        np.asarray(prediction[challenger], dtype=np.float64)
        - np.asarray(target[challenger], dtype=np.float64)
    )
    return {
        "anchor_count": int(prediction.shape[0]),
        "mean_gain_over_native": mean_gain,
        "mean_oracle_gain_over_native": mean_oracle_gain,
        "mean_hard_regret": float(np.mean(regret)),
        "relation_mse": float(np.mean(np.square(challenger_error))),
        "absolute_oracle_capture": (
            mean_gain / mean_oracle_gain if mean_oracle_gain > 0.0 else 0.0
        ),
        "native_choice_fraction": float(np.mean(selected == 0)),
        "nonnative_choice_count": int(np.sum(nonnative)),
        "beneficial_nonnative_count": int(np.sum(beneficial)),
        "harmful_nonnative_count": int(np.sum(harmful)),
        "neutral_nonnative_count": int(np.sum(neutral)),
    }


def _equal_scalar(left: float | int, right: object) -> bool:
    if isinstance(left, int):
        return isinstance(right, int) and left == right
    return isinstance(right, (int, float)) and math.isclose(
        left, float(right), rel_tol=0.0, abs_tol=1.0e-12
    )


def _classify(summary: dict[str, float | int]) -> str:
    passed = (
        float(summary["mean_gain_over_native"]) > 0.0
        and float(summary["absolute_oracle_capture"]) >= 0.10
        and int(summary["beneficial_nonnative_count"]) >= 1
        and int(summary["nonnative_choice_count"]) >= 1
    )
    return (
        "PASS_PROPER_MEAN_OUTER_TRAIN_QUALIFICATION"
        if passed
        else "FAIL_FROZEN_QUALIFICATION_GATE"
    )


def _failure_reasons(summary: dict[str, float | int]) -> list[str]:
    reasons: list[str] = []
    gain = float(summary["mean_gain_over_native"])
    capture = float(summary["absolute_oracle_capture"])
    if gain <= 0.0:
        reasons.append("NO_POSITIVE_GAIN")
    elif capture < 0.10:
        reasons.append("POSITIVE_BUT_BELOW_CAPTURE_GATE")
    if int(summary["beneficial_nonnative_count"]) < 1:
        reasons.append("NO_BENEFICIAL_NONNATIVE_SELECTION")
    if int(summary["nonnative_choice_count"]) < 1:
        reasons.append("NO_NONNATIVE_SELECTION")
    return reasons


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text("utf-8")))
    frozen_data = cast(dict[str, Any], config["data"])
    primary_result = cast(
        dict[str, Any], json.loads((args.primary / "result.json").read_text("utf-8"))
    )
    replay_result = cast(
        dict[str, Any], json.loads((args.replay / "result.json").read_text("utf-8"))
    )
    expected_inputs = {
        "bundle_manifest": sha256_file(args.bundle_root / "manifest.json"),
        "crossfit_manifest": sha256_file(args.crossfit_manifest),
        "mismatch_manifest": sha256_file(args.mismatch_manifest),
        "visual_manifest": sha256_file(args.visual_root / "manifest.json"),
        "parent_job_result": sha256_file(args.parent_job_result),
        "r005e3b_control_result": sha256_file(args.control_result),
    }
    frozen_hashes = {
        "bundle_manifest": frozen_data["bundle_manifest_sha256"],
        "crossfit_manifest": frozen_data["crossfit_manifest_sha256"],
        "mismatch_manifest": frozen_data["mismatch_manifest_sha256"],
        "visual_manifest": frozen_data["visual_manifest_sha256"],
        "parent_job_result": frozen_data["parent_fold0_seed20261025_result_sha256"],
        "r005e3b_control_result": frozen_data["r005e3b_control_result_sha256"],
    }
    if expected_inputs != frozen_hashes:
        raise ValueError("runtime inputs differ from preregistered hashes")
    parent_result = cast(
        dict[str, Any], json.loads(args.parent_job_result.read_text("utf-8"))
    )
    if (
        parent_result.get("schema") != "aorta.r005e3a.fold_job.v1"
        or parent_result.get("status") != "COMPLETED"
        or parent_result.get("outer_fold") != EXPECTED_FOLD
        or parent_result.get("seed") != EXPECTED_SEED
        or parent_result.get("objective") != "soft_selection_regret"
    ):
        raise ValueError("frozen R005E3A parent contract failed")
    control_result = cast(
        dict[str, Any], json.loads(args.control_result.read_text("utf-8"))
    )
    if (
        control_result.get("schema") != "aorta.r005e3b.score_realizability.v1"
        or control_result.get("status") != "COMPLETED"
        or control_result.get("outer_fold") != EXPECTED_FOLD
        or control_result.get("seed") != EXPECTED_SEED
        or control_result.get("output_activation") != "identity"
        or control_result.get("objective") != "soft_selection_regret"
        or control_result.get("initial_model_state_sha256")
        != parent_result.get("initial_model_state_sha256")
        or control_result.get("selected_epoch") != parent_result.get("selected_epoch")
        or control_result.get("parameter_count") != parent_result.get("parameter_count")
    ):
        raise ValueError("frozen R005E3B matched-control contract failed")
    bundle_manifest = cast(
        dict[str, Any],
        json.loads((args.bundle_root / "manifest.json").read_text("utf-8")),
    )
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text("utf-8"))
    )
    fold = next(
        row
        for row in cast(list[dict[str, Any]], crossfit["outer_folds"])
        if int(row["outer_fold"]) == EXPECTED_FOLD
    )
    train_groups = cast(list[str], fold["outer_train_groups"])
    expected_temperature, expected_temperature_ledger = _temperature_from_outer_train(
        bundle_root=args.bundle_root,
        bundle_manifest=bundle_manifest,
        groups=train_groups,
    )
    for label, root, result in (
        ("primary", args.primary, primary_result),
        ("replay", args.replay, replay_result),
    ):
        if (
            result.get("schema") != "aorta.r005e3d.proper_mean_qualification.v1"
            or result.get("status") != "COMPLETED"
            or result.get("outer_fold") != EXPECTED_FOLD
            or result.get("seed") != EXPECTED_SEED
            or result.get("output_activation") != "identity"
            or result.get("objective") != "native_relative_mean_squared_error"
            or result.get("code_commit") != args.code_commit
            or result.get("runner_sha256") != sha256_file(args.runner)
            or result.get("config_sha256") != sha256_file(args.config)
            or result.get("input_sha256") != expected_inputs
            or result.get("initial_model_state_sha256")
            != parent_result.get("initial_model_state_sha256")
            or result.get("parent_initial_model_state_sha256")
            != parent_result.get("initial_model_state_sha256")
            or result.get("initial_model_state_sha256")
            != control_result.get("initial_model_state_sha256")
            or result.get("r005e3b_control_result_sha256")
            != expected_inputs["r005e3b_control_result"]
            or result.get("selected_epoch") != parent_result.get("selected_epoch")
            or result.get("optimizer_step_count")
            != int(cast(int, parent_result["selected_epoch"])) * 560
            or result.get("training_bundle_index_order_sha256")
            != control_result.get("training_bundle_index_order_sha256")
            or result.get("parameter_count") != parent_result.get("parameter_count")
            or not math.isclose(
                float(result["temperature"]),
                expected_temperature,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or result.get("temperature_ledger") != expected_temperature_ledger
            or result.get("development_calibration_test_navdream_target_accessed")
            is not False
            or result.get("scientific_metrics_emitted") is not False
        ):
            raise ValueError(f"{label} result contract failed")
        physical_gpu = cast(dict[str, Any], result["physical_gpu"])
        if int(physical_gpu["physical_index"]) != 0:
            raise ValueError(f"{label} physical GPU differs from the frozen roster")
        epoch_ledger = cast(list[dict[str, Any]], result["epoch_ledger"])
        if len(epoch_ledger) != int(parent_result["selected_epoch"]) or [
            int(row["epoch"]) for row in epoch_ledger
        ] != list(range(1, int(parent_result["selected_epoch"]) + 1)):
            raise ValueError(f"{label} epoch ledger differs from the frozen budget")
        if (
            sha256_file(root / "predictions.npz") != result["prediction_sha256"]
            or sha256_file(root / "checkpoint.pt") != result["checkpoint_sha256"]
            or _state_sha256(root / "checkpoint.pt")
            != result["final_model_state_sha256"]
        ):
            raise ValueError(f"{label} artifact provenance failed")

    ignored = {
        "completed_at_utc",
        "runtime_seconds",
        "prediction_sha256",
        "checkpoint_sha256",
    }
    if {key: value for key, value in primary_result.items() if key not in ignored} != {
        key: value for key, value in replay_result.items() if key not in ignored
    }:
        raise ValueError("primary/replay logical results differ")
    primary = _load_npz(args.primary / "predictions.npz")
    replay = _load_npz(args.replay / "predictions.npz")
    if set(primary) != set(replay) or any(
        not np.array_equal(primary[key], replay[key]) for key in primary
    ):
        raise ValueError("primary/replay predictions are not bit exact")
    expected_keys = {
        "bundle_index",
        "target",
        "candidate_mask",
        "correct_prediction",
        "S1_refit_train_mismatch_prediction",
        "removed_prediction",
    }
    if set(primary) != expected_keys:
        raise ValueError("prediction key contract differs")
    indices = cast(npt.NDArray[np.int64], primary["bundle_index"])
    target = cast(npt.NDArray[np.float32], primary["target"])
    mask = cast(npt.NDArray[np.bool_], primary["candidate_mask"])
    if indices.shape != (560,) or target.shape != (560, 7) or mask.shape != (560, 7):
        raise ValueError("fold-0 outer-train tensor shapes differ")
    bundle_target = np.load(args.bundle_root / "target_margin.npy", mmap_mode="r")
    bundle_mask = np.load(args.bundle_root / "candidate_mask.npy", mmap_mode="r")
    if not np.array_equal(target, bundle_target[indices]) or not np.array_equal(
        mask, bundle_mask[indices]
    ):
        raise ValueError("target or mask differs from the frozen bundle")
    index_rows = {
        int(row["bundle_index"]): row
        for row in cast(list[dict[str, Any]], bundle_manifest["anchors"])
    }
    actual_groups = {
        str(index_rows[int(index)]["recording_group"]) for index in indices
    }
    if actual_groups != set(train_groups):
        raise ValueError("prediction roster differs from fold-0 outer-train groups")
    group_counts = {
        group: sum(
            str(index_rows[int(index)]["recording_group"]) == group for index in indices
        )
        for group in train_groups
    }
    if set(group_counts.values()) != {16}:
        raise ValueError("outer-train groups do not contain exactly sixteen anchors")

    recomputed: dict[str, dict[str, float | int]] = {}
    for condition in ("correct", "S1_refit_train_mismatch", "removed"):
        prediction = cast(npt.NDArray[np.float32], primary[f"{condition}_prediction"])
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise FloatingPointError(f"{condition} prediction is malformed")
        recomputed[condition] = _summary(prediction, target, mask)
        reported = cast(
            dict[str, object], primary_result["condition_summaries"][condition]
        )
        for key, value in recomputed[condition].items():
            if not _equal_scalar(value, reported.get(key)):
                raise ValueError(
                    f"independent metric recomputation failed: {condition}/{key}"
                )

    correct = recomputed["correct"]
    thresholds = cast(dict[str, Any], config["thresholds"])
    positive_gain = float(correct["mean_gain_over_native"]) > 0.0
    capture = float(correct["absolute_oracle_capture"])
    minimum_capture = capture >= float(thresholds["minimum_absolute_oracle_capture"])
    beneficial = int(correct["beneficial_nonnative_count"]) >= 1
    nonnative = int(correct["nonnative_choice_count"]) >= 1
    decision = _classify(correct)
    passed = decision == "PASS_PROPER_MEAN_OUTER_TRAIN_QUALIFICATION"
    return {
        "schema": "aorta.r005e3d.proper_mean_qualification_audit.v1",
        "status": "PASS",
        "support": "SUPPORTED" if passed else "UNSUPPORTED",
        "decision": decision,
        "mechanical_gates": {
            "primary_replay_predictions_bit_exact": True,
            "primary_replay_model_states_exact": True,
            "same_physical_gpu_and_frozen_roster": True,
            "provenance_exact": True,
            "parent_and_control_initialization_parameter_epoch_and_step_budget_exact": True,
            "proper_mean_objective_exact": True,
            "temperature_recomputed_from_outer_train_only": True,
            "fold_group_target_and_mask_exact": True,
            "independent_metric_recomputation_exact": True,
            "finite_values": True,
            "source_fit_outer_train_only": True,
            "sealed_roles_accessed": False,
        },
        "scientific_gates": {
            "mean_gain_over_native_strictly_positive": positive_gain,
            "absolute_oracle_capture_at_least_0_10": minimum_capture,
            "beneficial_nonnative_count_at_least_one": beneficial,
            "nonnative_choice_count_at_least_one": nonnative,
        },
        "failure_reasons": _failure_reasons(correct),
        "condition_summaries": recomputed,
        "claim_boundary": (
            "Fold-0 outer-train optimization/realizability qualification only; not OOF "
            "generalization, visual scene binding, Differential Query, appearance "
            "robustness, NavDream, behavior, or safety evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--mismatch-manifest", type=Path, required=True)
    parser.add_argument("--parent-job-result", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = audit(arguments)
    _write_json(arguments.output, result)
    print(json.dumps(result, sort_keys=True))


