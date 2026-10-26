"""Mechanically audit R005E3B and apply its frozen realizability gate."""

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
    return {
        "anchor_count": int(prediction.shape[0]),
        "mean_gain_over_native": mean_gain,
        "mean_oracle_gain_over_native": mean_oracle_gain,
        "mean_hard_regret": float(np.mean(regret)),
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
    return "PASS_LINEAR_REALIZABILITY" if passed else "FAIL_LINEAR_REALIZABILITY"


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
    }
    frozen_hashes = {
        "bundle_manifest": frozen_data["bundle_manifest_sha256"],
        "crossfit_manifest": frozen_data["crossfit_manifest_sha256"],
        "mismatch_manifest": frozen_data["mismatch_manifest_sha256"],
        "visual_manifest": frozen_data["visual_manifest_sha256"],
        "parent_job_result": frozen_data["parent_fold0_seed20261025_result_sha256"],
    }
    if expected_inputs != frozen_hashes:
        raise ValueError("runtime inputs differ from preregistered hashes")
    for label, root, result in (
        ("primary", args.primary, primary_result),
        ("replay", args.replay, replay_result),
    ):
        if (
            result.get("schema") != "aorta.r005e3b.score_realizability.v1"
            or result.get("status") != "COMPLETED"
            or result.get("outer_fold") != EXPECTED_FOLD
            or result.get("seed") != EXPECTED_SEED
            or result.get("output_activation") != "identity"
            or result.get("objective") != "soft_selection_regret"
            or result.get("code_commit") != args.code_commit
            or result.get("runner_sha256") != sha256_file(args.runner)
            or result.get("config_sha256") != sha256_file(args.config)
            or result.get("input_sha256") != expected_inputs
            or result.get("development_calibration_test_navdream_target_accessed")
            is not False
            or result.get("scientific_metrics_emitted") is not False
        ):
            raise ValueError(f"{label} result contract failed")
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
    positive_gain = float(correct["mean_gain_over_native"]) > 0.0
    capture = float(correct["absolute_oracle_capture"])
    minimum_capture = capture >= 0.10
    beneficial = int(correct["beneficial_nonnative_count"]) >= 1
    nonnative = int(correct["nonnative_choice_count"]) >= 1
    decision = _classify(correct)
    passed = decision == "PASS_LINEAR_REALIZABILITY"
    return {
        "schema": "aorta.r005e3b.score_realizability_audit.v1",
        "status": "PASS",
        "support": "SUPPORTED" if passed else "UNSUPPORTED",
        "decision": decision,
        "mechanical_gates": {
            "primary_replay_predictions_bit_exact": True,
            "primary_replay_model_states_exact": True,
            "provenance_exact": True,
            "fold_target_and_mask_exact": True,
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
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = audit(arguments)
    _write_json(arguments.output, result)
    print(json.dumps(result, sort_keys=True))
