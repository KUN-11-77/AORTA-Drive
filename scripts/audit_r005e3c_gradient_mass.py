"""Strictly audit R005E3C primary/replay and independently recompute final scores."""

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


IGNORED_REPLAY_FIELDS = {"completed_at_utc"}
EXPECTED_SCHEMA = "aorta.r005e3b.gradient_mass_diagnostic.v1"
EXPECTED_FOLD = 0
EXPECTED_SEED = 20261025
EXPECTED_TEMPERATURE = 0.125
EXPECTED_ANCHORS = 560
TIE_TOLERANCE = 1.0e-12


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text("utf-8")))


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return False


def _replay_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items() if key not in IGNORED_REPLAY_FIELDS
    }


def _category_summary(
    probability: npt.NDArray[np.float64],
    gradient: npt.NDArray[np.float64],
    target: npt.NDArray[np.float32],
    category: npt.NDArray[np.bool_],
) -> dict[str, float | int]:
    count = int(np.sum(category))
    utility = np.asarray(target[category], dtype=np.float64)
    values = gradient[category]
    return {
        "count": count,
        "utility_sum": float(np.sum(utility)),
        "utility_abs_sum": float(np.sum(np.abs(utility))),
        "probability_sum": float(np.sum(probability[category])),
        "score_gradient_sum": float(np.sum(values)),
        "score_gradient_abs_sum": float(np.sum(np.abs(values))),
        "score_gradient_abs_mean": (
            float(np.mean(np.abs(values))) if count else 0.0
        ),
    }


def _score_diagnostics_from_npz(
    prediction_path: Path, *, expected_sha256: str, temperature: float
) -> dict[str, Any]:
    if sha256_file(prediction_path) != expected_sha256:
        raise ValueError("R005E3B prediction bytes differ from the frozen hash")
    with np.load(prediction_path, allow_pickle=False) as payload:
        prediction = np.asarray(payload["correct_prediction"], dtype=np.float32)
        target = np.asarray(payload["target"], dtype=np.float32)
        mask = np.asarray(payload["candidate_mask"], dtype=np.bool_)
    if (
        prediction.shape != (EXPECTED_ANCHORS, 7)
        or target.shape != prediction.shape
        or mask.shape != prediction.shape
        or not bool(mask[:, 0].all())
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
        or not np.equal(prediction[:, 0], 0.0).all()
        or not np.equal(target[:, 0], 0.0).all()
    ):
        raise ValueError("frozen prediction tensors violate the panel contract")
    logits = np.where(mask, prediction.astype(np.float64) / temperature, -np.inf)
    logits -= np.max(logits, axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= np.sum(probability, axis=1, keepdims=True)
    utility = target.astype(np.float64)
    expected = np.sum(
        probability * np.where(mask, utility, 0.0), axis=1, keepdims=True
    )
    gradient = probability / temperature * (expected - utility)
    challenger = mask.copy()
    challenger[:, 0] = False
    positive = challenger & (target > target[:, :1] + TIE_TOLERANCE)
    nonpositive = challenger & ~positive
    positive_gradient = gradient[positive]
    nonpositive_gradient = gradient[nonpositive]
    count = int(np.sum(challenger))
    opportunity = np.any(positive, axis=1)
    return {
        "anchor_count": EXPECTED_ANCHORS,
        "valid_challenger_count": count,
        "opportunity_anchor_count": int(np.sum(opportunity)),
        "opportunity_anchor_fraction": float(np.mean(opportunity)),
        "positive_challenger_fraction": float(np.sum(positive) / count),
        "positive": _category_summary(probability, gradient, target, positive),
        "nonpositive": _category_summary(
            probability, gradient, target, nonpositive
        ),
        "positive_raise_direction_fraction": float(
            np.mean(positive_gradient < 0.0)
        ),
        "nonpositive_lower_direction_fraction": float(
            np.mean(nonpositive_gradient > 0.0)
        ),
    }


def _close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _close(left[key], right[key]) for key in left
        )
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)
    return bool(left == right)


def _required_science_contract(result: dict[str, Any]) -> bool:
    input_sha = result.get("input_sha256")
    initial = result.get("initial_score_diagnostics")
    final = result.get("final_score_diagnostics")
    geometry = result.get("initial_parameter_gradient_decomposition")
    if not all(isinstance(value, dict) for value in (input_sha, initial, final, geometry)):
        return False
    initial = cast(dict[str, Any], initial)
    final = cast(dict[str, Any], final)
    geometry = cast(dict[str, Any], geometry)
    try:
        for score in (initial, final):
            positive = cast(dict[str, Any], score["positive"])
            nonpositive = cast(dict[str, Any], score["nonpositive"])
            positive_count = int(positive["count"])
            nonpositive_count = int(nonpositive["count"])
            valid_count = int(score["valid_challenger_count"])
            if (
                int(score["anchor_count"]) != EXPECTED_ANCHORS
                or positive_count < 0
                or nonpositive_count < 0
                or positive_count + nonpositive_count != valid_count
                or not math.isclose(
                    float(score["positive_challenger_fraction"]),
                    positive_count / valid_count,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                or not math.isclose(
                    float(score["opportunity_anchor_fraction"]),
                    int(score["opportunity_anchor_count"]) / EXPECTED_ANCHORS,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
            ):
                return False
            for category in (positive, nonpositive):
                count = int(category["count"])
                if not math.isclose(
                    float(category["score_gradient_abs_sum"]),
                    count * float(category["score_gradient_abs_mean"]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    return False
        positive_norm = float(geometry["positive_parameter_gradient_norm"])
        nonpositive_norm = float(geometry["nonpositive_parameter_gradient_norm"])
        return (
            int(geometry["anchor_count"]) == EXPECTED_ANCHORS
            and positive_norm > 0.0
            and nonpositive_norm >= 0.0
            and math.isclose(
                float(geometry["nonpositive_to_positive_norm_ratio"]),
                nonpositive_norm / positive_norm,
                rel_tol=1.0e-7,
                abs_tol=1.0e-7,
            )
            and -1.0 <= float(geometry["positive_nonpositive_cosine"]) <= 1.0
            and float(geometry["combined_parameter_gradient_norm"]) >= 0.0
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def audit(
    primary: dict[str, Any],
    replay: dict[str, Any],
    *,
    config_sha256: str,
    runner_sha256: str,
    expected_commit: str,
    frozen_data: dict[str, Any],
    frozen_initialization_sha256: str,
    recomputed_final: dict[str, Any],
) -> dict[str, Any]:
    expected_input_sha = {
        "bundle_manifest_sha256": frozen_data["bundle_manifest_sha256"],
        "visual_manifest_sha256": frozen_data["visual_manifest_sha256"],
        "crossfit_manifest_sha256": frozen_data["crossfit_manifest_sha256"],
        "mismatch_manifest_sha256": frozen_data["mismatch_manifest_sha256"],
        "r005e3b_primary_predictions_sha256": frozen_data[
            "r005e3b_primary_predictions_sha256"
        ],
        "config": config_sha256,
    }
    physical_gpu = primary.get("physical_gpu")
    gates = {
        "schema_exact": primary.get("schema") == replay.get("schema") == EXPECTED_SCHEMA,
        "status_complete": primary.get("status") == replay.get("status") == "COMPLETED",
        "fold_seed_temperature_exact": (
            primary.get("outer_fold") == replay.get("outer_fold") == EXPECTED_FOLD
            and primary.get("seed") == replay.get("seed") == EXPECTED_SEED
            and primary.get("temperature")
            == replay.get("temperature")
            == EXPECTED_TEMPERATURE
        ),
        "initialization_exact": primary.get("initial_model_state_sha256")
        == replay.get("initial_model_state_sha256")
        == frozen_initialization_sha256,
        "input_hashes_exact": primary.get("input_sha256")
        == replay.get("input_sha256")
        == expected_input_sha,
        "primary_replay_exact_except_timestamp": _replay_payload(primary)
        == _replay_payload(replay),
        "required_scientific_contract": _required_science_contract(primary)
        and _required_science_contract(replay),
        "final_score_diagnostics_independently_recomputed": _close(
            primary.get("final_score_diagnostics"), recomputed_final
        )
        and _close(replay.get("final_score_diagnostics"), recomputed_final),
        "finite_values": _finite(primary) and _finite(replay),
        "weights_never_updated": primary.get("weights_updated") is False
        and replay.get("weights_updated") is False,
        "model_state_unchanged": primary.get("model_state_unchanged") is True
        and replay.get("model_state_unchanged") is True,
        "sealed_roles_untouched": primary.get(
            "development_calibration_test_navdream_target_accessed"
        )
        is False
        and replay.get("development_calibration_test_navdream_target_accessed")
        is False,
        "config_hash_exact": primary.get("input_sha256", {}).get("config")
        == replay.get("input_sha256", {}).get("config")
        == config_sha256,
        "runner_hash_exact": primary.get("runner_sha256")
        == replay.get("runner_sha256")
        == runner_sha256,
        "code_commit_exact": primary.get("code_commit")
        == replay.get("code_commit")
        == expected_commit,
        "same_frozen_physical_gpu": isinstance(physical_gpu, dict)
        and physical_gpu == replay.get("physical_gpu")
        and physical_gpu.get("physical_index") == 0,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        "schema": "aorta.r005e3c.gradient_mass_audit.v2",
        "status": status,
        "decision": (
            "MECHANICAL_PASS_WITH_ARCHIVE_LIMITATION__DESCRIPTIVE_ONLY"
            if status == "PASS"
            else "INVALID_DIAGNOSTIC"
        ),
        "mechanical_gates": gates,
        "archive_limitation": (
            "Initialization predictions and parameter-gradient tensors were not archived; "
            "their reported geometry is checked for required fields and mathematical "
            "self-consistency but cannot be independently recomputed from this compact archive."
        ),
        "scientific_result_gate": "NOT_DEFINED_BY_PROTOCOL",
        "claim_boundary": (
            "Read-only fold-0 outer-train soft-regret optimization diagnosis only; no "
            "method PASS, causal attribution, training unlock, OOF, visual binding, "
            "architecture, appearance, behavior, or safety claim."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--final-predictions", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text("utf-8")))
    frozen_data = cast(dict[str, Any], config["data"])
    frozen_model = cast(dict[str, Any], config["model"])
    recomputed = _score_diagnostics_from_npz(
        args.final_predictions,
        expected_sha256=str(frozen_data["r005e3b_primary_predictions_sha256"]),
        temperature=EXPECTED_TEMPERATURE,
    )
    result = audit(
        _read_json(args.primary),
        _read_json(args.replay),
        config_sha256=sha256_file(args.config),
        runner_sha256=sha256_file(args.runner),
        expected_commit=args.expected_commit,
        frozen_data=frozen_data,
        frozen_initialization_sha256=str(frozen_model["initialization_sha256"]),
        recomputed_final=recomputed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 2)
