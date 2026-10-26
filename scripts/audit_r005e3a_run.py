"""Audit R005E3A exact replay and apply its frozen group-level decision rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file
from audit_r005e2_scientific_run import _anchor_metrics, _load_npz, _state_sha256
from run_r005e3a_fold_job import (
    PARENT_COMMIT,
    PARENT_DECISION,
    SEEDS,
    TIE_TOLERANCE,
    _temperature_from_outer_train,
)


CONDITIONS = (
    "correct",
    "S1_primary_geometry",
    "S2_disjoint_geometry",
    "S3_hash_random",
    "removed",
)
METRICS = ("relation_mse", "decision_regret", "sign_error", "boundary_error")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _bootstrap(
    values: npt.NDArray[np.float64], *, seed: int, replicates: int
) -> list[float]:
    if values.shape != (42,):
        raise ValueError("bootstrap requires exactly 42 paired group values")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, 42, size=(replicates, 42))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _classify(
    *,
    objective_supported: bool,
    minimum_effect_supported: bool,
    binding_supported: bool,
) -> str:
    if objective_supported and minimum_effect_supported and binding_supported:
        return "GO_TO_FULL_OBJECTIVE_BY_VISUAL_ARM_CONFIRMATION_PREREGISTRATION"
    if objective_supported and minimum_effect_supported:
        return "MIXED_DECISION_IMPROVEMENT_WITHOUT_COMPLETE_SCENE_BINDING"
    return "STOP_SOFT_SELECTION_REGRET_OBJECTIVE"


def _choice(prediction: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]) -> int:
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        raise ValueError("candidate set has no legal member")
    return int(valid[np.argmax(prediction[valid])])


def audit(args: argparse.Namespace) -> dict[str, Any]:
    progress = cast(
        dict[str, Any], json.loads((args.run_root / "progress.json").read_text("utf-8"))
    )
    if progress.get("status") != "PRIMARY_REPLAY_COMPLETED":
        raise ValueError("primary/replay chain is incomplete")
    if (
        progress.get("development_calibration_test_navdream_target_accessed")
        is not False
        or progress.get("scientific_metrics_emitted") is not False
    ):
        raise ValueError("sealed-role or partial-metric progress contract failed")
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text("utf-8")))
    contract = cast(
        dict[str, Any],
        json.loads((args.run_root / "run_contract.json").read_text("utf-8")),
    )
    if (
        contract.get("phase_order") != ["primary", "replay"]
        or contract.get("job_count_per_phase") != 18
        or contract.get("same_job_gpu_for_replay") is not True
    ):
        raise ValueError("queue contract differs from preregistration")
    expected_inputs = {
        "config": sha256_file(args.config),
        "runner": sha256_file(args.runner),
        "bundle_manifest": sha256_file(args.bundle_root / "manifest.json"),
        "crossfit_manifest": sha256_file(args.crossfit_manifest),
        "mismatch_manifest": sha256_file(args.mismatch_manifest),
        "sensitivity_manifest": sha256_file(args.sensitivity_manifest),
        "parent_audit": sha256_file(args.parent_run_root / "audit.json"),
    }
    if contract.get("input_sha256") != expected_inputs:
        raise ValueError("run input provenance differs from current files")
    frozen_data = cast(dict[str, Any], config["data"])
    if (
        expected_inputs["crossfit_manifest"] != frozen_data["crossfit_manifest_sha256"]
        or expected_inputs["mismatch_manifest"]
        != frozen_data["primary_mismatch_manifest_sha256"]
        or expected_inputs["sensitivity_manifest"]
        != frozen_data["sensitivity_manifest_sha256"]
    ):
        raise ValueError("input manifest differs from the preregistered hashes")
    parent_audit = cast(
        dict[str, Any],
        json.loads((args.parent_run_root / "audit.json").read_text("utf-8")),
    )
    if (
        parent_audit.get("status") != "PASS"
        or parent_audit.get("support") != "UNSUPPORTED"
        or parent_audit.get("decision") != PARENT_DECISION
    ):
        raise ValueError("parent audit is not the frozen R005E2B STOP")

    bundle = cast(
        dict[str, Any],
        json.loads((args.bundle_root / "manifest.json").read_text("utf-8")),
    )
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text("utf-8"))
    )
    index_rows = {
        int(row["bundle_index"]): row
        for row in cast(list[dict[str, Any]], bundle["anchors"])
    }
    bundle_target = np.load(args.bundle_root / "target_margin.npy", mmap_mode="r")
    bundle_mask = np.load(args.bundle_root / "candidate_mask.npy", mmap_mode="r")
    ordered_groups = sorted(cast(dict[str, Any], crossfit["group_outer_fold"]))
    if len(ordered_groups) != 42:
        raise ValueError("cross-fit roster must contain 42 groups")
    assignment = cast(
        dict[str, int],
        json.loads((args.run_root / "gpu_assignment.json").read_text("utf-8")),
    )
    expected_jobs = {f"fold{fold}_seed{seed}" for fold in range(6) for seed in SEEDS}
    if set(assignment) != expected_jobs:
        raise ValueError("GPU assignment roster is incomplete")

    group_metrics: dict[tuple[str, int, str], dict[str, list[float]]] = {}
    seen: dict[tuple[str, int], set[int]] = {
        (condition, seed): set()
        for condition in ("parent", *CONDITIONS)
        for seed in SEEDS
    }
    choice_changes = {"beneficial": 0, "harmful": 0, "neutral": 0}
    intervention_flips = {
        condition: 0 for condition in CONDITIONS if condition != "correct"
    }
    job_audit: list[dict[str, Any]] = []
    expected_parameter_count: int | None = None

    for fold in range(6):
        fold_config = next(
            row
            for row in cast(list[dict[str, Any]], crossfit["outer_folds"])
            if int(row["outer_fold"]) == fold
        )
        expected_temperature, expected_temperature_ledger = (
            _temperature_from_outer_train(
                bundle_root=args.bundle_root,
                bundle_manifest=bundle,
                groups=cast(list[str], fold_config["outer_train_groups"]),
            )
        )
        for seed in SEEDS:
            name = f"fold{fold}_seed{seed}"
            primary_root = args.run_root / "primary" / name
            replay_root = args.run_root / "replay" / name
            primary_result = cast(
                dict[str, Any],
                json.loads((primary_root / "result.json").read_text("utf-8")),
            )
            replay_result = cast(
                dict[str, Any],
                json.loads((replay_root / "result.json").read_text("utf-8")),
            )
            parent_root = (
                args.parent_run_root / "primary" / f"fold{fold}_V2_correct_seed{seed}"
            )
            parent_result = cast(
                dict[str, Any],
                json.loads((parent_root / "result.json").read_text("utf-8")),
            )
            for result in (primary_result, replay_result):
                if (
                    result.get("status") != "COMPLETED"
                    or result.get("schema") != "aorta.r005e3a.fold_job.v1"
                    or result.get("code_commit") != contract.get("code_commit")
                    or result.get("outer_fold") != fold
                    or result.get("seed") != seed
                    or result.get("objective") != "soft_selection_regret"
                    or result.get("input_sha256")
                    != {
                        "bundle_manifest": expected_inputs["bundle_manifest"],
                        "crossfit_manifest": expected_inputs["crossfit_manifest"],
                        "mismatch_manifest": expected_inputs["mismatch_manifest"],
                        "sensitivity_manifest": expected_inputs["sensitivity_manifest"],
                        "parent_audit": expected_inputs["parent_audit"],
                    }
                    or result.get(
                        "development_calibration_test_navdream_target_accessed"
                    )
                    is not False
                    or result.get("scientific_metrics_emitted") is not False
                ):
                    raise ValueError(f"job contract failed: {name}")
                physical = cast(dict[str, Any], result["physical_gpu"])
                if int(physical["physical_index"]) != int(assignment[name]):
                    raise ValueError(f"same-GPU replay failed: {name}")
                if (
                    not math.isclose(
                        float(result["temperature"]),
                        expected_temperature,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                    or result["temperature_ledger"] != expected_temperature_ledger
                ):
                    raise ValueError(f"temperature recomputation failed: {name}")
                if (
                    parent_result.get("code_commit") != PARENT_COMMIT
                    or result["initial_model_state_sha256"]
                    != parent_result["refit_initial_state_sha256"]
                    or result["parent_refit_initial_state_sha256"]
                    != parent_result["refit_initial_state_sha256"]
                    or result["selected_epoch"] != parent_result["selected_epoch"]
                    or result["parent_selected_epoch"]
                    != parent_result["selected_epoch"]
                    or result["optimizer_step_count"]
                    != int(parent_result["selected_epoch"])
                    * (16 * len(cast(list[str], fold_config["outer_train_groups"])))
                ):
                    raise ValueError(f"parent compute match failed: {name}")
            ignored = {
                "completed_at_utc",
                "runtime_seconds",
                "checkpoint_sha256",
                "prediction_sha256",
            }
            if {
                key: value
                for key, value in primary_result.items()
                if key not in ignored
            } != {
                key: value for key, value in replay_result.items() if key not in ignored
            }:
                raise ValueError(f"logical replay mismatch: {name}")
            for root, result in (
                (primary_root, primary_result),
                (replay_root, replay_result),
            ):
                if (
                    sha256_file(root / "predictions.npz") != result["prediction_sha256"]
                    or sha256_file(root / "checkpoint.pt")
                    != result["checkpoint_sha256"]
                    or _state_sha256(root / "checkpoint.pt")
                    != result["final_model_state_sha256"]
                ):
                    raise ValueError(f"artifact provenance failed: {root}")
            primary = _load_npz(primary_root / "predictions.npz")
            replay = _load_npz(replay_root / "predictions.npz")
            if set(primary) != set(replay) or any(
                not np.array_equal(primary[key], replay[key]) for key in primary
            ):
                raise ValueError(f"prediction replay is not bit exact: {name}")
            expected_keys = {
                "bundle_index",
                "target",
                "candidate_mask",
                *(f"{condition}_prediction" for condition in CONDITIONS),
            }
            if set(primary) != expected_keys:
                raise ValueError(f"prediction key contract failed: {name}")
            indices = cast(npt.NDArray[np.int64], primary["bundle_index"])
            target = cast(npt.NDArray[np.float32], primary["target"])
            mask = cast(npt.NDArray[np.bool_], primary["candidate_mask"])
            if (
                indices.shape != (112,)
                or target.shape != (112, 7)
                or mask.shape != (112, 7)
            ):
                raise ValueError(f"outer-test tensor shape failed: {name}")
            if not np.array_equal(target, bundle_target[indices]) or not np.array_equal(
                mask, bundle_mask[indices]
            ):
                raise ValueError(f"bundle target/mask mismatch: {name}")
            actual_groups = {
                str(index_rows[int(index)]["recording_group"]) for index in indices
            }
            if actual_groups != set(cast(list[str], fold_config["outer_test_groups"])):
                raise ValueError(f"outer-test fold membership failed: {name}")
            parent = _load_npz(parent_root / "predictions.npz")
            if not (
                np.array_equal(indices, parent["bundle_index"])
                and np.array_equal(target, parent["target"])
                and np.array_equal(mask, parent["candidate_mask"])
            ):
                raise ValueError(f"parent OOF panel differs: {name}")
            parameter_count = int(primary_result["parameter_count"])
            if expected_parameter_count is None:
                expected_parameter_count = parameter_count
            elif parameter_count != expected_parameter_count:
                raise ValueError("treatment parameter count differs between jobs")

            predictions = {
                "parent": cast(npt.NDArray[np.float32], parent["prediction"]),
                **{
                    condition: cast(
                        npt.NDArray[np.float32], primary[f"{condition}_prediction"]
                    )
                    for condition in CONDITIONS
                },
            }
            if any(
                values.shape != (112, 7) or not np.isfinite(values).all()
                for values in predictions.values()
            ):
                raise FloatingPointError(f"nonfinite or malformed prediction: {name}")
            for row_index, bundle_index in enumerate(indices.tolist()):
                group = str(index_rows[bundle_index]["recording_group"])
                for condition, values in predictions.items():
                    if bundle_index in seen[(condition, seed)]:
                        raise ValueError("duplicate OOF prediction")
                    seen[(condition, seed)].add(bundle_index)
                    metrics = _anchor_metrics(
                        values[row_index], target[row_index], mask[row_index]
                    )
                    destination = group_metrics.setdefault(
                        (condition, seed, group), {metric: [] for metric in METRICS}
                    )
                    for metric, value in zip(METRICS, metrics, strict=True):
                        destination[metric].append(value)
                parent_choice = _choice(
                    predictions["parent"][row_index], mask[row_index]
                )
                treatment_choice = _choice(
                    predictions["correct"][row_index], mask[row_index]
                )
                utility_delta = float(target[row_index, treatment_choice]) - float(
                    target[row_index, parent_choice]
                )
                if utility_delta > TIE_TOLERANCE:
                    choice_changes["beneficial"] += 1
                elif utility_delta < -TIE_TOLERANCE:
                    choice_changes["harmful"] += 1
                else:
                    choice_changes["neutral"] += 1
                for condition in intervention_flips:
                    intervention_flips[condition] += int(
                        _choice(predictions[condition][row_index], mask[row_index])
                        != treatment_choice
                    )
            job_audit.append(
                {
                    "job": name,
                    "same_gpu_replay": True,
                    "prediction_exact": True,
                    "model_state_exact": True,
                    "parent_compute_match": True,
                    "temperature_recomputed": True,
                }
            )

    if any(len(indices) != 672 for indices in seen.values()):
        raise ValueError("OOF coverage is incomplete")
    group_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for (condition, seed, group), rows in group_metrics.items():
        if any(len(values) != 16 for values in rows.values()):
            raise ValueError("group does not contain exactly sixteen anchors")
        group_summary.setdefault(condition, {}).setdefault(str(seed), {})[group] = {
            metric: float(np.mean(values)) for metric, values in rows.items()
        }

    def seed_mean(condition: str, metric: str) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                np.mean(
                    [
                        group_summary[condition][str(seed)][group][metric]
                        for seed in SEEDS
                    ]
                )
                for group in ordered_groups
            ],
            dtype=np.float64,
        )

    statistics = cast(dict[str, Any], config["statistics"])
    bootstrap_seed = int(statistics["bootstrap_seed"])
    bootstrap_replicates = int(statistics["bootstrap_replicates"])
    condition_metrics = {
        condition: {
            metric: float(seed_mean(condition, metric).mean()) for metric in METRICS
        }
        for condition in ("parent", *CONDITIONS)
    }
    parent_difference = seed_mean("correct", "decision_regret") - seed_mean(
        "parent", "decision_regret"
    )
    parent_comparison = {
        "point": float(parent_difference.mean()),
        "ci95": _bootstrap(
            parent_difference, seed=bootstrap_seed, replicates=bootstrap_replicates
        ),
        "group_values": {
            group: float(value)
            for group, value in zip(ordered_groups, parent_difference)
        },
    }
    intervention_comparisons: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        if condition == "correct":
            continue
        difference = seed_mean(condition, "decision_regret") - seed_mean(
            "correct", "decision_regret"
        )
        intervention_comparisons[condition] = {
            "point": float(difference.mean()),
            "ci95": _bootstrap(
                difference, seed=bootstrap_seed, replicates=bootstrap_replicates
            ),
            "group_values": {
                group: float(value) for group, value in zip(ordered_groups, difference)
            },
        }
    thresholds = cast(dict[str, Any], config["thresholds"])
    oracle_gain = float(thresholds["contextual_oracle_gain_denominator"])
    recovery = -float(parent_difference.mean()) / oracle_gain
    parent_ci = cast(list[float], parent_comparison["ci95"])
    objective_supported = bool(parent_ci[1] < 0.0)
    minimum_effect_supported = bool(
        recovery >= float(thresholds["minimum_oracle_opportunity_recovery"])
    )
    binding_by_condition = {
        condition: bool(cast(list[float], comparison["ci95"])[0] > 0.0)
        for condition, comparison in intervention_comparisons.items()
    }
    binding_supported = all(binding_by_condition.values())
    decision = _classify(
        objective_supported=objective_supported,
        minimum_effect_supported=minimum_effect_supported,
        binding_supported=binding_supported,
    )
    all_values = [
        value for values in condition_metrics.values() for value in values.values()
    ]
    if not all(math.isfinite(value) for value in all_values) or not math.isfinite(
        recovery
    ):
        raise FloatingPointError("aggregate result contains a nonfinite value")
    return {
        "schema": "aorta.r005e3a.run_audit.v1",
        "status": "PASS",
        "support": (
            "SUPPORTED"
            if decision.startswith("GO_")
            else "MIXED" if decision.startswith("MIXED_") else "UNSUPPORTED"
        ),
        "decision": decision,
        "mechanical_gates": {
            "job_count_per_phase": 18,
            "primary_replay_predictions_bit_exact": True,
            "primary_replay_model_states_exact": True,
            "same_physical_gpu_per_job": True,
            "parent_initialization_epoch_and_step_budget_exact": True,
            "temperature_recomputed_from_outer_train_only": True,
            "fold_target_mask_and_OOF_coverage_exact": True,
            "finite_values": True,
            "source_only": True,
            "sealed_roles_accessed": False,
        },
        "scientific_gates": {
            "treatment_minus_parent_regret_ci_upper_below_zero": objective_supported,
            "minimum_oracle_opportunity_recovery": minimum_effect_supported,
            "all_scene_binding_interventions_harm_regret": binding_supported,
            "scene_binding_by_condition": binding_by_condition,
        },
        "condition_metrics": condition_metrics,
        "treatment_minus_parent_decision_regret": parent_comparison,
        "oracle_opportunity_recovery": recovery,
        "intervention_minus_correct_decision_regret": intervention_comparisons,
        "choice_changes_vs_parent": choice_changes,
        "intervention_top_choice_flips": intervention_flips,
        "parameter_count": expected_parameter_count,
        "group_metrics": group_summary,
        "job_audit": job_audit,
        "claim_boundary": (
            "Source-fit OOF loss-only objective qualification. It is not a proper "
            "distributional result, Differential Query, appearance robustness, NavDream, "
            "policy-level driving gain, or safety evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--mismatch-manifest", type=Path, required=True)
    parser.add_argument("--sensitivity-manifest", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = audit(arguments)
    _write_json(arguments.output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "group_metrics"},
            sort_keys=True,
        )
    )
