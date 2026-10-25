"""Audit exact replay and independently compute the frozen R005E2 decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file  # type: ignore[import-untyped]


ARMS = ("V0_removed", "V1_mismatch", "V2_correct")
SEEDS = (20261025, 20261026, 20261027)
TIE_TOLERANCE = 1.0e-12
ORACLE_GAIN = 0.01665036604823182


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _state_sha256(checkpoint: Path) -> str:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tensors = cast(dict[str, torch.Tensor], state["model"])
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _job_name(fold: int, arm: str, seed: int) -> str:
    return f"fold{fold}_{arm}_seed{seed}"


def _load_npz(path: Path) -> dict[str, npt.NDArray[Any]]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _anchor_metrics(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
) -> tuple[float, float, float, float]:
    challenger = mask.copy()
    challenger[0] = False
    relation_mse = (
        float(np.square(prediction - target)[challenger].mean())
        if bool(challenger.any())
        else 0.0
    )
    valid_indices = np.flatnonzero(mask)
    chosen = int(valid_indices[np.argmax(prediction[valid_indices])])
    oracle = float(np.max(target[valid_indices]))
    regret = oracle - float(target[chosen])
    evaluated = challenger & (np.abs(target) > TIE_TOLERANCE)
    sign_denominator = int(np.sum(evaluated))
    sign_error = (
        float(
            np.mean(
                np.where(
                    prediction[evaluated] > TIE_TOLERANCE,
                    1,
                    np.where(prediction[evaluated] < -TIE_TOLERANCE, -1, 0),
                )
                != np.sign(target[evaluated])
            )
        )
        if sign_denominator
        else 0.0
    )
    boundary_error = float(regret > TIE_TOLERANCE)
    return relation_mse, regret, sign_error, boundary_error


def _bootstrap(values: npt.NDArray[np.float64]) -> list[float]:
    if values.shape != (42,):
        raise ValueError("bootstrap requires exactly 42 group differences")
    generator = np.random.default_rng(20261025)
    draws = generator.integers(0, 42, size=(10000, 42))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    progress = cast(
        dict[str, Any], json.loads((args.run_root / "progress.json").read_text("utf-8"))
    )
    if progress.get("status") != "PRIMARY_REPLAY_COMPLETED":
        raise ValueError("primary/replay chain is not complete")
    bundle_manifest = cast(
        dict[str, Any],
        json.loads((args.bundle_root / "manifest.json").read_text("utf-8")),
    )
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text("utf-8"))
    )
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text("utf-8")))
    mismatch_hash = sha256_file(args.mismatch_manifest)
    if sha256_file(args.crossfit_manifest) != str(
        config["cross_fitting"]["manifest_sha256"]
    ):
        raise ValueError("audited cross-fit manifest differs from preregistration")
    if mismatch_hash != str(config["preflight"]["mismatch_donor_manifest"]["sha256"]):
        raise ValueError("audited mismatch manifest differs from preregistration")
    bundle_inputs = cast(dict[str, Any], bundle_manifest["inputs"])
    if (
        bundle_inputs["target_result_sha256"]
        != config["frozen_target"]["result_sha256"]
        or bundle_inputs["crossfit_manifest_sha256"]
        != config["cross_fitting"]["manifest_sha256"]
        or bundle_inputs["panel_roster_sha256"]
        != config["preflight"]["panel_identity_roster"]["sha256"]
    ):
        raise ValueError("scientific bundle provenance differs from preregistration")
    expected_group_fold = {
        str(group): int(fold)
        for group, fold in cast(dict[str, Any], crossfit["group_outer_fold"]).items()
    }
    index_rows = {
        int(row["bundle_index"]): row
        for row in cast(list[dict[str, Any]], bundle_manifest["anchors"])
    }
    bundle_target = np.load(args.bundle_root / "target_margin.npy", mmap_mode="r")
    bundle_mask = np.load(args.bundle_root / "candidate_mask.npy", mmap_mode="r")
    group_metrics: dict[tuple[str, int, str], dict[str, list[float]]] = {}
    initialization_by_fold_seed: dict[tuple[int, int], str] = {}
    oof_seen: dict[tuple[str, int], set[int]] = {
        (arm, seed): set() for arm in ARMS for seed in SEEDS
    }
    job_audit: list[dict[str, Any]] = []
    for fold in range(6):
        for arm in ARMS:
            for seed in SEEDS:
                name = _job_name(fold, arm, seed)
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
                for result in (primary_result, replay_result):
                    if (
                        result.get("status") != "COMPLETED"
                        or result.get("outer_fold") != fold
                        or result.get("arm") != arm
                        or result.get("seed") != seed
                        or result.get(
                            "development_calibration_test_navdream_target_accessed"
                        )
                        is not False
                        or result.get("scientific_metrics_emitted") is not False
                    ):
                        raise ValueError(f"job result contract failed: {name}")
                    input_hashes = cast(dict[str, Any], result["input_sha256"])
                    if (
                        input_hashes["bundle_manifest"]
                        != sha256_file(args.bundle_root / "manifest.json")
                        or input_hashes["crossfit_manifest"]
                        != sha256_file(args.crossfit_manifest)
                        or input_hashes["mismatch_manifest"] != mismatch_hash
                    ):
                        raise ValueError(f"job input provenance mismatch: {name}")
                assignment = cast(
                    dict[str, Any],
                    json.loads(
                        (args.run_root / "gpu_assignment.json").read_text("utf-8")
                    ),
                )
                expected_gpu = int(assignment[name])
                for result in (primary_result, replay_result):
                    physical = cast(dict[str, Any], result["physical_gpu"])
                    if int(physical["physical_index"]) != expected_gpu:
                        raise ValueError(
                            f"job physical GPU assignment mismatch: {name}"
                        )
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
                    key: value
                    for key, value in replay_result.items()
                    if key not in ignored
                }:
                    raise ValueError(f"job logical replay mismatch: {name}")
                initialization_key = (fold, seed)
                initialization_hash = str(
                    primary_result["selection_initial_state_sha256"]
                )
                registered_initialization = initialization_by_fold_seed.setdefault(
                    initialization_key, initialization_hash
                )
                if registered_initialization != initialization_hash:
                    raise ValueError(f"arm initialization mismatch: {name}")
                for root, result in (
                    (primary_root, primary_result),
                    (replay_root, replay_result),
                ):
                    if (
                        _state_sha256(root / "checkpoint.pt")
                        != result["final_model_state_sha256"]
                    ):
                        raise ValueError(f"checkpoint state hash mismatch: {name}")
                    if (
                        sha256_file(root / "predictions.npz")
                        != result["prediction_sha256"]
                    ):
                        raise ValueError(f"prediction artifact hash mismatch: {name}")
                primary = _load_npz(primary_root / "predictions.npz")
                replay = _load_npz(replay_root / "predictions.npz")
                if set(primary) != set(replay) or any(
                    not np.array_equal(primary[key], replay[key]) for key in primary
                ):
                    raise ValueError(f"prediction replay is not bit exact: {name}")
                required = {"bundle_index", "prediction", "target", "candidate_mask"}
                intervention = {
                    "mismatch_intervention_prediction",
                    "removed_intervention_prediction",
                }
                expected_keys = required | (
                    intervention if arm == "V2_correct" else set()
                )
                if set(primary) != expected_keys:
                    raise ValueError(f"prediction key contract failed: {name}")
                indices = cast(npt.NDArray[np.int64], primary["bundle_index"])
                prediction = cast(npt.NDArray[np.float32], primary["prediction"])
                target = cast(npt.NDArray[np.float32], primary["target"])
                mask = cast(npt.NDArray[np.bool_], primary["candidate_mask"])
                if indices.shape != (112,) or prediction.shape != (112, 7):
                    raise ValueError(f"outer-test shape mismatch: {name}")
                if not np.array_equal(
                    target, bundle_target[indices]
                ) or not np.array_equal(mask, bundle_mask[indices]):
                    raise ValueError(
                        f"job targets/membership differ from bundle: {name}"
                    )
                if not np.isfinite(prediction).all():
                    raise ValueError(f"job prediction is nonfinite: {name}")
                job_groups = {
                    str(index_rows[int(index)]["recording_group"]) for index in indices
                }
                if len(job_groups) != 7 or any(
                    expected_group_fold[group] != fold for group in job_groups
                ):
                    raise ValueError(f"outer-test fold isolation failed: {name}")
                if oof_seen[(arm, seed)].intersection(indices.tolist()):
                    raise ValueError(f"duplicate OOF prediction: {name}")
                oof_seen[(arm, seed)].update(indices.tolist())
                prediction_sets = {arm: prediction}
                if arm == "V2_correct":
                    prediction_sets["V2_mismatch_intervention"] = cast(
                        npt.NDArray[np.float32],
                        primary["mismatch_intervention_prediction"],
                    )
                    prediction_sets["V2_removed_intervention"] = cast(
                        npt.NDArray[np.float32],
                        primary["removed_intervention_prediction"],
                    )
                for condition, condition_prediction in prediction_sets.items():
                    if (
                        condition_prediction.shape != (112, 7)
                        or not np.isfinite(condition_prediction).all()
                    ):
                        raise ValueError(
                            f"intervention prediction contract failed: {name}"
                        )
                    for row_index, bundle_index in enumerate(indices.tolist()):
                        group = str(index_rows[bundle_index]["recording_group"])
                        anchor_metric_values = _anchor_metrics(
                            condition_prediction[row_index],
                            target[row_index],
                            mask[row_index],
                        )
                        destination = group_metrics.setdefault(
                            (condition, seed, group),
                            {
                                "relation_mse": [],
                                "decision_regret": [],
                                "sign_error": [],
                                "boundary_error": [],
                            },
                        )
                        for key, value in zip(
                            destination, anchor_metric_values, strict=True
                        ):
                            destination[key].append(value)
                job_audit.append(
                    {
                        "name": name,
                        "selected_epoch": int(primary_result["selected_epoch"]),
                        "prediction_arrays_exact": True,
                        "checkpoint_state_exact": True,
                    }
                )
    if any(len(indices) != 672 for indices in oof_seen.values()):
        raise ValueError("OOF coverage is incomplete")
    group_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for (condition, seed, group), metric_rows in group_metrics.items():
        if any(len(rows) != 16 for rows in metric_rows.values()):
            raise ValueError("group metric must contain exactly sixteen anchors")
        group_summary.setdefault(condition, {}).setdefault(str(seed), {})[group] = {
            key: float(np.mean(rows)) for key, rows in metric_rows.items()
        }
    expected_conditions = {
        *ARMS,
        "V2_mismatch_intervention",
        "V2_removed_intervention",
    }
    if set(group_summary) != expected_conditions:
        raise ValueError("aggregate condition roster mismatch")
    ordered_groups = sorted(expected_group_fold)

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

    comparisons: dict[str, dict[str, Any]] = {}
    comparison_specs = (
        ("V2_minus_V0_relation_mse", "V2_correct", "V0_removed", "relation_mse"),
        ("V2_minus_V1_relation_mse", "V2_correct", "V1_mismatch", "relation_mse"),
        ("V2_minus_V0_regret", "V2_correct", "V0_removed", "decision_regret"),
        ("V2_minus_V1_regret", "V2_correct", "V1_mismatch", "decision_regret"),
        (
            "V2_mismatch_intervention_minus_correct_regret",
            "V2_mismatch_intervention",
            "V2_correct",
            "decision_regret",
        ),
        (
            "V2_removed_intervention_minus_correct_regret",
            "V2_removed_intervention",
            "V2_correct",
            "decision_regret",
        ),
    )
    for name, left, right, metric_name in comparison_specs:
        difference = seed_mean(left, metric_name) - seed_mean(right, metric_name)
        comparisons[name] = {
            "point": float(difference.mean()),
            "ci95": _bootstrap(difference),
            "group_values": {
                group: float(value) for group, value in zip(ordered_groups, difference)
            },
        }
    arm_metrics = {
        condition: {
            metric: float(seed_mean(condition, metric).mean())
            for metric in (
                "relation_mse",
                "decision_regret",
                "sign_error",
                "boundary_error",
            )
        }
        for condition in expected_conditions
    }
    recovery = (
        arm_metrics["V0_removed"]["decision_regret"]
        - arm_metrics["V2_correct"]["decision_regret"]
    ) / ORACLE_GAIN
    proper_pass = (
        comparisons["V2_minus_V0_relation_mse"]["ci95"][1] < 0.0
        and comparisons["V2_minus_V1_relation_mse"]["ci95"][1] < 0.0
    )
    regret_pass = (
        comparisons["V2_minus_V0_regret"]["ci95"][1] < 0.0
        and comparisons["V2_minus_V1_regret"]["ci95"][1] < 0.0
    )
    intervention_pass = (
        comparisons["V2_mismatch_intervention_minus_correct_regret"]["ci95"][0] > 0.0
        and comparisons["V2_removed_intervention_minus_correct_regret"]["ci95"][0] > 0.0
    )
    effect_pass = recovery >= 0.10
    if proper_pass and regret_pass and intervention_pass and effect_pass:
        support, decision = "SUPPORTED", "GO_VISUAL_INFORMATION_RECOVERABLE"
    elif proper_pass:
        support, decision = "MIXED", "HOLD_VISUAL_SIGNAL_NOT_DECISION_SUFFICIENT"
    else:
        support, decision = "UNSUPPORTED", "STOP_CURRENT_VISUAL_MAIN_CLAIM"
    if not math.isfinite(recovery):
        raise FloatingPointError("oracle-normalized recovery is nonfinite")
    return {
        "schema": "aorta.r005e2.scientific_run_audit.v1",
        "status": "PASS",
        "support": support,
        "decision": decision,
        "mechanical_gates": {
            "job_count_per_phase": 54,
            "primary_replay_prediction_bit_exact": True,
            "primary_replay_model_state_exact": True,
            "same_fold_seed_initialization_across_arms": True,
            "same_physical_gpu_per_job_primary_replay": True,
            "oof_coverage_exact": True,
            "fold_isolation_exact": True,
            "target_membership_exact": True,
            "finite_values": True,
            "source_only": True,
            "preregistered_input_hashes_exact": True,
            "development_calibration_test_navdream_target_accessed": False,
        },
        "metric_contract": {
            "independent_unit": "recording_group",
            "anchors_per_group": 16,
            "seed_aggregation": "mean_within_group_before_group_inference",
            "tie_tolerance": TIE_TOLERANCE,
            "bootstrap_seed": 20261025,
            "bootstrap_replicates": 10000,
        },
        "arm_metrics": arm_metrics,
        "comparisons": comparisons,
        "oracle_normalized_visual_recovery": float(recovery),
        "gate_components": {
            "proper_risk": proper_pass,
            "decision_regret": regret_pass,
            "scene_specific_interventions": intervention_pass,
            "minimum_effect": effect_pass,
        },
        "group_metrics": group_summary,
        "job_audit": job_audit,
        "claim_boundary": (
            "Source-fit OOF recovery of native-history U_ctx only; not Differential Query "
            "superiority, appearance robustness, pseudo-closed-loop value, driving gain, or safety."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--mismatch-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = audit(args)
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "support": result["support"],
                "decision": result["decision"],
            }
        )
    )
