"""Audit R005E2S replay and compute the frozen secondary sensitivity verdict."""

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
from audit_r005e2_scientific_run import (
    SEEDS,
    _anchor_metrics,
    _bootstrap,
    _load_npz,
    _state_sha256,
)


SCHEMES = ("S2_disjoint_geometry", "S3_hash_random")
PARENT_DECISION = "STOP_CURRENT_VISUAL_MAIN_CLAIM"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _classify(
    relation_support: dict[str, bool], decision_support: dict[str, bool]
) -> str:
    if all(decision_support.values()):
        return "CONSISTENT_DECISION_BINDING_SECONDARY"
    if all(relation_support.values()):
        return "RELATION_ONLY_BINDING_SECONDARY"
    scheme_support = {
        scheme: relation_support[scheme] or decision_support[scheme]
        for scheme in SCHEMES
    }
    if any(scheme_support.values()):
        return "DONOR_SENSITIVE_OR_INCONSISTENT_SECONDARY"
    return "NO_CONSISTENT_SCENE_BINDING_SECONDARY"


def _top_choice(
    prediction: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]
) -> int:
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        raise ValueError("candidate mask contains no legal candidate")
    return int(valid[np.argmax(prediction[valid])])


def _fold_row(manifest: dict[str, Any], fold: int) -> dict[str, Any]:
    rows = [
        row
        for row in cast(list[dict[str, Any]], manifest["folds"])
        if int(row["outer_fold"]) == fold
    ]
    if len(rows) != 1:
        raise ValueError("sensitivity fold is absent or duplicated")
    return rows[0]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    progress = cast(
        dict[str, Any], json.loads((args.run_root / "progress.json").read_text("utf-8"))
    )
    if progress.get("status") != "PRIMARY_REPLAY_COMPLETED":
        raise ValueError("primary/replay chain is incomplete")
    if (
        progress.get("development_calibration_test_navdream_target_accessed") is not False
        or progress.get("scientific_metrics_emitted") is not False
    ):
        raise ValueError("sealed-role or partial-metric progress contract failed")
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text("utf-8")))
    contract = cast(
        dict[str, Any], json.loads((args.run_root / "run_contract.json").read_text("utf-8"))
    )
    if contract.get("phase_order") != ["primary", "replay"]:
        raise ValueError("run phase contract differs from preregistration")
    expected_inputs = {
        "config": sha256_file(args.config),
        "runner": sha256_file(args.runner),
        "bundle_manifest": sha256_file(args.bundle_root / "manifest.json"),
        "crossfit_manifest": sha256_file(args.crossfit_manifest),
        "sensitivity_manifest": sha256_file(args.sensitivity_manifest),
        "parent_audit": sha256_file(args.parent_run_root / "audit.json"),
    }
    if contract.get("input_sha256") != expected_inputs:
        raise ValueError("run input provenance differs from current files")
    if expected_inputs["sensitivity_manifest"] != str(
        config["inputs"]["donor_manifest_sha256"]
    ):
        raise ValueError("sensitivity manifest differs from preregistration")
    parent_audit = cast(
        dict[str, Any], json.loads((args.parent_run_root / "audit.json").read_text("utf-8"))
    )
    if (
        parent_audit.get("status") != "PASS"
        or parent_audit.get("support") != "UNSUPPORTED"
        or parent_audit.get("decision") != PARENT_DECISION
    ):
        raise ValueError("parent audit is not the frozen STOP result")

    bundle = cast(
        dict[str, Any], json.loads((args.bundle_root / "manifest.json").read_text("utf-8"))
    )
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text("utf-8"))
    )
    sensitivity = cast(
        dict[str, Any], json.loads(args.sensitivity_manifest.read_text("utf-8"))
    )
    index_rows = {
        int(row["bundle_index"]): row
        for row in cast(list[dict[str, Any]], bundle["anchors"])
    }
    identity_rows = {
        str(row["identity"]): row
        for row in cast(list[dict[str, Any]], bundle["anchors"])
    }
    ordered_groups = sorted(cast(dict[str, Any], crossfit["group_outer_fold"]))
    if len(ordered_groups) != 42:
        raise ValueError("cross-fit group roster must contain 42 groups")
    assignment = cast(
        dict[str, Any], json.loads((args.run_root / "gpu_assignment.json").read_text("utf-8"))
    )
    if set(assignment) != {f"fold{fold}" for fold in range(6)}:
        raise ValueError("GPU assignment roster is incomplete")

    group_metrics: dict[tuple[str, int, str], dict[str, list[float]]] = {}
    flip_rows: dict[str, list[bool]] = {scheme: [] for scheme in SCHEMES}
    flip_intersection = 0
    flip_union = 0
    distance_rows: dict[str, list[float]] = {scheme: [] for scheme in SCHEMES}
    dino_rows: dict[str, list[float]] = {scheme: [] for scheme in SCHEMES}
    seen: dict[tuple[str, int], set[int]] = {
        (condition, seed): set()
        for condition in ("V2_correct", *SCHEMES)
        for seed in SEEDS
    }
    job_audit: list[dict[str, Any]] = []
    for fold in range(6):
        primary_root = args.run_root / "primary" / f"fold{fold}"
        replay_root = args.run_root / "replay" / f"fold{fold}"
        primary_result = cast(
            dict[str, Any], json.loads((primary_root / "result.json").read_text("utf-8"))
        )
        replay_result = cast(
            dict[str, Any], json.loads((replay_root / "result.json").read_text("utf-8"))
        )
        for result in (primary_result, replay_result):
            if (
                result.get("status") != "COMPLETED"
                or result.get("outer_fold") != fold
                or result.get("parent_decision") != PARENT_DECISION
                or result.get("parent_decision_is_immutable") is not True
                or result.get("development_calibration_test_navdream_target_accessed") is not False
                or result.get("scientific_metrics_emitted") is not False
                or result.get("code_commit") != contract.get("code_commit")
                or result.get("input_sha256") != {
                    "bundle_manifest": expected_inputs["bundle_manifest"],
                    "crossfit_manifest": expected_inputs["crossfit_manifest"],
                    "sensitivity_manifest": expected_inputs["sensitivity_manifest"],
                    "parent_audit": expected_inputs["parent_audit"],
                }
            ):
                raise ValueError(f"fold job contract failed: fold{fold}")
            physical = cast(dict[str, Any], result["physical_gpu"])
            if int(physical["physical_index"]) != int(assignment[f"fold{fold}"]):
                raise ValueError(f"same-GPU replay contract failed: fold{fold}")
        ignored = {"completed_at_utc", "runtime_seconds"}
        if {
            key: value for key, value in primary_result.items() if key not in ignored
        } != {key: value for key, value in replay_result.items() if key not in ignored}:
            raise ValueError(f"logical replay mismatch: fold{fold}")
        for root, result in ((primary_root, primary_result), (replay_root, replay_result)):
            if sha256_file(root / "predictions.npz") != result["prediction_sha256"]:
                raise ValueError(f"prediction hash mismatch: {root}")
        primary = _load_npz(primary_root / "predictions.npz")
        replay = _load_npz(replay_root / "predictions.npz")
        if set(primary) != set(replay) or any(
            not np.array_equal(primary[key], replay[key]) for key in primary
        ):
            raise ValueError(f"prediction replay is not bit exact: fold{fold}")
        expected_keys = {
            "seeds",
            "bundle_index",
            "target",
            "candidate_mask",
            "S2_disjoint_geometry_prediction",
            "S3_hash_random_prediction",
            "S2_disjoint_geometry_dino_rms",
            "S3_hash_random_dino_rms",
        }
        if set(primary) != expected_keys:
            raise ValueError(f"prediction key contract failed: fold{fold}")
        indices = cast(npt.NDArray[np.int64], primary["bundle_index"])
        target = cast(npt.NDArray[np.float32], primary["target"])
        mask = cast(npt.NDArray[np.bool_], primary["candidate_mask"])
        if indices.shape != (112,) or target.shape != (112, 7) or mask.shape != (112, 7):
            raise ValueError(f"fold tensor shape mismatch: fold{fold}")
        fold_config = next(
            row
            for row in cast(list[dict[str, Any]], crossfit["outer_folds"])
            if int(row["outer_fold"]) == fold
        )
        fold_groups = set(cast(list[str], fold_config["outer_test_groups"]))
        actual_groups = {str(index_rows[int(index)]["recording_group"]) for index in indices}
        if actual_groups != fold_groups:
            raise ValueError(f"outer-fold membership mismatch: fold{fold}")

        donor_fold = _fold_row(sensitivity, fold)
        for scheme in SCHEMES:
            scheme_stage = cast(dict[str, Any], donor_fold["schemes"])[scheme]["outer_test"]
            donor_rows = cast(list[dict[str, Any]], scheme_stage["rows"])
            donor_by_recipient = {str(row["recipient"]): str(row["donor"]) for row in donor_rows}
            if len(donor_rows) != 112 or len(donor_by_recipient) != 112:
                raise ValueError(f"donor roster is not one-to-one: fold{fold} {scheme}")
            if len(set(donor_by_recipient.values())) != 112:
                raise ValueError(f"donor reuse detected: fold{fold} {scheme}")
            for row in donor_rows:
                recipient = str(row["recipient"])
                donor = str(row["donor"])
                if recipient not in identity_rows or donor not in identity_rows:
                    raise ValueError("donor mapping contains an unknown identity")
                recipient_group = str(identity_rows[recipient]["recording_group"])
                donor_group = str(identity_rows[donor]["recording_group"])
                if recipient_group not in fold_groups or donor_group in fold_groups or donor_group == recipient_group:
                    raise ValueError(f"donor fold/group isolation failed: fold{fold} {scheme}")
                distance_rows[scheme].append(float(row["standardized_squared_distance"]))
            prediction = cast(npt.NDArray[np.float32], primary[f"{scheme}_prediction"])
            dino = cast(npt.NDArray[np.float32], primary[f"{scheme}_dino_rms"])
            if prediction.shape != (3, 112, 7) or dino.shape != (112,):
                raise ValueError(f"sensitivity output shape mismatch: fold{fold} {scheme}")
            if not np.isfinite(prediction).all() or not np.isfinite(dino).all():
                raise ValueError(f"nonfinite sensitivity output: fold{fold} {scheme}")
            dino_rows[scheme].extend(float(value) for value in dino)

        correct_by_seed: dict[int, npt.NDArray[np.float32]] = {}
        for seed_index, seed in enumerate(SEEDS):
            parent_root = args.parent_run_root / "primary" / f"fold{fold}_V2_correct_seed{seed}"
            parent_result = cast(
                dict[str, Any], json.loads((parent_root / "result.json").read_text("utf-8"))
            )
            if (
                sha256_file(parent_root / "checkpoint.pt") != parent_result["checkpoint_sha256"]
                or _state_sha256(parent_root / "checkpoint.pt") != parent_result["final_model_state_sha256"]
            ):
                raise ValueError(f"parent checkpoint provenance failed: fold{fold} seed{seed}")
            ledger = cast(list[dict[str, Any]], primary_result["checkpoint_ledger"])[seed_index]
            if (
                int(ledger["seed"]) != seed
                or ledger["checkpoint_sha256"] != parent_result["checkpoint_sha256"]
                or ledger["model_state_sha256"] != parent_result["final_model_state_sha256"]
            ):
                raise ValueError(f"checkpoint ledger mismatch: fold{fold} seed{seed}")
            parent = _load_npz(parent_root / "predictions.npz")
            if not (
                np.array_equal(indices, parent["bundle_index"])
                and np.array_equal(target, parent["target"])
                and np.array_equal(mask, parent["candidate_mask"])
            ):
                raise ValueError(f"parent/sensitivity membership mismatch: fold{fold} seed{seed}")
            correct = cast(npt.NDArray[np.float32], parent["prediction"])
            correct_by_seed[seed] = correct
            conditions = {
                "V2_correct": correct,
                SCHEMES[0]: cast(npt.NDArray[np.float32], primary[f"{SCHEMES[0]}_prediction"])[seed_index],
                SCHEMES[1]: cast(npt.NDArray[np.float32], primary[f"{SCHEMES[1]}_prediction"])[seed_index],
            }
            for row_index, bundle_index in enumerate(indices.tolist()):
                group = str(index_rows[bundle_index]["recording_group"])
                for condition, values in conditions.items():
                    if bundle_index in seen[(condition, seed)]:
                        raise ValueError("duplicate OOF sensitivity prediction")
                    seen[(condition, seed)].add(bundle_index)
                    metric_values = _anchor_metrics(values[row_index], target[row_index], mask[row_index])
                    destination = group_metrics.setdefault(
                        (condition, seed, group),
                        {"relation_mse": [], "decision_regret": [], "sign_error": [], "boundary_error": []},
                    )
                    for key, value in zip(destination, metric_values, strict=True):
                        destination[key].append(value)
                s2_flip = _top_choice(conditions[SCHEMES[0]][row_index], mask[row_index]) != _top_choice(correct[row_index], mask[row_index])
                s3_flip = _top_choice(conditions[SCHEMES[1]][row_index], mask[row_index]) != _top_choice(correct[row_index], mask[row_index])
                flip_rows[SCHEMES[0]].append(s2_flip)
                flip_rows[SCHEMES[1]].append(s3_flip)
                flip_intersection += int(s2_flip and s3_flip)
                flip_union += int(s2_flip or s3_flip)
        job_audit.append({"fold": fold, "prediction_arrays_exact": True, "same_gpu_replay": True, "parent_checkpoint_states_verified": True})

    if any(len(values) != 672 for values in seen.values()):
        raise ValueError("OOF sensitivity coverage is incomplete")
    group_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for (condition, seed, group), metric_rows in group_metrics.items():
        if any(len(rows) != 16 for rows in metric_rows.values()):
            raise ValueError("group metric does not contain exactly 16 anchors")
        group_summary.setdefault(condition, {}).setdefault(str(seed), {})[group] = {
            key: float(np.mean(values)) for key, values in metric_rows.items()
        }

    def seed_mean(condition: str, metric: str) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                np.mean([group_summary[condition][str(seed)][group][metric] for seed in SEEDS])
                for group in ordered_groups
            ],
            dtype=np.float64,
        )

    condition_metrics = {
        condition: {
            metric: float(seed_mean(condition, metric).mean())
            for metric in ("relation_mse", "decision_regret", "sign_error", "boundary_error")
        }
        for condition in ("V2_correct", *SCHEMES)
    }
    comparisons: dict[str, dict[str, Any]] = {}
    relation_support: dict[str, bool] = {}
    decision_support: dict[str, bool] = {}
    for scheme in SCHEMES:
        for metric in ("relation_mse", "decision_regret"):
            difference = seed_mean(scheme, metric) - seed_mean("V2_correct", metric)
            comparison = {
                "point": float(difference.mean()),
                "ci95": _bootstrap(difference),
                "positive_group_fraction": float(np.mean(difference > 0.0)),
                "worst_group": ordered_groups[int(np.argmin(difference))],
                "worst_group_value": float(np.min(difference)),
                "group_values": {group: float(value) for group, value in zip(ordered_groups, difference)},
            }
            comparisons[f"{scheme}_minus_correct_{metric}"] = comparison
            supported = bool(comparison["ci95"][0] > 0.0)
            if metric == "relation_mse":
                relation_support[scheme] = supported
            else:
                decision_support[scheme] = supported
    classification = _classify(relation_support, decision_support)
    distance_summary = {
        scheme: {
            "geometry_standardized_squared_distance_mean": float(np.mean(distance_rows[scheme])),
            "geometry_standardized_squared_distance_p95": float(np.quantile(distance_rows[scheme], 0.95)),
            "geometry_standardized_squared_distance_max": float(np.max(distance_rows[scheme])),
            "dino_rms_mean": float(np.mean(dino_rows[scheme])),
            "dino_rms_p95": float(np.quantile(dino_rows[scheme], 0.95)),
            "dino_rms_max": float(np.max(dino_rows[scheme])),
        }
        for scheme in SCHEMES
    }
    flip_summary = {
        scheme: {
            "count": int(np.sum(flip_rows[scheme])),
            "fraction": float(np.mean(flip_rows[scheme])),
        }
        for scheme in SCHEMES
    }
    flip_summary["intersection"] = {"count": flip_intersection, "fraction": flip_intersection / (672 * len(SEEDS))}
    flip_summary["union"] = {"count": flip_union, "fraction": flip_union / (672 * len(SEEDS))}
    all_scalars = [
        value
        for condition in condition_metrics.values()
        for value in condition.values()
    ]
    if not all(math.isfinite(value) for value in all_scalars):
        raise FloatingPointError("aggregate metric is nonfinite")
    return {
        "schema": "aorta.r005e2s.run_audit.v1",
        "status": "PASS",
        "classification": classification,
        "parent_decision": PARENT_DECISION,
        "parent_decision_is_immutable": True,
        "mechanical_gates": {
            "job_count_per_phase": 6,
            "primary_replay_prediction_bit_exact": True,
            "same_physical_gpu_per_fold_primary_replay": True,
            "parent_checkpoint_file_and_state_hashes_exact": True,
            "oof_coverage_exact": True,
            "fold_and_donor_isolation_exact": True,
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
            "bootstrap_seed": 20261025,
            "bootstrap_replicates": 10000,
        },
        "condition_metrics": condition_metrics,
        "comparisons": comparisons,
        "relation_support": relation_support,
        "decision_support": decision_support,
        "donor_distance_ledger": distance_summary,
        "top_choice_flip": flip_summary,
        "group_metrics": group_summary,
        "job_audit": job_audit,
        "claim_boundary": (
            "Secondary source-fit OOF sensitivity to two preconstructed donor rules only; "
            "cannot change the parent STOP and is not conditional mutual information, "
            "Differential Query, appearance robustness, NavDream, driving gain, or safety."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--sensitivity-manifest", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = audit(parsed)
    _write_json(parsed.output, result)
    print(json.dumps({"status": result["status"], "classification": result["classification"], "parent_decision": result["parent_decision"]}))
