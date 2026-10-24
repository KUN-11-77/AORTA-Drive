"""Group-level inference helpers for R004H6."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from aorta_drive.evaluation.appearance_mechanism import mechanism_condition_supported

CONFIRMATORY_CONDITIONS = ("clean", "low_exposure", "low_contrast", "cool_cast")


def validate_serialized_teacher_projection(
    serialized_teacher: np.ndarray[Any, Any],
    exact_teacher: np.ndarray[Any, Any],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Validate a float32 field artifact against its authoritative float64 teacher.

    R004H6 computed metrics from the source float64 teacher and stored a float32 copy
    inside each portable field artifact.  A hard salient-edge threshold can therefore
    select a different edge after serialization.  Audits must first prove that the
    artifact is the exact float32 projection, then replay metrics with the authoritative
    float64 array used by the runner.
    """

    serialized = np.asarray(serialized_teacher)
    exact = np.asarray(exact_teacher)
    if serialized.dtype != np.float32 or exact.dtype != np.float64:
        raise ValueError("teacher replay requires float32 artifact and float64 source")
    if serialized.shape != exact.shape or serialized.ndim != 2:
        raise ValueError("serialized and exact teacher shapes disagree")
    if not np.all(np.isfinite(serialized)) or not np.all(np.isfinite(exact)):
        raise ValueError("teacher replay arrays must be finite")
    if not np.array_equal(serialized, exact.astype(np.float32)):
        raise ValueError("serialized teacher is not the exact float32 source projection")
    return np.ascontiguousarray(exact)


def deterministic_group_derangement(groups: Iterable[str]) -> dict[str, str]:
    """Map each group to a different donor using a result-independent order."""

    unique = tuple(set(groups))
    if len(unique) < 2:
        raise ValueError("at least two unique groups are required")
    ordered = tuple(
        sorted(unique, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))
    )
    return {group: ordered[(index + 1) % len(ordered)] for index, group in enumerate(ordered)}


def _bootstrap_ci(
    values: np.ndarray[Any, np.dtype[np.float64]], *, seed: int, replicates: int
) -> tuple[float, float]:
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be a finite vector with at least two groups")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = np.mean(values[draws], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_heldout_ledgers(
    ledgers: Sequence[Mapping[str, Any]],
    *,
    all_conditions: Sequence[str],
    seed: int = 20261024,
    replicates: int = 10_000,
    expected_groups: int = 24,
    anchors_per_group: int = 16,
) -> dict[str, Any]:
    """Aggregate anchors within groups and apply the preregistered H6 gates."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ledgers:
        grouped[str(row["recording_group"])].append(row)
    if len(grouped) != expected_groups or any(
        len(rows) != anchors_per_group for rows in grouped.values()
    ):
        raise ValueError("incomplete R004H6 recording-group panel")
    if tuple(all_conditions[:4]) != CONFIRMATORY_CONDITIONS:
        raise ValueError("confirmatory condition order changed")

    condition_summaries: dict[str, dict[str, Any]] = {}
    group_names = tuple(sorted(grouped))
    per_group: dict[str, dict[str, Any]] = {group: {"conditions": {}} for group in group_names}
    for condition_index, condition in enumerate(all_conditions):
        rows_by_group: list[tuple[float, float, float, float, float, float]] = []
        for group in group_names:
            rows = grouped[group]
            values = (
                float(np.mean([r["conditions"][condition]["task_mae_m"] for r in rows])),
                float(np.mean([r["zero_task_mae_m"] for r in rows])),
                float(np.mean([r["fit_teacher_prior"]["task_mae_m"] for r in rows])),
                float(np.mean([r["conditions"][condition]["appearance_mae_m"] for r in rows])),
                float(np.mean([r["conditions"]["clean"]["edge_energy_mae_m"] for r in rows])),
                float(np.mean([r["conditions"][condition]["teacher_cosine"] for r in rows])),
            )
            rows_by_group.append(values)
            per_group[group]["conditions"][condition] = {
                "task_mae_m": values[0],
                "zero_task_mae_m": values[1],
                "fit_teacher_prior_task_mae_m": values[2],
                "appearance_mae_m": values[3],
                "clean_edge_energy_mae_m": values[4],
                "teacher_cosine": values[5],
                "salient_edge_mae_m": float(np.mean([
                    r["conditions"][condition]["salient_edge_mae_m"] for r in rows
                ])),
                "spatial_pool_task_mae_m": float(np.mean([
                    r["spatial_pool"][condition]["task_mae_m"] for r in rows
                ])),
            }
        array = np.asarray(rows_by_group, dtype=np.float64)
        prior_margin = array[:, 2] - array[:, 0]
        zero_margin = array[:, 1] - array[:, 0]
        stability_margin = array[:, 4] - array[:, 3]
        prior_ci = _bootstrap_ci(prior_margin, seed=seed + 10 * condition_index, replicates=replicates)
        zero_ci = _bootstrap_ci(zero_margin, seed=seed + 10 * condition_index + 1, replicates=replicates)
        stability_ci = _bootstrap_ci(stability_margin, seed=seed + 10 * condition_index + 2, replicates=replicates)
        macro = np.mean(array, axis=0)
        condition_summaries[condition] = {
            "task_mae_m": float(macro[0]),
            "zero_task_mae_m": float(macro[1]),
            "fit_teacher_prior_task_mae_m": float(macro[2]),
            "appearance_mae_m": float(macro[3]),
            "clean_edge_energy_mae_m": float(macro[4]),
            "teacher_cosine": float(macro[5]),
            "prior_minus_learned_m": float(np.mean(prior_margin)),
            "prior_minus_learned_ci95_m": list(prior_ci),
            "zero_minus_learned_m": float(np.mean(zero_margin)),
            "zero_minus_learned_ci95_m": list(zero_ci),
            "stability_margin_m": float(np.mean(stability_margin)),
            "stability_margin_ci95_m": list(stability_ci),
            "legacy_mechanism_gate": mechanism_condition_supported(
                visual_task_mae_m=float(macro[0]),
                zero_task_mae_m=float(macro[1]),
                appearance_mae_m=float(macro[3]),
                clean_edge_energy_mae_m=float(macro[4]),
                teacher_cosine=float(macro[5]),
            ),
        }

    correct: list[float] = []
    mismatch: list[float] = []
    removed: list[float] = []
    for group in group_names:
        rows = grouped[group]
        correct.append(float(np.mean([r["conditions"]["clean"]["task_mae_m"] for r in rows])))
        mismatch.append(float(np.mean([r["mismatched_clean"]["task_mae_m"] for r in rows])))
        removed.append(float(np.mean([r["image_removed"]["task_mae_m"] for r in rows])))
        per_group[group]["scene_information_controls"] = {
            "correct_clean_task_mae_m": correct[-1],
            "mismatched_clean_task_mae_m": mismatch[-1],
            "image_removed_task_mae_m": removed[-1],
            "mismatch_minus_correct_m": mismatch[-1] - correct[-1],
            "removed_minus_correct_m": removed[-1] - correct[-1],
            "uniform_fit_mean_DINO_task_mae_m": float(np.mean([
                r["uniform_fit_mean_DINO"]["task_mae_m"] for r in rows
            ])),
            "fit_teacher_prior_salient_edge_mae_m": float(np.mean([
                r["fit_teacher_prior"]["salient_edge_mae_m"] for r in rows
            ])),
            "mismatched_salient_edge_mae_m": float(np.mean([
                r["mismatched_clean"]["salient_edge_mae_m"] for r in rows
            ])),
            "removed_salient_edge_mae_m": float(np.mean([
                r["image_removed"]["salient_edge_mae_m"] for r in rows
            ])),
        }
    correct_array = np.asarray(correct)
    mismatch_margin = np.asarray(mismatch) - correct_array
    removed_margin = np.asarray(removed) - correct_array
    mismatch_ci = _bootstrap_ci(mismatch_margin, seed=seed + 1000, replicates=replicates)
    removed_ci = _bootstrap_ci(removed_margin, seed=seed + 1001, replicates=replicates)
    controls = {
        "correct_clean_task_mae_m": float(np.mean(correct_array)),
        "mismatched_clean_task_mae_m": float(np.mean(mismatch)),
        "image_removed_task_mae_m": float(np.mean(removed)),
        "mismatch_minus_correct_m": float(np.mean(mismatch_margin)),
        "mismatch_minus_correct_ci95_m": list(mismatch_ci),
        "removed_minus_correct_m": float(np.mean(removed_margin)),
        "removed_minus_correct_ci95_m": list(removed_ci),
        "uniform_fit_mean_DINO_task_mae_m": float(np.mean([
            per_group[group]["scene_information_controls"]["uniform_fit_mean_DINO_task_mae_m"]
            for group in group_names
        ])),
    }
    perturbation_summary: dict[str, dict[str, float]] = {}
    for condition in (*all_conditions, "mismatched_clean", "image_removed"):
        group_values = []
        for group in group_names:
            rows = grouped[group]
            item = {
                key: float(np.mean([r["perturbation_ledger"][condition][key] for r in rows]))
                for key in ("pixel_srgb_mae", "dense_feature_rms", "dense_feature_one_minus_cosine")
            }
            per_group[group].setdefault("perturbation_ledger", {})[condition] = item
            group_values.append(item)
        perturbation_summary[condition] = {
            key: float(np.mean([row[key] for row in group_values]))
            for key in ("pixel_srgb_mae", "dense_feature_rms", "dense_feature_one_minus_cosine")
        }
    severity_checks = {}
    for base in ("low_exposure", "low_contrast", "cool_cast"):
        strong = f"{base}_twice"
        severity_checks[base] = {
            key: perturbation_summary[strong][key] > perturbation_summary[base][key]
            for key in ("pixel_srgb_mae", "dense_feature_rms", "dense_feature_one_minus_cosine")
        }
    claim_1 = all(
        condition_summaries[name]["task_mae_m"] < condition_summaries[name]["zero_task_mae_m"]
        and condition_summaries[name]["task_mae_m"] < condition_summaries[name]["fit_teacher_prior_task_mae_m"]
        and condition_summaries[name]["prior_minus_learned_ci95_m"][0] > 0.0
        and condition_summaries[name]["teacher_cosine"] > 0.0
        and (name == "clean" or condition_summaries[name]["appearance_mae_m"] < condition_summaries[name]["clean_edge_energy_mae_m"])
        for name in CONFIRMATORY_CONDITIONS
    )
    claim_2 = mismatch_ci[0] > 0.0 and removed_ci[0] > 0.0
    decision = "SUPPORTED" if claim_1 and claim_2 else "MIXED" if claim_1 or claim_2 else "UNSUPPORTED"
    failure_counts = {
        condition: {
            "learned_not_better_than_zero": sum(
                per_group[group]["conditions"][condition]["task_mae_m"]
                >= per_group[group]["conditions"][condition]["zero_task_mae_m"]
                for group in group_names
            ),
            "learned_not_better_than_fit_prior": sum(
                per_group[group]["conditions"][condition]["task_mae_m"]
                >= per_group[group]["conditions"][condition]["fit_teacher_prior_task_mae_m"]
                for group in group_names
            ),
        }
        for condition in CONFIRMATORY_CONDITIONS
    }
    failure_counts["scene_information_controls"] = {
        "mismatch_not_worse_than_correct": int(np.sum(mismatch_margin <= 0.0)),
        "removed_not_worse_than_correct": int(np.sum(removed_margin <= 0.0)),
    }
    worst_group = max(
        group_names,
        key=lambda group: per_group[group]["conditions"]["clean"]["task_mae_m"],
    )
    return {
        "conditions": condition_summaries,
        "scene_information_controls": controls,
        "perturbation_summary": perturbation_summary,
        "stronger_than_base_checks": severity_checks,
        "per_group": per_group,
        "failure_group_counts": failure_counts,
        "worst_clean_task_group": worst_group,
        "claim_1_heldout_generalization_supported": bool(claim_1),
        "claim_2_scene_conditioning_supported": bool(claim_2),
        "decision": decision,
    }
