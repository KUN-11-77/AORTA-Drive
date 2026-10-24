"""Group-level inference helpers for R004H6."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from aorta_drive.evaluation.appearance_mechanism import mechanism_condition_supported

CONFIRMATORY_CONDITIONS = ("clean", "low_exposure", "low_contrast", "cool_cast")


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
    for condition_index, condition in enumerate(all_conditions):
        rows_by_group: list[tuple[float, float, float, float, float, float]] = []
        for group in group_names:
            rows = grouped[group]
            rows_by_group.append(
                (
                    float(np.mean([r["conditions"][condition]["task_mae_m"] for r in rows])),
                    float(np.mean([r["zero_task_mae_m"] for r in rows])),
                    float(np.mean([r["fit_teacher_prior"]["task_mae_m"] for r in rows])),
                    float(np.mean([r["conditions"][condition]["appearance_mae_m"] for r in rows])),
                    float(np.mean([r["conditions"]["clean"]["edge_energy_mae_m"] for r in rows])),
                    float(np.mean([r["conditions"][condition]["teacher_cosine"] for r in rows])),
                )
            )
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
    return {
        "conditions": condition_summaries,
        "scene_information_controls": controls,
        "claim_1_heldout_generalization_supported": bool(claim_1),
        "claim_2_scene_conditioning_supported": bool(claim_2),
        "decision": decision,
    }
