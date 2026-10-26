"""Post-hoc, read-only failure anatomy for the frozen R005E2B OOF outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


ARMS = ("V0_removed", "V1_mismatch", "V2_correct")
SEEDS = (20261025, 20261026, 20261027)
TIE = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _job_name(fold: int, arm: str, seed: int) -> str:
    return f"fold{fold}_{arm}_seed{seed}"


def _load_npz(path: Path) -> dict[str, npt.NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _choice(prediction: npt.NDArray[np.floating[Any]], mask: npt.NDArray[np.bool_]) -> int:
    valid = np.flatnonzero(mask)
    if valid.size == 0 or valid[0] != 0:
        raise ValueError("every panel must contain native candidate at index zero")
    return int(valid[np.argmax(prediction[valid])])


def _anchor_mse(
    prediction: npt.NDArray[np.floating[Any]],
    target: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.bool_],
) -> float:
    challenger = mask.copy()
    challenger[0] = False
    return float(np.square(prediction - target)[challenger].mean()) if challenger.any() else 0.0


def _pair_error(
    prediction: npt.NDArray[np.floating[Any]],
    target: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.bool_],
) -> tuple[int, int]:
    valid = np.flatnonzero(mask)
    errors = 0
    count = 0
    for offset, left in enumerate(valid):
        for right in valid[offset + 1 :]:
            truth = float(target[left] - target[right])
            if abs(truth) <= TIE:
                continue
            estimate = float(prediction[left] - prediction[right])
            predicted_sign = 1 if estimate > TIE else -1 if estimate < -TIE else 0
            errors += int(predicted_sign != int(np.sign(truth)))
            count += 1
    return errors, count


def _safe_correlation(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    if x.size < 2 or float(x.std()) <= 0.0 or float(y.std()) <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _summarize_channel(
    predictions: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
    masks: npt.NDArray[np.bool_],
) -> dict[str, float]:
    challenger = masks.copy()
    challenger[:, 0] = False
    p = predictions[challenger]
    t = targets[challenger]
    target_variance = float(np.var(t))
    slope = float(np.cov(t, p, ddof=0)[0, 1] / target_variance) if target_variance > 0 else 0.0
    pair_errors = 0
    pair_count = 0
    for row in range(predictions.shape[0]):
        errors, count = _pair_error(predictions[row], targets[row], masks[row])
        pair_errors += errors
        pair_count += count
    return {
        "prediction_mean": float(np.mean(p)),
        "target_mean": float(np.mean(t)),
        "prediction_std": float(np.std(p)),
        "target_std": float(np.std(t)),
        "std_ratio": float(np.std(p) / np.std(t)) if float(np.std(t)) > 0 else 0.0,
        "pearson": _safe_correlation(t.astype(np.float64), p.astype(np.float64)),
        "least_squares_slope": slope,
        "pair_ranking_error": float(pair_errors / pair_count) if pair_count else 0.0,
        "pair_count": float(pair_count),
    }


def _choice_change(
    reference: npt.NDArray[np.float64],
    alternative: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
    masks: npt.NDArray[np.bool_],
) -> dict[str, float]:
    beneficial = harmful = neutral = changed = 0
    utility_differences: list[float] = []
    for row in range(reference.shape[0]):
        reference_choice = _choice(reference[row], masks[row])
        alternative_choice = _choice(alternative[row], masks[row])
        delta = float(targets[row, alternative_choice] - targets[row, reference_choice])
        utility_differences.append(delta)
        changed += int(reference_choice != alternative_choice)
        beneficial += int(delta > TIE)
        harmful += int(delta < -TIE)
        neutral += int(abs(delta) <= TIE)
    count = reference.shape[0]
    return {
        "anchor_count": float(count),
        "choice_changed_fraction": float(changed / count),
        "beneficial_fraction": float(beneficial / count),
        "harmful_fraction": float(harmful / count),
        "neutral_fraction": float(neutral / count),
        "mean_alternative_minus_reference_true_utility": float(np.mean(utility_differences)),
    }


def _difficulty_bin(gap: float) -> str:
    if gap <= 0.01:
        return "top2_gap_le_0p01"
    if gap <= 0.05:
        return "top2_gap_0p01_to_0p05"
    return "top2_gap_gt_0p05"


def diagnose(run_root: Path, bundle_manifest_path: Path) -> dict[str, Any]:
    audit = json.loads((run_root / "audit.json").read_text("utf-8"))
    if audit.get("status") != "PASS" or audit.get("decision") != "STOP_CURRENT_VISUAL_MAIN_CLAIM":
        raise ValueError("failure anatomy is registered only for the audited R005E2B STOP result")
    manifest = json.loads(bundle_manifest_path.read_text("utf-8"))
    rows = {int(row["bundle_index"]): row for row in manifest["anchors"]}
    total = len(rows)
    if total != 672:
        raise ValueError("expected the frozen 672-anchor bundle")

    by_condition_seed: dict[tuple[str, int], dict[str, npt.NDArray[Any]]] = {}
    selected_epochs: dict[str, list[int]] = defaultdict(list)
    source_hashes: dict[str, str] = {}
    for arm in ARMS:
        for seed in SEEDS:
            prediction = np.zeros((total, 7), dtype=np.float64)
            target = np.zeros((total, 7), dtype=np.float64)
            mask = np.zeros((total, 7), dtype=np.bool_)
            mismatch_intervention = np.zeros((total, 7), dtype=np.float64)
            removed_intervention = np.zeros((total, 7), dtype=np.float64)
            seen: set[int] = set()
            for fold in range(6):
                name = _job_name(fold, arm, seed)
                root = run_root / "primary" / name
                result = json.loads((root / "result.json").read_text("utf-8"))
                archive_path = root / "predictions.npz"
                archive = _load_npz(archive_path)
                if _sha256(archive_path) != result["prediction_sha256"]:
                    raise ValueError(f"prediction hash mismatch: {name}")
                replay = _load_npz(run_root / "replay" / name / "predictions.npz")
                if set(archive) != set(replay) or any(
                    not np.array_equal(archive[key], replay[key]) for key in archive
                ):
                    raise ValueError(f"primary/replay mismatch: {name}")
                indices = archive["bundle_index"].astype(np.int64)
                if seen.intersection(indices.tolist()):
                    raise ValueError(f"duplicate OOF index: {name}")
                seen.update(indices.tolist())
                prediction[indices] = archive["prediction"]
                target[indices] = archive["target"]
                mask[indices] = archive["candidate_mask"]
                if arm == "V2_correct":
                    mismatch_intervention[indices] = archive["mismatch_intervention_prediction"]
                    removed_intervention[indices] = archive["removed_intervention_prediction"]
                selected_epochs[arm].append(int(result["selected_epoch"]))
                source_hashes[name] = result["prediction_sha256"]
            if len(seen) != total:
                raise ValueError(f"incomplete OOF coverage for {arm}/{seed}")
            by_condition_seed[(arm, seed)] = {
                "prediction": prediction,
                "target": target,
                "mask": mask,
            }
            if arm == "V2_correct":
                by_condition_seed[("V2_mismatch_intervention", seed)] = {
                    "prediction": mismatch_intervention,
                    "target": target,
                    "mask": mask,
                }
                by_condition_seed[("V2_removed_intervention", seed)] = {
                    "prediction": removed_intervention,
                    "target": target,
                    "mask": mask,
                }

    base_target = by_condition_seed[("V2_correct", SEEDS[0])]["target"]
    base_mask = by_condition_seed[("V2_correct", SEEDS[0])]["mask"]
    for payload in by_condition_seed.values():
        if not np.array_equal(payload["target"], base_target) or not np.array_equal(payload["mask"], base_mask):
            raise ValueError("target or panel membership differs across conditions")

    conditions = (*ARMS, "V2_mismatch_intervention", "V2_removed_intervention")
    channel_summary: dict[str, Any] = {}
    seed_choices: dict[str, npt.NDArray[np.int64]] = {}
    seed_predictions: dict[str, npt.NDArray[np.float64]] = {}
    for condition in conditions:
        stacked = np.stack([by_condition_seed[(condition, seed)]["prediction"] for seed in SEEDS])
        seed_predictions[condition] = stacked
        channel_summary[condition] = _summarize_channel(
            stacked.reshape(-1, 7),
            np.repeat(base_target[None, ...], len(SEEDS), axis=0).reshape(-1, 7),
            np.repeat(base_mask[None, ...], len(SEEDS), axis=0).reshape(-1, 7),
        )
        choices = np.asarray(
            [[_choice(stacked[s, row], base_mask[row]) for row in range(total)] for s in range(len(SEEDS))],
            dtype=np.int64,
        )
        seed_choices[condition] = choices
        channel_summary[condition]["three_seed_choice_unanimous_fraction"] = float(
            np.mean(np.all(choices == choices[0:1], axis=0))
        )
        prediction_dispersion = np.std(stacked, axis=0)
        valid = np.repeat(base_mask[None, ...], len(SEEDS), axis=0)
        channel_summary[condition]["mean_candidate_prediction_seed_std"] = float(
            np.mean(np.repeat(prediction_dispersion[None, ...], len(SEEDS), axis=0)[valid])
        )

    change_summary: dict[str, Any] = {}
    for seed in SEEDS:
        correct = by_condition_seed[("V2_correct", seed)]
        for other in ("V0_removed", "V1_mismatch", "V2_mismatch_intervention", "V2_removed_intervention"):
            key = f"V2_correct_to_{other}_seed{seed}"
            change_summary[key] = _choice_change(
                correct["prediction"],
                by_condition_seed[(other, seed)]["prediction"],
                base_target,
                base_mask,
            )

    mismatch_categories: dict[str, Any] = {}
    for control in ("V0_removed", "V1_mismatch"):
        counts = defaultdict(int)
        mse_deltas: list[float] = []
        utility_deltas: list[float] = []
        for seed in SEEDS:
            correct_prediction = by_condition_seed[("V2_correct", seed)]["prediction"]
            control_prediction = by_condition_seed[(control, seed)]["prediction"]
            for row in range(total):
                mse_delta = _anchor_mse(control_prediction[row], base_target[row], base_mask[row]) - _anchor_mse(
                    correct_prediction[row], base_target[row], base_mask[row]
                )
                correct_choice = _choice(correct_prediction[row], base_mask[row])
                control_choice = _choice(control_prediction[row], base_mask[row])
                utility_delta = float(base_target[row, correct_choice] - base_target[row, control_choice])
                mse_deltas.append(mse_delta)
                utility_deltas.append(utility_delta)
                improved = mse_delta > TIE
                if not improved:
                    counts["mse_not_improved"] += 1
                elif utility_delta > TIE:
                    counts["mse_improved_choice_beneficial"] += 1
                elif utility_delta < -TIE:
                    counts["mse_improved_choice_harmful"] += 1
                else:
                    counts["mse_improved_choice_neutral"] += 1
        mismatch_categories[control] = {
            "counts": dict(counts),
            "observation_count": len(mse_deltas),
            "mean_control_minus_correct_anchor_mse": float(np.mean(mse_deltas)),
            "mean_correct_minus_control_choice_utility": float(np.mean(utility_deltas)),
            "pearson_mse_improvement_vs_choice_utility": _safe_correlation(
                np.asarray(mse_deltas, dtype=np.float64),
                np.asarray(utility_deltas, dtype=np.float64),
            ),
        }

    difficulty: dict[str, Any] = {}
    gap_bins: dict[str, list[int]] = defaultdict(list)
    gaps: list[float] = []
    strict_native_opportunity = 0
    oracle_gains: list[float] = []
    best_set_sizes: list[int] = []
    for row in range(total):
        valid_target = base_target[row, base_mask[row]]
        values = np.sort(valid_target)
        gap = float(values[-1] - values[-2]) if values.size >= 2 else 0.0
        gaps.append(gap)
        gap_bins[_difficulty_bin(gap)].append(row)
        best = float(values[-1])
        gain = best - float(base_target[row, 0])
        oracle_gains.append(gain)
        strict_native_opportunity += int(gain > TIE)
        best_set_sizes.append(int(np.sum(np.abs(valid_target - best) <= TIE)))
    for condition in conditions:
        condition_rows: dict[str, Any] = {}
        for name, indices in gap_bins.items():
            regrets: list[float] = []
            for seed in SEEDS:
                prediction = by_condition_seed[(condition, seed)]["prediction"]
                for row in indices:
                    selected = _choice(prediction[row], base_mask[row])
                    regrets.append(float(np.max(base_target[row, base_mask[row]]) - base_target[row, selected]))
            condition_rows[name] = {
                "anchor_count": len(indices),
                "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
            }
        difficulty[condition] = condition_rows

    epoch_summary = {
        arm: {
            "count": len(values),
            "min": min(values),
            "median": float(np.median(values)),
            "max": max(values),
            "mean": float(np.mean(values)),
        }
        for arm, values in selected_epochs.items()
    }
    return {
        "schema": "aorta.r005e2b.failure_anatomy.v1",
        "status": "POSTHOC_EXPLANATORY_ONLY",
        "parent_support": audit["support"],
        "parent_decision": audit["decision"],
        "claim_boundary": (
            "Explains the frozen source-fit OOF STOP only; does not alter the parent decision, "
            "prove information-theoretic impossibility, or authorize new training."
        ),
        "input_provenance": {
            "run_audit_sha256": _sha256(run_root / "audit.json"),
            "bundle_manifest_sha256": _sha256(bundle_manifest_path),
            "prediction_artifact_count": len(source_hashes),
        },
        "channel_summary": channel_summary,
        "choice_change_summary": change_summary,
        "regression_decision_mismatch": mismatch_categories,
        "difficulty": {
            "decision_topology": {
                "anchor_count": total,
                "strict_oracle_improvement_anchor_count": strict_native_opportunity,
                "strict_oracle_improvement_anchor_fraction": float(strict_native_opportunity / total),
                "exact_top2_tie_anchor_count": int(np.sum(np.asarray(gaps) <= TIE)),
                "exact_top2_tie_anchor_fraction": float(np.mean(np.asarray(gaps) <= TIE)),
                "mean_oracle_gain_over_native": float(np.mean(oracle_gains)),
                "mean_positive_oracle_gain_over_native": float(
                    np.mean([value for value in oracle_gains if value > TIE])
                ) if strict_native_opportunity else 0.0,
                "mean_best_set_size": float(np.mean(best_set_sizes)),
                "max_best_set_size": max(best_set_sizes),
            },
            "top2_gap": {
                "min": min(gaps),
                "median": float(np.median(gaps)),
                "mean": float(np.mean(gaps)),
                "max": max(gaps),
            },
            "by_condition": difficulty,
        },
        "optimization": {"selected_epoch": epoch_summary},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = diagnose(args.run_root, args.bundle_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "parent_decision": result["parent_decision"]}))
