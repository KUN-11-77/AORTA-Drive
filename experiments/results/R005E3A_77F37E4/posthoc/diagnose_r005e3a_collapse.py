"""Read-only post-result diagnosis for R005E3A score collapse.

This script never trains, selects a checkpoint, or changes the frozen R005E3A
decision.  It replays each primary checkpoint on that job's outer-train split
and reports whether the apparent OOF improvement came from learning positive
challenger utility or from reverting to the native action.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from aorta_drive.models.full_context_scorer import FullContextSetScorer
from aorta_drive.training.r005e2_science import ARM_CORRECT
from run_r005e2_fold_job import _dataset, _loader, _new_model, _predict
from run_r005e3a_fold_job import SEEDS


def summarize_panel_predictions(
    prediction: npt.NDArray[np.floating[Any]],
    target: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.bool_],
) -> dict[str, float | int]:
    """Summarize hard decisions without treating padded candidates as valid."""

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if prediction.ndim != 2 or prediction.shape[1] < 1:
        raise ValueError("panel arrays must have shape [anchor, candidate]")
    if not bool(np.all(mask[:, 0])):
        raise ValueError("native candidate must be valid for every anchor")
    valid_prediction = np.asarray(prediction[mask], dtype=np.float64)
    valid_target = np.asarray(target[mask], dtype=np.float64)
    if not np.isfinite(valid_prediction).all() or not np.isfinite(valid_target).all():
        raise ValueError("valid predictions and targets must be finite")

    masked_prediction = np.where(mask, prediction, -np.inf)
    masked_target = np.where(mask, target, -np.inf)
    selected = np.argmax(masked_prediction, axis=1)
    row = np.arange(prediction.shape[0])
    native_utility = np.asarray(target[:, 0], dtype=np.float64)
    selected_utility = np.asarray(target[row, selected], dtype=np.float64)
    oracle_utility = np.asarray(np.max(masked_target, axis=1), dtype=np.float64)
    selected_gain = selected_utility - native_utility
    oracle_gain = oracle_utility - native_utility
    hard_regret = oracle_utility - selected_utility

    challenger_mask = mask.copy()
    challenger_mask[:, 0] = False
    challenger_scores = np.asarray(prediction[challenger_mask], dtype=np.float64)
    if challenger_scores.size == 0:
        raise ValueError("diagnostic requires at least one valid challenger")
    mean_oracle_gain = float(np.mean(oracle_gain))
    mean_selected_gain = float(np.mean(selected_gain))
    absolute_capture = (
        mean_selected_gain / mean_oracle_gain if mean_oracle_gain > 0.0 else math.nan
    )
    return {
        "anchor_count": int(prediction.shape[0]),
        "valid_candidate_count": int(mask.sum()),
        "valid_challenger_count": int(challenger_scores.size),
        "valid_score_mean": float(np.mean(valid_prediction)),
        "valid_score_std": float(np.std(valid_prediction)),
        "valid_score_min": float(np.min(valid_prediction)),
        "valid_score_max": float(np.max(valid_prediction)),
        "challenger_score_minus_one_fraction": float(
            np.mean(challenger_scores == -1.0)
        ),
        "native_choice_fraction": float(np.mean(selected == 0)),
        "nonnative_choice_count": int(np.sum(selected != 0)),
        "mean_selected_gain_over_native": mean_selected_gain,
        "mean_oracle_gain_over_native": mean_oracle_gain,
        "mean_hard_regret": float(np.mean(hard_regret)),
        "absolute_oracle_capture": absolute_capture,
    }


def _load_model(
    checkpoint_path: Path, seed: int, device: torch.device
) -> FullContextSetScorer:
    model, _ = _new_model(seed, device)
    checkpoint = cast(
        dict[str, Any],
        torch.load(checkpoint_path, map_location=device, weights_only=False),
    )
    model.load_state_dict(cast(dict[str, torch.Tensor], checkpoint["model"]))
    model.eval()
    return model


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostic requested but CUDA is unavailable")
    bundle = cast(
        dict[str, Any],
        json.loads((args.bundle_root / "manifest.json").read_text("utf-8")),
    )
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text("utf-8"))
    )
    mismatch = cast(
        dict[str, Any], json.loads(args.mismatch_manifest.read_text("utf-8"))
    )
    jobs: list[dict[str, Any]] = []
    for outer_fold in range(6):
        fold = next(
            row
            for row in cast(list[dict[str, Any]], crossfit["outer_folds"])
            if int(row["outer_fold"]) == outer_fold
        )
        for seed in SEEDS:
            job_name = f"fold{outer_fold}_seed{seed}"
            checkpoint_path = args.run_root / "primary" / job_name / "checkpoint.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            dataset = _dataset(
                groups=cast(list[str], fold["outer_train_groups"]),
                stage="refit_train",
                arm=ARM_CORRECT,
                outer_fold=outer_fold,
                bundle_root=args.bundle_root,
                bundle_manifest=bundle,
                visual_root=args.visual_root,
                mismatch_manifest=mismatch,
            )
            loader = _loader(dataset, shuffle=False, seed=seed, workers=args.workers)
            model = _load_model(checkpoint_path, seed, device)
            _, prediction, target, mask = _predict(model, loader, device)
            summary = summarize_panel_predictions(prediction, target, mask)
            jobs.append(
                {
                    "job": job_name,
                    "outer_fold": outer_fold,
                    "seed": seed,
                    **summary,
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "schema": "aorta.r005e3a.posthoc_collapse_diagnosis.v1",
        "status": "POSTHOC_READ_ONLY_DIAGNOSIS",
        "changes_frozen_decision": False,
        "scientific_run_root": str(args.run_root),
        "job_count": len(jobs),
        "all_jobs_native_only": all(
            int(job["nonnative_choice_count"]) == 0 for job in jobs
        ),
        "all_jobs_challenger_exact_minus_one": all(
            float(job["challenger_score_minus_one_fraction"]) == 1.0 for job in jobs
        ),
        "mean_absolute_oracle_capture": float(
            np.mean([float(job["absolute_oracle_capture"]) for job in jobs])
        ),
        "jobs": jobs,
        "claim_boundary": [
            "posthoc_mechanism_diagnosis_only",
            "does_not_change_frozen_MIXED_decision",
            "does_not_test_Differential_Query_or_NavDream",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), "utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--mismatch-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
