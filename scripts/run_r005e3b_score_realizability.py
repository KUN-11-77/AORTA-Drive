"""Run the frozen R005E3B one-factor score-realizability qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from aorta_drive.models.full_context_scorer import FullContextSetScorer
from aorta_drive.training.r005e2_science import (
    ARM_CORRECT,
    ARM_MISMATCH,
    ARM_REMOVED,
    ScientificDataset,
)
from aorta_drive.vision.feature_cache import sha256_file
from run_r005e2_fold_job import (
    _dataset,
    _loader,
    _new_model,
    _physical_gpu_identity,
    _predict,
    _schedule,
    _tensor_state_sha256,
    _write_json,
)
from run_r005e3a_fold_job import (
    _temperature_from_outer_train,
    _train_soft_epoch,
)


OUTER_FOLD = 0
SEED = 20261025
TIE_TOLERANCE = 1.0e-12


def _objective_parent_result(
    parent_run_root: Path, *, outer_fold: int, seed: int
) -> tuple[Path, dict[str, Any]]:
    root = parent_run_root / "primary" / f"fold{outer_fold}_seed{seed}"
    result = cast(dict[str, Any], json.loads((root / "result.json").read_text("utf-8")))
    if (
        result.get("schema") != "aorta.r005e3a.fold_job.v1"
        or result.get("status") != "COMPLETED"
        or result.get("outer_fold") != outer_fold
        or result.get("seed") != seed
        or result.get("objective") != "soft_selection_regret"
    ):
        raise ValueError("R005E3A parent job contract failed")
    return root, result


def _decision_summary(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    *,
    temperature: float,
) -> dict[str, float | int]:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask shapes differ")
    if not bool(mask[:, 0].all()):
        raise ValueError("native must be valid")
    if not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("valid values must be finite")
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
    challenger_scores = np.asarray(prediction[challenger], dtype=np.float64)
    logits = np.where(
        mask, np.asarray(prediction, dtype=np.float64) / temperature, -np.inf
    )
    logits -= np.max(logits, axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= np.sum(probability, axis=1, keepdims=True)
    expected_utility = np.sum(
        probability * np.where(mask, np.asarray(target, dtype=np.float64), 0.0),
        axis=1,
        keepdims=True,
    )
    score_gradient = (
        probability
        / temperature
        * (expected_utility - np.asarray(target, dtype=np.float64))
    )
    positive_oracle = mask & (target > target[:, :1] + TIE_TOLERANCE)
    positive_gradient = np.abs(score_gradient[positive_oracle])
    return {
        "anchor_count": int(prediction.shape[0]),
        "mean_gain_over_native": mean_gain,
        "mean_oracle_gain_over_native": mean_oracle_gain,
        "mean_hard_regret": float(np.mean(regret)),
        "absolute_oracle_capture": (
            mean_gain / mean_oracle_gain if mean_oracle_gain > 0.0 else math.nan
        ),
        "native_choice_fraction": float(np.mean(selected == 0)),
        "nonnative_choice_count": int(np.sum(nonnative)),
        "beneficial_nonnative_count": int(np.sum(beneficial)),
        "harmful_nonnative_count": int(np.sum(harmful)),
        "neutral_nonnative_count": int(np.sum(neutral)),
        "challenger_score_min": float(np.min(challenger_scores)),
        "challenger_score_max": float(np.max(challenger_scores)),
        "challenger_score_std": float(np.std(challenger_scores)),
        "challenger_exact_minus_one_fraction": float(
            np.mean(challenger_scores == -1.0)
        ),
        "positive_oracle_score_gradient_abs_mean": (
            float(np.mean(positive_gradient)) if positive_gradient.size else math.nan
        ),
        "positive_oracle_score_gradient_abs_min": (
            float(np.min(positive_gradient)) if positive_gradient.size else math.nan
        ),
    }


def _predict_summary(
    model: FullContextSetScorer,
    loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    temperature: float,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
    dict[str, float | int],
]:
    indices, prediction, target, mask = _predict(model, loader, device)
    return (
        indices,
        prediction,
        target,
        mask,
        _decision_summary(prediction, target, mask, temperature=temperature),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    if args.outer_fold != OUTER_FOLD or args.seed != SEED:
        raise ValueError("R005E3B freezes fold 0 and seed 20261025")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("R005E3B requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("R005E3B requires CUDA")
    args.output_directory.mkdir(parents=True)
    started = time.perf_counter()
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
    fold = next(
        row
        for row in cast(list[dict[str, Any]], crossfit["outer_folds"])
        if int(row["outer_fold"]) == OUTER_FOLD
    )
    train_groups = cast(list[str], fold["outer_train_groups"])
    train = _dataset(
        groups=train_groups,
        stage="refit_train",
        arm=ARM_CORRECT,
        outer_fold=OUTER_FOLD,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    mismatch_train = _dataset(
        groups=train_groups,
        stage="refit_train",
        arm=ARM_MISMATCH,
        outer_fold=OUTER_FOLD,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    removed_train = _dataset(
        groups=train_groups,
        stage="refit_train",
        arm=ARM_REMOVED,
        outer_fold=OUTER_FOLD,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    parent_root, parent = _objective_parent_result(
        args.parent_run_root, outer_fold=OUTER_FOLD, seed=SEED
    )
    selected_epoch = int(parent["selected_epoch"])
    temperature, temperature_ledger = _temperature_from_outer_train(
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        groups=train_groups,
    )
    device = torch.device("cuda")
    model, initial_hash = _new_model(SEED, device)
    if initial_hash != parent["initial_model_state_sha256"]:
        raise ValueError("initialization differs from frozen parent")
    model.output_activation = "identity"
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    train_loader = _loader(train, shuffle=True, seed=SEED, workers=args.workers)
    condition_datasets: dict[str, ScientificDataset] = {
        "correct": train,
        "S1_refit_train_mismatch": mismatch_train,
        "removed": removed_train,
    }
    condition_loaders = {
        name: _loader(dataset, shuffle=False, seed=SEED, workers=args.workers)
        for name, dataset in condition_datasets.items()
    }
    expected_steps = selected_epoch * len(train_loader)
    scheduler = _schedule(optimizer, expected_steps)

    _, _, _, _, initial_summary = _predict_summary(
        model,
        condition_loaders["correct"],
        device,
        temperature=temperature,
    )
    epoch_ledger: list[dict[str, Any]] = []
    training_order: list[int] = []
    for epoch in range(1, selected_epoch + 1):
        training_order.extend(
            _train_soft_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                temperature=temperature,
            )
        )
        _, _, _, _, summary = _predict_summary(
            model,
            condition_loaders["correct"],
            device,
            temperature=temperature,
        )
        epoch_ledger.append({"epoch": epoch, **summary})
    if len(training_order) != expected_steps:
        raise RuntimeError("optimizer step budget differs")

    payload: dict[str, npt.NDArray[Any]] = {}
    condition_summaries: dict[str, dict[str, float | int]] = {}
    expected_indices: npt.NDArray[np.int64] | None = None
    expected_target: npt.NDArray[np.float32] | None = None
    expected_mask: npt.NDArray[np.bool_] | None = None
    for name, loader in condition_loaders.items():
        indices, prediction, target, mask, summary = _predict_summary(
            model,
            loader,
            device,
            temperature=temperature,
        )
        if expected_indices is None:
            expected_indices, expected_target, expected_mask = indices, target, mask
        elif (
            expected_target is None
            or expected_mask is None
            or not (
                np.array_equal(expected_indices, indices)
                and np.array_equal(expected_target, target)
                and np.array_equal(expected_mask, mask)
            )
        ):
            raise ValueError("condition roster, target, or mask differs")
        payload[f"{name}_prediction"] = prediction
        condition_summaries[name] = summary
    if expected_indices is None or expected_target is None or expected_mask is None:
        raise RuntimeError("no predictions produced")
    payload.update(
        {
            "bundle_index": expected_indices,
            "target": expected_target,
            "candidate_mask": expected_mask,
        }
    )
    prediction_path = args.output_directory / "predictions.npz"
    np.savez(prediction_path, **payload)
    checkpoint_path = args.output_directory / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "output_activation": "identity",
            "outer_fold": OUTER_FOLD,
            "seed": SEED,
            "selected_epoch": selected_epoch,
            "temperature": temperature,
        },
        checkpoint_path,
    )
    result = {
        "schema": "aorta.r005e3b.score_realizability.v1",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold": OUTER_FOLD,
        "seed": SEED,
        "output_activation": "identity",
        "objective": "soft_selection_regret",
        "temperature": temperature,
        "temperature_ledger": temperature_ledger,
        "selected_epoch": selected_epoch,
        "optimizer_step_count": expected_steps,
        "training_bundle_index_order_sha256": hashlib.sha256(
            np.asarray(training_order, dtype=np.int64).tobytes()
        ).hexdigest(),
        "initial_model_state_sha256": initial_hash,
        "parent_initial_model_state_sha256": parent["initial_model_state_sha256"],
        "parent_job_result_sha256": sha256_file(parent_root / "result.json"),
        "input_sha256": {
            "bundle_manifest": sha256_file(args.bundle_root / "manifest.json"),
            "crossfit_manifest": sha256_file(args.crossfit_manifest),
            "mismatch_manifest": sha256_file(args.mismatch_manifest),
            "visual_manifest": sha256_file(args.visual_root / "manifest.json"),
            "parent_job_result": sha256_file(parent_root / "result.json"),
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_train_summary": initial_summary,
        "epoch_ledger": epoch_ledger,
        "condition_summaries": condition_summaries,
        "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "final_model_state_sha256": _tensor_state_sha256(model),
        "code_commit": args.code_commit,
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config),
        "physical_gpu": _physical_gpu_identity(args.physical_gpu),
        "runtime_seconds": time.perf_counter() - started,
        "development_calibration_test_navdream_target_accessed": False,
        "scientific_metrics_emitted": False,
        "claim_boundary": "outer-train optimization and realizability only",
    }
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--mismatch-manifest", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
