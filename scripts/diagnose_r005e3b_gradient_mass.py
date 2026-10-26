"""Diagnose R005E3B opportunity and gradient mass without updating weights."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
import yaml  # type: ignore[import-untyped]
from torch import Tensor

from aorta_drive.models.full_context_scorer import FullContextSetScorer
from aorta_drive.training.r005e2_science import ARM_CORRECT
from aorta_drive.vision.feature_cache import sha256_file
from run_r005e2_fold_job import (
    _dataset,
    _forward,
    _loader,
    _new_model,
    _physical_gpu_identity,
    _predict,
    _tensor_state_sha256,
    _to_device,
    _write_json,
)
from run_r005e3a_fold_job import _temperature_from_outer_train


OUTER_FOLD = 0
SEED = 20261025
TIE_TOLERANCE = 1.0e-12


def _score_gradient_arrays(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    *,
    temperature: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
]:
    logits = np.where(
        mask, np.asarray(prediction, dtype=np.float64) / temperature, -np.inf
    )
    logits -= np.max(logits, axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= np.sum(probability, axis=1, keepdims=True)
    utility = np.asarray(target, dtype=np.float64)
    expected = np.sum(probability * np.where(mask, utility, 0.0), axis=1, keepdims=True)
    score_gradient = probability / temperature * (expected - utility)
    challenger = mask.copy()
    challenger[:, 0] = False
    positive = challenger & (target > target[:, :1] + TIE_TOLERANCE)
    nonpositive = challenger & ~positive
    return probability, score_gradient, positive, nonpositive


def _category_summary(
    probability: npt.NDArray[np.float64],
    score_gradient: npt.NDArray[np.float64],
    target: npt.NDArray[np.float32],
    category: npt.NDArray[np.bool_],
) -> dict[str, float | int]:
    count = int(np.sum(category))
    if count == 0:
        return {
            "count": 0,
            "utility_sum": 0.0,
            "utility_abs_sum": 0.0,
            "probability_sum": 0.0,
            "score_gradient_sum": 0.0,
            "score_gradient_abs_sum": 0.0,
            "score_gradient_abs_mean": 0.0,
        }
    utility = np.asarray(target[category], dtype=np.float64)
    gradient = score_gradient[category]
    return {
        "count": count,
        "utility_sum": float(np.sum(utility)),
        "utility_abs_sum": float(np.sum(np.abs(utility))),
        "probability_sum": float(np.sum(probability[category])),
        "score_gradient_sum": float(np.sum(gradient)),
        "score_gradient_abs_sum": float(np.sum(np.abs(gradient))),
        "score_gradient_abs_mean": float(np.mean(np.abs(gradient))),
    }


def _score_diagnostics(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    *,
    temperature: float,
) -> dict[str, Any]:
    probability, score_gradient, positive, nonpositive = _score_gradient_arrays(
        prediction, target, mask, temperature=temperature
    )
    challenger = mask.copy()
    challenger[:, 0] = False
    opportunity = np.any(positive, axis=1)
    positive_gradient = score_gradient[positive]
    nonpositive_gradient = score_gradient[nonpositive]
    challenger_count = int(np.sum(challenger))
    return {
        "anchor_count": int(prediction.shape[0]),
        "valid_challenger_count": challenger_count,
        "opportunity_anchor_count": int(np.sum(opportunity)),
        "opportunity_anchor_fraction": float(np.mean(opportunity)),
        "positive_challenger_fraction": float(
            int(np.sum(positive)) / max(1, challenger_count)
        ),
        "positive": _category_summary(probability, score_gradient, target, positive),
        "nonpositive": _category_summary(
            probability, score_gradient, target, nonpositive
        ),
        "positive_raise_direction_fraction": (
            float(np.mean(positive_gradient < 0.0)) if positive_gradient.size else 0.0
        ),
        "nonpositive_lower_direction_fraction": (
            float(np.mean(nonpositive_gradient > 0.0))
            if nonpositive_gradient.size
            else 0.0
        ),
    }


def _accumulators(
    model: FullContextSetScorer,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    positive = {
        name: torch.zeros_like(parameter, dtype=torch.float32)
        for name, parameter in model.named_parameters()
    }
    nonpositive = {
        name: torch.zeros_like(parameter, dtype=torch.float32)
        for name, parameter in model.named_parameters()
    }
    return positive, nonpositive


def _add_gradients(
    accumulator: dict[str, Tensor],
    names: list[str],
    gradients: tuple[Tensor | None, ...],
) -> None:
    for name, gradient in zip(names, gradients, strict=True):
        if gradient is not None:
            accumulator[name].add_(gradient.detach().float())


def _gradient_geometry(
    positive: dict[str, Tensor], nonpositive: dict[str, Tensor]
) -> dict[str, float]:
    dot = torch.zeros((), device=next(iter(positive.values())).device)
    positive_sq = torch.zeros_like(dot)
    nonpositive_sq = torch.zeros_like(dot)
    for name in positive:
        left = positive[name]
        right = nonpositive[name]
        dot += torch.sum(left * right)
        positive_sq += torch.sum(left.square())
        nonpositive_sq += torch.sum(right.square())
    positive_norm = torch.sqrt(positive_sq)
    nonpositive_norm = torch.sqrt(nonpositive_sq)
    if float(positive_norm) <= 0.0 or not bool(torch.isfinite(positive_norm)):
        raise FloatingPointError("positive parameter-gradient norm is invalid")
    if not bool(torch.isfinite(nonpositive_norm)):
        raise FloatingPointError("nonpositive parameter-gradient norm is invalid")
    denominator = positive_norm * nonpositive_norm
    cosine = dot / denominator if float(denominator) > 0.0 else torch.zeros_like(dot)
    combined_norm = torch.sqrt(positive_sq + nonpositive_sq + 2.0 * dot)
    return {
        "positive_parameter_gradient_norm": float(positive_norm.cpu()),
        "nonpositive_parameter_gradient_norm": float(nonpositive_norm.cpu()),
        "nonpositive_to_positive_norm_ratio": float(
            (nonpositive_norm / positive_norm).cpu()
        ),
        "positive_nonpositive_cosine": float(cosine.cpu()),
        "combined_parameter_gradient_norm": float(combined_norm.cpu()),
    }


def _parameter_gradient_decomposition(
    model: FullContextSetScorer,
    loader: torch.utils.data.DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    temperature: float,
) -> dict[str, float | int]:
    model.eval()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    positive_accumulator, nonpositive_accumulator = _accumulators(model)
    anchor_count = 0
    for cpu_batch in loader:
        anchor_count += 1
        batch = _to_device(cpu_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = _forward(model, batch)
        utility = batch["target_margin"].float()
        candidate_mask = batch["candidate_mask"]
        logits = prediction.float() / temperature
        probability = torch.softmax(
            logits.masked_fill(~candidate_mask, float("-inf")), dim=1
        )
        expected = (probability * utility.masked_fill(~candidate_mask, 0.0)).sum(
            dim=1, keepdim=True
        )
        score_gradient = probability / temperature * (expected - utility)
        challenger = candidate_mask.clone()
        challenger[:, 0] = False
        positive_mask = challenger & (utility > utility[:, :1] + TIE_TOLERANCE)
        nonpositive_mask = challenger & ~positive_mask
        positive_gradients = torch.autograd.grad(
            prediction,
            parameters,
            grad_outputs=(score_gradient * positive_mask).to(prediction.dtype),
            retain_graph=True,
            allow_unused=True,
        )
        nonpositive_gradients = torch.autograd.grad(
            prediction,
            parameters,
            grad_outputs=(score_gradient * nonpositive_mask).to(prediction.dtype),
            allow_unused=True,
        )
        _add_gradients(positive_accumulator, names, positive_gradients)
        _add_gradients(nonpositive_accumulator, names, nonpositive_gradients)
    geometry = _gradient_geometry(positive_accumulator, nonpositive_accumulator)
    return {"anchor_count": anchor_count, **geometry}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("diagnostic requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("diagnostic requires CUDA")
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text("utf-8")))
    bundle_manifest = cast(
        dict[str, Any],
        json.loads((args.bundle_root / "manifest.json").read_text("utf-8")),
    )
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text("utf-8"))
    )
    mismatch = cast(
        dict[str, Any], json.loads(args.mismatch_manifest.read_text("utf-8"))
    )
    frozen_data = cast(dict[str, Any], config["data"])
    actual_inputs = {
        "bundle_manifest_sha256": sha256_file(args.bundle_root / "manifest.json"),
        "visual_manifest_sha256": sha256_file(args.visual_root / "manifest.json"),
        "crossfit_manifest_sha256": sha256_file(args.crossfit_manifest),
        "mismatch_manifest_sha256": sha256_file(args.mismatch_manifest),
        "r005e3b_primary_predictions_sha256": sha256_file(args.final_predictions),
    }
    if any(actual_inputs[key] != frozen_data[key] for key in actual_inputs):
        raise ValueError("diagnostic input differs from preregistered hash")
    fold = next(
        row
        for row in cast(list[dict[str, Any]], crossfit["outer_folds"])
        if int(row["outer_fold"]) == OUTER_FOLD
    )
    groups = cast(list[str], fold["outer_train_groups"])
    temperature, temperature_ledger = _temperature_from_outer_train(
        bundle_root=args.bundle_root,
        bundle_manifest=bundle_manifest,
        groups=groups,
    )
    dataset = _dataset(
        groups=groups,
        stage="refit_train",
        arm=ARM_CORRECT,
        outer_fold=OUTER_FOLD,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle_manifest,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    device = torch.device("cuda")
    model, initial_hash = _new_model(SEED, device)
    model.output_activation = "identity"
    frozen_model = cast(dict[str, Any], config["model"])
    if initial_hash != frozen_model["initialization_sha256"]:
        raise ValueError("initialization differs from preregistration")
    before_hash = _tensor_state_sha256(model)
    evaluation_loader = _loader(dataset, shuffle=False, seed=SEED, workers=args.workers)
    indices, initial_prediction, target, mask = _predict(
        model, evaluation_loader, device
    )
    final = np.load(args.final_predictions, allow_pickle=False)
    if not (
        np.array_equal(indices, final["bundle_index"])
        and np.array_equal(target, final["target"])
        and np.array_equal(mask, final["candidate_mask"])
    ):
        raise ValueError("final prediction roster differs from initial diagnostic")
    gradient_loader = _loader(dataset, shuffle=False, seed=SEED, workers=args.workers)
    parameter_geometry = _parameter_gradient_decomposition(
        model, gradient_loader, device, temperature=temperature
    )
    after_hash = _tensor_state_sha256(model)
    if before_hash != after_hash or initial_hash != before_hash:
        raise RuntimeError("read-only diagnostic changed model state")
    initial_diagnostics = _score_diagnostics(
        initial_prediction, target, mask, temperature=temperature
    )
    final_diagnostics = _score_diagnostics(
        cast(npt.NDArray[np.float32], final["correct_prediction"]),
        target,
        mask,
        temperature=temperature,
    )
    return {
        "schema": "aorta.r005e3b.gradient_mass_diagnostic.v1",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold": OUTER_FOLD,
        "seed": SEED,
        "temperature": temperature,
        "temperature_ledger": temperature_ledger,
        "initial_model_state_sha256": initial_hash,
        "model_state_unchanged": True,
        "initial_score_diagnostics": initial_diagnostics,
        "final_score_diagnostics": final_diagnostics,
        "initial_parameter_gradient_decomposition": parameter_geometry,
        "input_sha256": {
            **actual_inputs,
            "config": sha256_file(args.config),
        },
        "runner_sha256": sha256_file(Path(__file__)),
        "code_commit": args.code_commit,
        "physical_gpu": _physical_gpu_identity(args.physical_gpu),
        "development_calibration_test_navdream_target_accessed": False,
        "weights_updated": False,
        "claim_boundary": (
            "Source-fit fold-0 outer-train read-only optimization diagnostic; not "
            "OOF, scene binding, architecture, appearance, behavior, or safety evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--mismatch-manifest", type=Path, required=True)
    parser.add_argument("--final-predictions", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    _write_json(arguments.output, result)
    print(json.dumps(result, sort_keys=True))
