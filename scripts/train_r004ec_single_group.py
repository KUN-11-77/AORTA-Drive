"""C1 audit: preregistered four-epoch overfit of one complete train group."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.models.quotient_field import QuotientFieldPredictor, quotient_gradient_loss
from aorta_drive.training.r004ec import (
    JoinedAnchor,
    deterministic_interleaved_epoch,
    load_joined_anchors,
    load_training_anchor,
    qualifies_single_group_overfit,
    sample_dense_features_torch,
)
from aorta_drive.vision.feature_cache import sha256_file


def _load_tensors(record: JoinedAnchor, device: torch.device) -> tuple[torch.Tensor, ...]:
    loaded = load_training_anchor(record)
    dense = torch.from_numpy(loaded.dense_features).to(device)
    patch_xy = torch.from_numpy(loaded.patch_xy).to(device)
    mask = torch.from_numpy(loaded.mask).to(device)
    features = sample_dense_features_torch(dense, patch_xy, mask).unsqueeze(0)
    labels = torch.from_numpy(loaded.label_m).to(device=device, dtype=torch.float32).unsqueeze(0)
    return features, mask.unsqueeze(0), labels


def _evaluate_group(
    model: QuotientFieldPredictor,
    records: Sequence[JoinedAnchor],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: list[float] = []
    hubers: list[float] = []
    tails: list[float] = []
    maxima: list[float] = []
    with torch.no_grad():
        for record in records:
            features, mask, labels = _load_tensors(record, device)
            prediction = model(features, mask)
            loss = quotient_gradient_loss(prediction, labels)
            totals.append(float(loss.total.item()))
            hubers.append(float(loss.mean_huber.item()))
            tails.append(float(loss.centered_smooth_max.item()))
            maxima.append(float(loss.maximum_residual.item()))
    return {
        "mean_objective": sum(totals) / len(totals),
        "mean_huber": sum(hubers) / len(hubers),
        "mean_centered_smooth_max": sum(tails) / len(tails),
        "exact_group_max_residual_m": max(maxima),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/r004ec_training.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage = config["stages"]["C1_single_group_overfit"]
    frozen = config["frozen_inputs"]
    if sha256_file(args.feature_manifest) != frozen["feature_manifest_sha256"]:
        raise ValueError("feature manifest hash disagrees with frozen C1 config")
    if sha256_file(args.label_manifest) != frozen["label_manifest_sha256"]:
        raise ValueError("label manifest hash disagrees with frozen C1 config")
    records = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
        allowed_splits=("train",),
    )
    first_group = records[0].recording_group
    selected = tuple(record for record in records if record.recording_group == first_group)
    if len(selected) != 16 or {record.anchor_index for record in selected} != set(range(16)):
        raise ValueError("C1 requires the complete lexicographically first train group")

    seed = int(config["stages"]["C2_full_registered_training"]["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = QuotientFieldPredictor().to(device)
    betas_raw = stage["betas"]
    if not isinstance(betas_raw, list) or len(betas_raw) != 2:
        raise ValueError("C1 AdamW betas must contain exactly two values")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage["learning_rate"]),
        betas=(float(betas_raw[0]), float(betas_raw[1])),
        weight_decay=float(stage["weight_decay"]),
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    initial = _evaluate_group(model, selected, device)
    epoch_ledger: list[dict[str, object]] = []
    epoch_metrics: list[dict[str, float]] = []
    step_ledger: list[dict[str, object]] = []
    all_finite = True
    maximum_epochs = int(stage["epochs"])
    for epoch in range(maximum_epochs):
        model.train()
        for step_in_epoch, record in enumerate(
            deterministic_interleaved_epoch(selected, seed=seed, epoch=epoch)
        ):
            features, mask, labels = _load_tensors(record, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, mask)
            loss = quotient_gradient_loss(prediction, labels)
            torch.autograd.backward(loss.total)
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            gradients_finite = bool(gradients) and all(
                bool(torch.isfinite(gradient).all().item()) for gradient in gradients
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(stage["gradient_clip_global_norm"])
            )
            step_finite = (
                bool(torch.isfinite(loss.total).item())
                and gradients_finite
                and math.isfinite(float(grad_norm.item()))
            )
            all_finite = all_finite and step_finite
            if not step_finite:
                raise FloatingPointError("nonfinite C1 loss or gradient")
            optimizer.step()
            step_ledger.append(
                {
                    "epoch": epoch + 1,
                    "step_in_epoch": step_in_epoch + 1,
                    "anchor_index": record.anchor_index,
                    "loss": float(loss.total.item()),
                    "preclip_gradient_norm": float(grad_norm.item()),
                }
            )
        metrics = _evaluate_group(model, selected, device)
        epoch_metrics.append(metrics)
        epoch_ledger.append({"epoch": epoch + 1, **metrics})
        print(json.dumps(epoch_ledger[-1], sort_keys=True), flush=True)

    final = epoch_metrics[-1]
    evaluated_maxima = tuple(
        epoch["exact_group_max_residual_m"] for epoch in epoch_metrics
    )
    mean_ratio = final["mean_objective"] / initial["mean_objective"]
    best_max_ratio = min(evaluated_maxima) / initial["exact_group_max_residual_m"]
    passed = all_finite and qualifies_single_group_overfit(
        initial_mean_objective=initial["mean_objective"],
        final_mean_objective=final["mean_objective"],
        initial_group_max_residual=initial["exact_group_max_residual_m"],
        evaluated_group_max_residuals=evaluated_maxima,
        maximum_mean_ratio=float(
            stage["acceptance"]["final_mean_objective_relative_to_initial_at_most"]
        ),
        maximum_best_max_ratio=float(
            stage["acceptance"][
                "best_exact_group_max_residual_relative_to_initial_at_most"
            ]
        ),
    )
    torch.cuda.synchronize(device)
    result = {
        "experiment_id": "R004E-C1-single-group-overfit",
        "status": "PASS" if passed else "FAIL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "label_manifest_sha256": sha256_file(args.label_manifest),
        "seed": seed,
        "recording_group": first_group,
        "independent_group_count": 1,
        "nested_anchor_count": 16,
        "epochs": maximum_epochs,
        "optimizer_updates": len(step_ledger),
        "initial": initial,
        "epoch_ledger": epoch_ledger,
        "final_mean_objective_ratio": mean_ratio,
        "best_exact_group_max_residual_ratio": best_max_ratio,
        "all_steps_and_gradients_finite": all_finite,
        "step_ledger": step_ledger,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_test_or_target_accessed": False,
        "checkpoint_reusable_for_C2": False,
        "claim_boundary": (
            "One-group optimization/data-signal sanity only. The group is intentionally "
            "reused for fitting and evaluation, so this establishes no generalization, "
            "checkpoint quality, calibration, certificate utility, OOD robustness, or Gate B pass."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
