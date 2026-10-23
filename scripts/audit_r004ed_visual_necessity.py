"""Compare the frozen visual checkpoint with zero and train-label-mean controls."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.models.quotient_field import QuotientFieldPredictor, oriented_grid_gradients
from aorta_drive.training.r004ec import (
    JoinedAnchor,
    load_joined_anchors,
    load_teacher_arrays,
    load_training_anchor,
    qualifies_visual_necessity,
    sample_dense_features_torch,
    summarize_group_maxima,
)
from aorta_drive.vision.feature_cache import sha256_file


def _group_metrics(group_maxima: dict[str, float]) -> dict[str, Any]:
    if len(group_maxima) != 24:
        raise ValueError("control evaluation must contain exactly 24 groups")
    worst5, all24 = summarize_group_maxima(tuple(group_maxima.values()))
    return {
        "development_group_max_residuals_m": dict(sorted(group_maxima.items())),
        "development_worst5_mean_m": worst5,
        "development_all24_mean_m": all24,
    }


def _evaluate_fixed_field(
    field: torch.Tensor, records: Sequence[JoinedAnchor]
) -> dict[str, Any]:
    predicted_edges = oriented_grid_gradients(field.unsqueeze(0)).squeeze(0)
    maxima: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        teacher = load_teacher_arrays(record)
        label = torch.from_numpy(teacher.label_m).to(dtype=torch.float32).unsqueeze(0)
        label_edges = oriented_grid_gradients(label).squeeze(0)
        residual = float(torch.max(torch.abs(predicted_edges - label_edges)).item())
        maxima[record.recording_group] = max(maxima[record.recording_group], residual)
        counts[record.recording_group] += 1
    if any(count != 16 for count in counts.values()):
        raise ValueError("every development group must contain 16 anchors")
    return _group_metrics(maxima)


def _evaluate_visual(
    model: QuotientFieldPredictor,
    records: Sequence[JoinedAnchor],
    device: torch.device,
) -> dict[str, Any]:
    maxima: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for record in records:
            loaded = load_training_anchor(record)
            dense = torch.from_numpy(loaded.dense_features).to(device)
            patch_xy = torch.from_numpy(loaded.patch_xy).to(device)
            mask = torch.from_numpy(loaded.mask).to(device)
            features = sample_dense_features_torch(dense, patch_xy, mask).unsqueeze(0)
            label = torch.from_numpy(loaded.label_m).to(
                device=device, dtype=torch.float32
            ).unsqueeze(0)
            prediction = model(features, mask.unsqueeze(0))
            residual = torch.abs(
                oriented_grid_gradients(prediction) - oriented_grid_gradients(label)
            ).max()
            scalar = float(residual.item())
            if not math.isfinite(scalar):
                raise FloatingPointError("nonfinite visual residual")
            maxima[record.recording_group] = max(maxima[record.recording_group], scalar)
            counts[record.recording_group] += 1
    if any(count != 16 for count in counts.values()):
        raise ValueError("every development group must contain 16 anchors")
    return _group_metrics(maxima)


def _checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return cast(dict[str, Any], payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--c2-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    hashes = {
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "label_manifest_sha256": sha256_file(args.label_manifest),
        "c2_result_sha256": sha256_file(args.c2_result),
        "c2_best_checkpoint_sha256": sha256_file(args.checkpoint),
    }
    for key, value in hashes.items():
        if value != frozen[key]:
            raise ValueError(f"{key} disagrees with frozen R004E-D config")
    records = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
    )
    train = tuple(record for record in records if record.split == "train")
    development = tuple(record for record in records if record.split == "development")
    if len(train) != 1056 or len(development) != 384:
        raise ValueError("R004E-D requires the complete frozen train/development sets")

    started = time.perf_counter()
    mean_field = np.zeros((64, 64), dtype=np.float64)
    for record in train:
        label = load_teacher_arrays(record).label_m
        mean_field += label - np.mean(label)
    mean_field /= len(train)
    mean_field -= np.mean(mean_field)
    zero_metrics = _evaluate_fixed_field(torch.zeros(64, 64), development)
    prior_metrics = _evaluate_fixed_field(
        torch.from_numpy(mean_field).to(dtype=torch.float32), development
    )

    checkpoint = _checkpoint(args.checkpoint)
    if checkpoint.get("completed_epoch") != int(frozen["visual_checkpoint_epoch"]):
        raise ValueError("checkpoint epoch disagrees with frozen R004E-D config")
    model = QuotientFieldPredictor()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model = model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    visual_metrics = _evaluate_visual(model, development, device)
    torch.cuda.synchronize(device)

    def pair(metrics: dict[str, Any]) -> tuple[float, float]:
        return (
            float(metrics["development_worst5_mean_m"]),
            float(metrics["development_all24_mean_m"]),
        )

    passed = qualifies_visual_necessity(
        pair(visual_metrics), pair(zero_metrics), pair(prior_metrics)
    )
    result = {
        "experiment_id": "R004E-D-development-visual-necessity",
        "status": "PASS" if passed else "FAIL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        **hashes,
        "train_group_count": len({record.recording_group for record in train}),
        "development_group_count": len(
            {record.recording_group for record in development}
        ),
        "systems": {
            "zero_field": zero_metrics,
            "train_label_mean_spatial_prior": prior_metrics,
            "frozen_visual_checkpoint": visual_metrics,
        },
        "strictly_better_than_both_controls_on_both_endpoints": passed,
        "mean_prior_zero_gauge_error_m": float(abs(np.mean(mean_field))),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_test_or_target_accessed": False,
        "claim_boundary": (
            "Development-only visual-necessity diagnostic. It does not establish "
            "held-out accuracy, calibration coverage, architecture superiority, utility, "
            "OOD robustness, or Gate B pass."
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
