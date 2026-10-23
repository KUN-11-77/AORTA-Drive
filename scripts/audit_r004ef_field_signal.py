"""Distinguish useful average visual signal from tail failure or sign inversion."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.models.quotient_field import QuotientFieldPredictor, oriented_grid_gradients
from aorta_drive.training.r004ec import (
    load_joined_anchors,
    load_training_anchor,
    sample_dense_features_torch,
    summarize_group_maxima,
)
from aorta_drive.training.r004ef import decide_field_signal
from aorta_drive.vision.feature_cache import sha256_file


def _checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return cast(dict[str, Any], payload)


def _system_summary(
    group_means: dict[str, float], group_maxima: dict[str, float]
) -> dict[str, Any]:
    if len(group_means) != 24 or set(group_means) != set(group_maxima):
        raise ValueError("every system must contain the same 24 recording groups")
    worst5, all24 = summarize_group_maxima(tuple(group_maxima.values()))
    mean_values = tuple(group_means.values())
    return {
        "mean_of_group_mean_edge_mae_m": sum(mean_values) / 24,
        "worst5_group_maximum_mean_m": worst5,
        "all24_group_maximum_mean_m": all24,
        "group_mean_edge_mae_m": dict(sorted(group_means.items())),
        "group_maximum_edge_error_m": dict(sorted(group_maxima.items())),
    }


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
    support_fraction = float(config["decision"]["group_support_fraction"])
    high_gradient_threshold = float(config["diagnostics"]["high_label_gradient_m"])
    hashes = {
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "label_manifest_sha256": sha256_file(args.label_manifest),
        "c2_result_sha256": sha256_file(args.c2_result),
        "c2_best_checkpoint_sha256": sha256_file(args.checkpoint),
    }
    for key, hash_value in hashes.items():
        if hash_value != frozen[key]:
            raise ValueError(f"{key} disagrees with frozen R004E-F config")
    records = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
        allowed_splits=("development",),
    )
    if len(records) != 384 or len({record.recording_group for record in records}) != 24:
        raise ValueError("R004E-F requires exactly 24 complete development groups")

    checkpoint = _checkpoint(args.checkpoint)
    if checkpoint.get("completed_epoch") != int(frozen["visual_checkpoint_epoch"]):
        raise ValueError("checkpoint epoch disagrees with frozen R004E-F config")
    device = torch.device(args.device)
    model = QuotientFieldPredictor().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    names = ("zero_field", "visual", "sign_flipped_visual")
    residual_sums: dict[str, dict[str, float]] = {
        name: defaultdict(float) for name in names
    }
    residual_maxima: dict[str, dict[str, float]] = {
        name: defaultdict(float) for name in names
    }
    edge_counts: Counter[str] = Counter()
    dot_sums: defaultdict[str, float] = defaultdict(float)
    pred_square_sums: defaultdict[str, float] = defaultdict(float)
    label_square_sums: defaultdict[str, float] = defaultdict(float)
    high_gradient_correct: Counter[str] = Counter()
    high_gradient_counts: Counter[str] = Counter()
    anchor_counts: Counter[str] = Counter()

    with torch.no_grad():
        for record in records:
            loaded = load_training_anchor(record)
            dense = torch.from_numpy(loaded.dense_features).to(device)
            patch_xy = torch.from_numpy(loaded.patch_xy).to(device)
            mask = torch.from_numpy(loaded.mask).to(device)
            features = sample_dense_features_torch(dense, patch_xy, mask).unsqueeze(0)
            prediction = model(features, mask.unsqueeze(0))
            predicted_edges = oriented_grid_gradients(prediction).squeeze(0)
            label = torch.from_numpy(loaded.label_m).to(
                device=device, dtype=torch.float32
            ).unsqueeze(0)
            label_edges = oriented_grid_gradients(label).squeeze(0)
            residuals = {
                "zero_field": torch.abs(label_edges),
                "visual": torch.abs(predicted_edges - label_edges),
                "sign_flipped_visual": torch.abs(-predicted_edges - label_edges),
            }
            group = record.recording_group
            anchor_counts[group] += 1
            edge_counts[group] += int(label_edges.numel())
            for name, residual in residuals.items():
                residual_sums[name][group] += float(residual.sum().item())
                residual_maxima[name][group] = max(
                    residual_maxima[name][group], float(residual.max().item())
                )
            dot_sums[group] += float(torch.dot(predicted_edges, label_edges).item())
            pred_square_sums[group] += float(torch.dot(predicted_edges, predicted_edges).item())
            label_square_sums[group] += float(torch.dot(label_edges, label_edges).item())
            high = torch.abs(label_edges) >= high_gradient_threshold
            high_gradient_counts[group] += int(high.sum().item())
            high_gradient_correct[group] += int(
                ((predicted_edges * label_edges > 0.0) & high).sum().item()
            )
    torch.cuda.synchronize(device)
    if len(anchor_counts) != 24 or set(anchor_counts.values()) != {16}:
        raise ValueError("every development group must contain exactly 16 anchors")

    group_means = {
        name: {
            group: residual_sums[name][group] / edge_counts[group]
            for group in sorted(edge_counts)
        }
        for name in names
    }
    cosine = {
        group: dot_sums[group]
        / math.sqrt(pred_square_sums[group] * label_square_sums[group])
        for group in sorted(edge_counts)
    }
    high_gradient_sign_accuracy = {
        group: high_gradient_correct[group] / high_gradient_counts[group]
        for group in sorted(edge_counts)
    }
    group_events: dict[str, dict[str, bool]] = {}
    event_counts: Counter[str] = Counter()
    for group in sorted(edge_counts):
        visual_mean = group_means["visual"][group]
        zero_mean = group_means["zero_field"][group]
        flipped_mean = group_means["sign_flipped_visual"][group]
        visual_max = residual_maxima["visual"][group]
        zero_max = residual_maxima["zero_field"][group]
        flipped_max = residual_maxima["sign_flipped_visual"][group]
        events = {
            "tail_objective_mismatch": (
                visual_mean < zero_mean and visual_max > zero_max and cosine[group] > 0.0
            ),
            "global_sign_inversion": (
                flipped_mean < visual_mean
                and flipped_mean < zero_mean
                and flipped_max < visual_max
                and flipped_max < zero_max
                and cosine[group] < 0.0
            ),
            "no_transferable_signal": (
                visual_mean >= zero_mean and flipped_mean >= zero_mean
            ),
        }
        group_events[group] = events
        event_counts.update(name for name, occurred in events.items() if occurred)

    registered_counts = {
        name: int(event_counts[name])
        for name in (
            "tail_objective_mismatch",
            "global_sign_inversion",
            "no_transferable_signal",
        )
    }
    decision = decide_field_signal(
        registered_counts, group_count=24, support_fraction=support_fraction
    )
    result = {
        "experiment_id": "R004E-F-field-signal-versus-tail",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        **hashes,
        "development_group_count": 24,
        "anchors_per_group": 16,
        "systems": {
            name: _system_summary(group_means[name], residual_maxima[name])
            for name in names
        },
        "group_cosine_similarity": cosine,
        "high_label_gradient_threshold_m": high_gradient_threshold,
        "group_high_gradient_sign_accuracy": high_gradient_sign_accuracy,
        "group_events": group_events,
        "event_counts": registered_counts,
        "support_rule": {
            "fraction": support_fraction,
            "minimum_groups": decision.minimum_support_groups,
            "unit": "development_recording_group",
            "precedence": [
                "global_sign_inversion",
                "tail_objective_mismatch",
                "no_transferable_visual_signal",
                "mixed_or_partial_signal",
            ],
        },
        "hypothesis_support": {
            "tail_objective_mismatch": decision.tail_objective_mismatch_supported,
            "global_sign_inversion": decision.global_sign_inversion_supported,
            "no_transferable_signal": decision.no_transferable_signal_supported,
        },
        "verdict": decision.verdict,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_test_or_target_accessed": False,
        "claim_boundary": (
            "Frozen-checkpoint development diagnostic. A supported hypothesis guides "
            "the next registered redesign but is not causal proof, held-out evidence, "
            "calibration coverage, architecture superiority, utility, or Gate B pass."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
