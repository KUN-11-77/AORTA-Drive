"""Localize why the frozen R004E-C2 visual field loses to the zero field."""

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
)
from aorta_drive.training.r004ee import (
    EDGE_BOTH_OBSERVED,
    EDGE_CATEGORY_NAMES,
    decide_attribution,
    oriented_edge_endpoint_sums,
    oriented_edge_evidence_categories,
)
from aorta_drive.vision.feature_cache import sha256_file


def _checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return cast(dict[str, Any], payload)


def _complete_group_summary(values: dict[str, float]) -> dict[str, Any]:
    if len(values) != 24:
        return {
            "group_count": len(values),
            "worst5_mean_m": None,
            "all24_mean_m": None,
            "group_maxima_m": dict(sorted(values.items())),
        }
    ordered = sorted(values.values(), reverse=True)
    return {
        "group_count": 24,
        "worst5_mean_m": sum(ordered[:5]) / 5,
        "all24_mean_m": sum(ordered) / 24,
        "group_maxima_m": dict(sorted(values.items())),
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
    diagnostic = config["diagnostic"]
    tolerance = float(diagnostic["numeric_tolerance_m"])
    edge_length = float(diagnostic["edge_length_m"])
    support_fraction = float(diagnostic["group_support_fraction"])
    hashes = {
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "label_manifest_sha256": sha256_file(args.label_manifest),
        "c2_result_sha256": sha256_file(args.c2_result),
        "c2_best_checkpoint_sha256": sha256_file(args.checkpoint),
    }
    for key, hash_value in hashes.items():
        if hash_value != frozen[key]:
            raise ValueError(f"{key} disagrees with frozen R004E-E config")

    records = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
        allowed_splits=("development",),
    )
    if len(records) != 384 or len({record.recording_group for record in records}) != 24:
        raise ValueError("R004E-E requires exactly 24 complete development groups")

    checkpoint = _checkpoint(args.checkpoint)
    if checkpoint.get("completed_epoch") != int(frozen["visual_checkpoint_epoch"]):
        raise ValueError("checkpoint epoch disagrees with frozen R004E-E config")
    model = QuotientFieldPredictor()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    anchor_counts: Counter[str] = Counter()
    group_best: dict[str, dict[str, Any]] = {}
    category_group_maxima: dict[str, dict[str, float]] = {
        name: {} for name in EDGE_CATEGORY_NAMES
    }
    observed_nonovershoot_group_maxima: dict[str, float] = {}
    category_edge_counts: Counter[str] = Counter()
    category_residual_above_certified_zero_counts: Counter[str] = Counter()
    predicted_overshoot_edge_count = 0
    total_edge_count = 0
    label_bound_violation_count = 0
    label_max_bound_excess_m = -math.inf

    with torch.no_grad():
        for record in records:
            loaded = load_training_anchor(record)
            dense = torch.from_numpy(loaded.dense_features).to(device)
            patch_xy = torch.from_numpy(loaded.patch_xy).to(device)
            mask = torch.from_numpy(loaded.mask).to(device)
            features = sample_dense_features_torch(dense, patch_xy, mask).unsqueeze(0)
            prediction = model(features, mask.unsqueeze(0))
            predicted_edges = oriented_grid_gradients(prediction).squeeze(0).cpu()

            label64 = torch.from_numpy(loaded.label_m).unsqueeze(0)
            label_edges64 = oriented_grid_gradients(label64).squeeze(0)
            label_edges = label_edges64.to(torch.float32)
            residual = torch.abs(predicted_edges - label_edges)
            evidence = torch.from_numpy(loaded.mask.any(axis=1).reshape(64, 64))
            categories = oriented_edge_evidence_categories(evidence)
            error_sums = oriented_edge_endpoint_sums(
                torch.from_numpy(loaded.total_error_m)
            )
            label_bound = edge_length + error_sums
            label_excess = torch.abs(label_edges64) - label_bound
            label_bound_violation_count += int((label_excess > tolerance).sum().item())
            label_max_bound_excess_m = max(
                label_max_bound_excess_m, float(label_excess.max().item())
            )

            overshoot = torch.abs(predicted_edges) > edge_length + tolerance
            above_certified_zero = residual.to(torch.float64) > label_bound + tolerance
            predicted_overshoot_edge_count += int(overshoot.sum().item())
            total_edge_count += int(residual.numel())
            anchor_counts[record.recording_group] += 1

            for category_id, category_name in enumerate(EDGE_CATEGORY_NAMES):
                selected = categories == category_id
                selected_count = int(selected.sum().item())
                category_edge_counts[category_name] += selected_count
                category_residual_above_certified_zero_counts[category_name] += int(
                    (above_certified_zero & selected).sum().item()
                )
                if selected_count:
                    category_maximum = float(residual[selected].max().item())
                    previous = category_group_maxima[category_name].get(record.recording_group)
                    if previous is None or category_maximum > previous:
                        category_group_maxima[category_name][record.recording_group] = (
                            category_maximum
                        )

            observed_nonovershoot = (categories == EDGE_BOTH_OBSERVED) & ~overshoot
            if bool(observed_nonovershoot.any().item()):
                observed_maximum = float(residual[observed_nonovershoot].max().item())
                previous = observed_nonovershoot_group_maxima.get(record.recording_group)
                if previous is None or observed_maximum > previous:
                    observed_nonovershoot_group_maxima[record.recording_group] = (
                        observed_maximum
                    )

            maximum_value, maximum_index_tensor = torch.max(residual, dim=0)
            maximum_index = int(maximum_index_tensor.item())
            scalar = float(maximum_value.item())
            previous_best = group_best.get(record.recording_group)
            if previous_best is None or scalar > float(previous_best["residual_m"]):
                category_id = int(categories[maximum_index].item())
                group_best[record.recording_group] = {
                    "residual_m": scalar,
                    "anchor_index": record.anchor_index,
                    "log_name": record.log_name,
                    "scene_token": record.scene_token,
                    "edge_index_right_then_down": maximum_index,
                    "edge_category": EDGE_CATEGORY_NAMES[category_id],
                    "predicted_gradient_m": float(predicted_edges[maximum_index].item()),
                    "label_gradient_m": float(label_edges64[maximum_index].item()),
                    "teacher_endpoint_error_sum_m": float(error_sums[maximum_index].item()),
                    "predicted_gradient_overshoot": bool(overshoot[maximum_index].item()),
                    "missing_evidence_incident": category_id != EDGE_BOTH_OBSERVED,
                    "fully_observed_nonovershoot": (
                        category_id == EDGE_BOTH_OBSERVED
                        and not bool(overshoot[maximum_index].item())
                    ),
                }

    torch.cuda.synchronize(device)
    if set(anchor_counts.values()) != {16} or len(anchor_counts) != 24:
        raise ValueError("every development group must contain exactly 16 anchors")
    if label_bound_violation_count:
        raise AssertionError("teacher gradients violate the registered Lipschitz ledger")

    attribution_counts = {
        "gradient_overshoot": sum(
            bool(item["predicted_gradient_overshoot"]) for item in group_best.values()
        ),
        "missing_evidence_incident": sum(
            bool(item["missing_evidence_incident"]) for item in group_best.values()
        ),
        "fully_observed_nonovershoot": sum(
            bool(item["fully_observed_nonovershoot"]) for item in group_best.values()
        ),
    }
    decision = decide_attribution(
        attribution_counts,
        group_count=24,
        support_fraction=support_fraction,
    )
    result = {
        "experiment_id": "R004E-E-excess-residual-localization",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        **hashes,
        "development_group_count": 24,
        "anchors_per_group": 16,
        "edge_length_m": edge_length,
        "numeric_tolerance_m": tolerance,
        "total_edge_count": total_edge_count,
        "predicted_overshoot_edge_count": predicted_overshoot_edge_count,
        "predicted_overshoot_edge_fraction": predicted_overshoot_edge_count / total_edge_count,
        "edge_category_counts": dict(category_edge_counts),
        "residual_above_certified_zero_counts_by_category": dict(
            category_residual_above_certified_zero_counts
        ),
        "teacher_lipschitz_ledger": {
            "violation_count": label_bound_violation_count,
            "maximum_excess_over_edge_length_plus_endpoint_errors_m": (
                label_max_bound_excess_m
            ),
        },
        "group_global_maxima": dict(sorted(group_best.items())),
        "group_maximum_attribution_counts": attribution_counts,
        "support_rule": {
            "fraction": support_fraction,
            "minimum_groups": decision.minimum_support_groups,
            "unit": "development_recording_group",
            "tie_rule": "earliest_anchor_then_right_then_down_edge",
        },
        "hypothesis_support": {
            "gradient_overshoot": decision.gradient_overshoot_supported,
            "missing_evidence": decision.missing_evidence_supported,
            "fully_observed_nonovershoot": decision.observed_nonovershoot_supported,
        },
        "recommended_next_control": decision.recommendation,
        "category_group_maxima": {
            name: _complete_group_summary(values)
            for name, values in category_group_maxima.items()
        },
        "fully_observed_nonovershoot_group_maxima": _complete_group_summary(
            observed_nonovershoot_group_maxima
        ),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_test_or_target_accessed": False,
        "claim_boundary": (
            "Frozen-checkpoint development diagnostic. Group-level attribution supports "
            "a next falsification control but is not causal proof, held-out evidence, "
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
