"""Run the frozen read-only R004H2D scale-interface conditioning diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.heterogeneous import deterministic_group_roles
from aorta_drive.models.heterogeneous_envelope import (
    BoundedEdgeScaleHead,
    bounded_laplace_scale_loss,
)
from aorta_drive.models.quotient_field import (
    QuotientFieldPredictor,
    oriented_grid_gradients,
)
from aorta_drive.training.r004ec import (
    JoinedAnchor,
    load_joined_anchors,
    load_training_anchor,
    sample_dense_features_torch,
)
from aorta_drive.training.r004h2d import (
    analytic_adamw_first_step,
    initial_scale_gradient,
    rms_only_normalize,
)
from aorta_drive.vision.feature_cache import sha256_file


def _checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return cast(dict[str, Any], payload)


def _read_roles(path: Path) -> dict[str, tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"recording_group", "role", "historical_exposure"}
    if not rows or set(rows[0]) != required:
        raise ValueError("role manifest columns disagree with the registered contract")
    roles = {
        role: tuple(row["recording_group"] for row in rows if row["role"] == role)
        for role in ("fit", "selection", "proxy_cal")
    }
    if tuple(len(roles[role]) for role in roles) != (42, 8, 16):
        raise ValueError("role counts disagree with the registered contract")
    return roles


def _source_train_groups(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = tuple(row["recording_group"] for row in rows if row["split"] == "train")
    if len(groups) != 66 or len(set(groups)) != 66:
        raise ValueError("source manifest must contain 66 unique train groups")
    return groups


def _load_tensors(record: JoinedAnchor, device: torch.device) -> tuple[torch.Tensor, ...]:
    loaded = load_training_anchor(record)
    dense = torch.from_numpy(loaded.dense_features).to(device)
    patch_xy = torch.from_numpy(loaded.patch_xy).to(device)
    mask = torch.from_numpy(loaded.mask).to(device)
    features = sample_dense_features_torch(dense, patch_xy, mask).unsqueeze(0)
    labels = torch.from_numpy(loaded.label_m).to(device=device, dtype=torch.float32).unsqueeze(0)
    return features, mask.unsqueeze(0), labels


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _feature_statistics(spatial: torch.Tensor) -> dict[str, float]:
    channel_rms = torch.sqrt(torch.mean(spatial.square(), dim=(0, 2, 3)))
    return {
        "mean": float(spatial.mean().item()),
        "standard_deviation": float(spatial.std(unbiased=False).item()),
        "rms": float(torch.sqrt(torch.mean(spatial.square())).item()),
        "maximum_absolute": float(torch.max(torch.abs(spatial)).item()),
        "minimum_channel_rms": float(channel_rms.min().item()),
        "median_channel_rms": float(channel_rms.median().item()),
        "maximum_channel_rms": float(channel_rms.max().item()),
    }


def _autograd_crosscheck(
    spatial: torch.Tensor,
    predicted_potential: torch.Tensor,
    labels: torch.Tensor,
    analytic_weight: torch.Tensor,
    analytic_bias: torch.Tensor,
) -> float:
    head = BoundedEdgeScaleHead().to(device=spatial.device, dtype=spatial.dtype)
    loss = bounded_laplace_scale_loss(
        predicted_potential,
        labels,
        head(spatial),
    ).total
    torch.autograd.backward(loss)
    projection_bias = head.projection.bias
    if (
        head.projection.weight.grad is None
        or projection_bias is None
        or projection_bias.grad is None
    ):
        raise RuntimeError("autograd cross-check did not produce both head gradients")
    actual_weight = head.projection.weight.grad[:, :, 0, 0]
    actual_bias = projection_bias.grad
    torch.testing.assert_close(actual_weight, analytic_weight, rtol=2.0e-5, atol=2.0e-3)
    torch.testing.assert_close(actual_bias, analytic_bias, rtol=2.0e-5, atol=2.0e-5)
    return max(
        float(torch.max(torch.abs(actual_weight - analytic_weight)).item()),
        float(torch.max(torch.abs(actual_bias - analytic_bias)).item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--recording-group-manifest", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--stage-p-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/r004h2d_conditioning.yaml")
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_hashes = {
        "recording_group_manifest_sha256": sha256_file(args.recording_group_manifest),
        "role_manifest_sha256": sha256_file(args.role_manifest),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "label_manifest_sha256": sha256_file(args.label_manifest),
    }
    for name, actual in expected_hashes.items():
        if actual != config["inputs"][name]:
            raise ValueError(f"{name} disagrees with the frozen H2D config")
    stage_p_hash = sha256_file(args.stage_p_checkpoint)
    if stage_p_hash != config["predecessor"]["selected_stage_p_checkpoint_sha256"]:
        raise ValueError("Stage-P checkpoint hash disagrees with the frozen H2D config")
    stage_p_payload = _checkpoint(args.stage_p_checkpoint)
    if stage_p_payload.get("config_sha256") != config["predecessor"]["h1_config_sha256"]:
        raise ValueError("Stage-P checkpoint provenance disagrees with H1")

    roles = _read_roles(args.role_manifest)
    regenerated = deterministic_group_roles(
        _source_train_groups(args.recording_group_manifest),
        namespace="AORTA-R004EG-roles-v1",
        fit_count=42,
        selection_count=8,
        proxy_cal_count=16,
    )
    if roles["fit"] != regenerated.fit or roles["selection"] != regenerated.selection:
        raise ValueError("fit/selection roles are not a deterministic replay")
    if roles["proxy_cal"] != regenerated.proxy_cal:
        raise ValueError("proxy roles are not a deterministic replay")

    records = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
        allowed_splits=("train",),
    )
    by_group: dict[str, list[JoinedAnchor]] = defaultdict(list)
    for record in records:
        by_group[record.recording_group].append(record)
    sample: list[tuple[str, JoinedAnchor]] = []
    for role in ("fit", "selection"):
        group = roles[role][0]
        ordered = sorted(by_group[group], key=lambda item: item.anchor_index)
        if len(ordered) != 16:
            raise ValueError("diagnostic group must contain exactly 16 anchors")
        sample.extend((role, record) for record in ordered[:4])
    if len(sample) != int(config["sample"]["total_anchors"]):
        raise ValueError("diagnostic sample size disagrees with registration")

    torch.manual_seed(20261024)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("R004H2D requires the registered CUDA device")
    torch.cuda.init()  # type: ignore[no-untyped-call]
    potential = QuotientFieldPredictor().to(device)
    potential.load_state_dict(stage_p_payload["model_state_dict"], strict=True)
    potential.eval()
    for parameter in potential.parameters():
        parameter.requires_grad_(False)

    first_step = config["first_step_counterfactual"]
    diagnosis = config["diagnosis"]
    ledgers: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for role, record in sample:
        features, mask, labels = _load_tensors(record, device)
        with torch.no_grad():
            spatial = potential.spatial_features(features, mask).detach()
            predicted = potential.potential_from_spatial_features(spatial).detach()
            residual = torch.abs(
                oriented_grid_gradients(predicted) - oriented_grid_gradients(labels)
            )
        normalized, feature_rms = rms_only_normalize(spatial)
        raw_gradient = initial_scale_gradient(spatial, residual)
        normalized_gradient = initial_scale_gradient(normalized, residual)
        crosscheck_error = _autograd_crosscheck(
            spatial,
            predicted,
            labels,
            raw_gradient.weight,
            raw_gradient.bias,
        )
        step_arguments = {
            "learning_rate": float(first_step["learning_rate"]),
            "weight_decay": float(first_step["weight_decay"]),
            "epsilon": float(first_step["epsilon"]),
            "gradient_clip_norm": float(first_step["global_gradient_clip_norm"]),
            "cap_position_tolerance": float(first_step["cap_position_tolerance"]),
            "target_maximum_logit_change": float(
                first_step["target_maximum_absolute_logit_change"]
            ),
        }
        raw_step = analytic_adamw_first_step(spatial, raw_gradient, **step_arguments)
        normalized_step = analytic_adamw_first_step(
            normalized, normalized_gradient, **step_arguments
        )
        raw_cap_fraction = raw_step.lower_cap_fraction + raw_step.upper_cap_fraction
        normalized_cap_fraction = (
            normalized_step.lower_cap_fraction + normalized_step.upper_cap_fraction
        )
        change_ratio = raw_step.maximum_absolute_logit_change / max(
            normalized_step.maximum_absolute_logit_change,
            torch.finfo(torch.float32).tiny,
        )
        activation_vote = (
            raw_cap_fraction
            >= float(
                diagnosis["activation_scale_vote_requires"][
                    "raw_first_step_cap_fraction_at_least"
                ]
            )
            and normalized_cap_fraction == 0.0
            and change_ratio
            >= float(
                diagnosis["activation_scale_vote_requires"][
                    "raw_to_normalized_max_logit_change_ratio_at_least"
                ]
            )
            and normalized_step.maximum_absolute_logit_change
            <= float(
                diagnosis["activation_scale_vote_requires"][
                    "normalized_maximum_absolute_logit_change_at_most"
                ]
            )
        )
        parameterization_vote = (
            raw_gradient.bounded_to_log_scale_derivative_ratio
            >= float(
                diagnosis["parameterization_vote_requires"][
                    "bounded_to_log_scale_initial_derivative_ratio_at_least"
                ]
            )
        )
        learning_rate_vote = (
            not activation_vote
            and raw_cap_fraction
            >= float(
                diagnosis["learning_rate_vote_requires"][
                    "raw_first_step_cap_fraction_at_least"
                ]
            )
            and raw_step.safe_learning_rate
            <= float(first_step["learning_rate"])
            * float(
                diagnosis["learning_rate_vote_requires"][
                    "derived_safe_learning_rate_at_most_fraction_of_registered"
                ]
            )
        )
        ledgers.append(
            {
                "role": role,
                "recording_group": record.recording_group,
                "anchor_index": record.anchor_index,
                "scene_token": record.scene_token,
                "feature_statistics": _feature_statistics(spatial),
                "rms_normalizer": feature_rms,
                "proper_loss_at_uniform_1m": float(torch.mean(residual).item()),
                "raw_gradient": {
                    "weight_norm": raw_gradient.weight_norm,
                    "bias_norm": raw_gradient.bias_norm,
                    "full_norm": raw_gradient.full_norm,
                },
                "normalized_gradient": {
                    "weight_norm": normalized_gradient.weight_norm,
                    "bias_norm": normalized_gradient.bias_norm,
                    "full_norm": normalized_gradient.full_norm,
                },
                "bounded_to_log_scale_initial_derivative_ratio": (
                    raw_gradient.bounded_to_log_scale_derivative_ratio
                ),
                "autograd_maximum_absolute_error": crosscheck_error,
                "raw_first_step": raw_step.__dict__,
                "normalized_first_step": normalized_step.__dict__,
                "raw_to_normalized_max_logit_change_ratio": change_ratio,
                "votes": {
                    "activation_scale": activation_vote,
                    "parameterization": parameterization_vote,
                    "learning_rate": learning_rate_vote,
                },
            }
        )

    vote_counts = {
        name: sum(bool(row["votes"][name]) for row in ledgers)  # type: ignore[index]
        for name in ("activation_scale", "parameterization", "learning_rate")
    }
    support = int(diagnosis["minimum_supporting_anchors"])
    if vote_counts["parameterization"] >= support:
        conclusion = "PARAMETERIZATION_DOMINANT"
    elif vote_counts["activation_scale"] >= support:
        conclusion = "TRUNK_ACTIVATION_SCALE_DOMINANT"
    elif vote_counts["learning_rate"] >= support:
        conclusion = "REGISTERED_LEARNING_RATE_DOMINANT"
    else:
        conclusion = "UNRESOLVED"
    raw_gradients = [float(row["raw_gradient"]["full_norm"]) for row in ledgers]  # type: ignore[index]
    normalized_gradients = [
        float(row["normalized_gradient"]["full_norm"]) for row in ledgers  # type: ignore[index]
    ]
    elapsed_seconds = time.perf_counter() - started
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "stage_p_checkpoint_sha256": stage_p_hash,
        **expected_hashes,
        "sample_size": len(ledgers),
        "sample_roles": {"fit": 4, "selection": 4},
        "vote_counts": vote_counts,
        "minimum_supporting_anchors": support,
        "conclusion": conclusion,
        "summary": {
            "median_raw_full_gradient_norm": median(raw_gradients),
            "median_normalized_full_gradient_norm": median(normalized_gradients),
            "median_raw_to_normalized_gradient_norm_ratio": median(
                raw / normalized
                for raw, normalized in zip(raw_gradients, normalized_gradients, strict=True)
            ),
            "median_raw_safe_learning_rate": median(
                float(row["raw_first_step"]["safe_learning_rate"])  # type: ignore[index]
                for row in ledgers
            ),
        },
        "anchor_ledger": ledgers,
        "optimizer_constructed": False,
        "parameter_mutation": False,
        "checkpoint_written": False,
        "prohibited_payload_accessed": False,
        "h2_revision_registration_authorized": conclusion != "UNRESOLVED",
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": peak_cuda_bytes,
        "elapsed_seconds": elapsed_seconds,
        "claim_boundary": config["claim_boundary"],
    }
    output_path = args.output_directory / "result.json"
    _write_json(output_path, result)
    elapsed = elapsed_seconds
    peak_gib = peak_cuda_bytes / 1024**3
    output_gib = _directory_bytes(args.output_directory) / 1024**3
    if elapsed > float(config["resources"]["maximum_wall_minutes"]) * 60.0:
        raise RuntimeError("H2D wall time exceeded its registered cap")
    if peak_gib > float(config["resources"]["maximum_peak_gpu_gib"]):
        raise RuntimeError("H2D peak GPU allocation exceeded its registered cap")
    if output_gib > float(config["resources"]["maximum_output_gib"]):
        raise RuntimeError("H2D output size exceeded its registered cap")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
