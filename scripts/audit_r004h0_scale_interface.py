"""Audit the registered R004H0 scale-only structural interface on one fit anchor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.heterogeneous import deterministic_group_roles
from aorta_drive.models.heterogeneous_envelope import (
    HeterogeneousEnvelopePredictor,
    bounded_laplace_scale_loss,
)
from aorta_drive.models.quotient_field import QuotientFieldPredictor
from aorta_drive.training.r004ec import (
    load_joined_anchors,
    load_training_anchor,
    sample_dense_features_torch,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return cast(dict[str, Any], payload)


def _read_roles(path: Path) -> tuple[list[str], list[str], list[str], str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"recording_group", "role", "historical_exposure"}
    if not rows or set(rows[0]) != required:
        raise ValueError("role manifest columns disagree with the registered contract")
    if len(rows) != 66 or len({row["recording_group"] for row in rows}) != 66:
        raise ValueError("role manifest must contain 66 unique groups")
    exposure_values = {row["historical_exposure"] for row in rows}
    if len(exposure_values) != 1:
        raise ValueError("every role must carry the same explicit historical exposure")
    by_role = {
        role: [row["recording_group"] for row in rows if row["role"] == role]
        for role in ("fit", "selection", "proxy_cal")
    }
    if any(row["role"] not in by_role for row in rows):
        raise ValueError("role manifest contains an unknown role")
    return (
        by_role["fit"],
        by_role["selection"],
        by_role["proxy_cal"],
        next(iter(exposure_values)),
    )


def _read_source_train_groups(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = [row["recording_group"] for row in rows if row["split"] == "train"]
    if len(groups) != 66 or len(set(groups)) != 66:
        raise ValueError("source manifest must contain 66 unique train groups")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recording-group-manifest", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    hashes = {
        "recording_group_manifest_sha256": _sha256(args.recording_group_manifest),
        "role_manifest_sha256": _sha256(args.role_manifest),
        "feature_manifest_sha256": _sha256(args.feature_manifest),
        "label_manifest_sha256": _sha256(args.label_manifest),
        "structural_checkpoint_sha256": _sha256(args.checkpoint),
    }
    for key, value in hashes.items():
        if value != frozen[key]:
            raise ValueError(f"{key} disagrees with the registered config")

    fit, selection, proxy_cal, exposure = _read_roles(args.role_manifest)
    role_config = config["roles"]
    expected_counts = role_config["counts"]
    if (len(fit), len(selection), len(proxy_cal)) != (
        int(expected_counts["fit"]),
        int(expected_counts["selection"]),
        int(expected_counts["proxy_cal"]),
    ):
        raise ValueError("role counts disagree with the registered config")
    if exposure != role_config["historical_exposure_value"]:
        raise ValueError("historical exposure label disagrees with registration")
    regenerated = deterministic_group_roles(
        _read_source_train_groups(args.recording_group_manifest),
        namespace=str(role_config["namespace"]),
        fit_count=len(fit),
        selection_count=len(selection),
        proxy_cal_count=len(proxy_cal),
    )
    if tuple(fit) != regenerated.fit:
        raise ValueError("fit role order is not a deterministic replay")
    if tuple(selection) != regenerated.selection:
        raise ValueError("selection role order is not a deterministic replay")
    if tuple(proxy_cal) != regenerated.proxy_cal:
        raise ValueError("proxy-cal role order is not a deterministic replay")

    records = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
        allowed_splits=("train",),
    )
    fit_group = fit[0]
    candidates = [record for record in records if record.recording_group == fit_group]
    if len(candidates) != 16:
        raise ValueError("registered first fit group must contain exactly 16 anchors")
    record = min(candidates, key=lambda item: item.anchor_index)
    loaded = load_training_anchor(record)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("R004H0 requires the registered CUDA device")
    # PyTorch 2.7 may reject peak-memory reset before the selected CUDA context
    # has been initialized.  Initialize the runtime before starting the measured
    # region; this does not load a payload or execute the model.
    torch.cuda.init()  # type: ignore[no-untyped-call]
    torch.cuda.reset_peak_memory_stats(device)
    potential_predictor = QuotientFieldPredictor().to(device)
    checkpoint = _checkpoint(args.checkpoint)
    potential_predictor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    potential_predictor.eval()
    model = HeterogeneousEnvelopePredictor(potential_predictor).to(device).train()
    architecture = config["architecture"]
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != int(architecture["scale_head_trainable_parameter_count"]):
        raise ValueError("trainable parameter count disagrees with registration")

    dense = torch.from_numpy(loaded.dense_features).to(device)
    patch_xy = torch.from_numpy(loaded.patch_xy).to(device)
    mask = torch.from_numpy(loaded.mask).to(device)
    features = sample_dense_features_torch(dense, patch_xy, mask).unsqueeze(0)
    mask_batch = mask.unsqueeze(0)
    label = torch.from_numpy(loaded.label_m).to(device=device, dtype=torch.float32).unsqueeze(0)

    started = time.perf_counter()
    spatial = potential_predictor.spatial_features(features, mask_batch)
    direct_potential = potential_predictor.potential_from_spatial_features(spatial)
    forward_potential = potential_predictor(features, mask_batch)
    replay_max_abs = float(torch.max(torch.abs(direct_potential - forward_potential)).item())
    if replay_max_abs > float(architecture["potential_refactor_replay_max_abs_m"]):
        raise AssertionError("potential refactor changed represented output")

    output = model(features, mask_batch)
    if not torch.equal(output.potential, forward_potential):
        raise AssertionError("wrapper changed the frozen potential output")
    potential_gauge_abs = float(torch.abs(output.potential.mean()).item())
    if potential_gauge_abs > float(architecture["potential_gauge_abs_tolerance_m"]):
        raise AssertionError("potential gauge exceeded the registered tolerance")
    expected_edges = int(architecture["edge_count"])
    if output.edge_scales_m.shape != (1, expected_edges):
        raise AssertionError("scale head produced the wrong edge shape/order")
    if not torch.isfinite(output.edge_scales_m).all():
        raise AssertionError("scale head produced nonfinite values")
    scale_min = float(architecture["scale_min_m"])
    scale_max = float(architecture["scale_max_m"])
    if torch.any(output.edge_scales_m < scale_min) or torch.any(
        output.edge_scales_m > scale_max
    ):
        raise AssertionError("scale head violated registered bounds")
    initial_error = float(
        torch.max(
            torch.abs(output.edge_scales_m - float(architecture["initial_scale_m"]))
        ).item()
    )
    if initial_error > float(architecture["initial_scale_max_abs_tolerance_m"]):
        raise AssertionError("scale initialization exceeded registered tolerance")

    loss = bounded_laplace_scale_loss(output.potential, label, output.edge_scales_m)
    loss.total.backward()  # type: ignore[no-untyped-call]
    potential_gradients = [
        parameter.grad for parameter in model.potential_predictor.parameters()
    ]
    if any(gradient is not None for gradient in potential_gradients):
        raise AssertionError("a frozen potential parameter received a gradient")
    scale_gradients = [parameter.grad for parameter in model.scale_head.parameters()]
    if any(gradient is None for gradient in scale_gradients):
        raise AssertionError("a scale-head parameter did not receive a gradient")
    finite_scale_gradients = [
        gradient for gradient in scale_gradients if gradient is not None
    ]
    if not all(torch.isfinite(gradient).all() for gradient in finite_scale_gradients):
        raise AssertionError("a scale-head gradient was nonfinite")
    gradient_nonzero = sum(
        int(torch.count_nonzero(gradient).item()) for gradient in finite_scale_gradients
    )
    if gradient_nonzero == 0:
        raise AssertionError("all scale-head gradients were zero")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    if peak_gib > float(config["resources"]["maximum_peak_gpu_gib"]):
        raise AssertionError("peak GPU allocation exceeded the registered limit")
    if elapsed > float(config["resources"]["maximum_wall_minutes"]) * 60.0:
        raise AssertionError("wall time exceeded the registered limit")

    signature = tuple(inspect.signature(HeterogeneousEnvelopePredictor.forward).parameters)
    if signature != ("self", "node_slot_features", "node_slot_mask"):
        raise AssertionError("wrapper forward exposes an unregistered input")

    result = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": _sha256(args.config),
        **hashes,
        "role_counts": {
            "fit": len(fit),
            "selection": len(selection),
            "proxy_cal": len(proxy_cal),
        },
        "historical_exposure": exposure,
        "anchor": {
            "recording_group": record.recording_group,
            "anchor_index": record.anchor_index,
            "scene_token": record.scene_token,
            "role": "fit",
        },
        "checks": {
            "strict_old_state_dict_load": True,
            "potential_replay_max_abs_m": replay_max_abs,
            "potential_wrapper_exact_replay": True,
            "scale_trainable_parameter_count": trainable_parameters,
            "edge_count": int(output.edge_scales_m.numel()),
            "initial_scale_max_abs_error_m": initial_error,
            "potential_gradient_count": 0,
            "nonzero_scale_gradient_elements": gradient_nonzero,
            "candidate_or_planner_forward_inputs": False,
            "optimizer_step_count": 0,
            "loss_total": float(loss.total.detach().item()),
            "loss_scaled_absolute_residual": float(
                loss.mean_scaled_absolute_residual.detach().item()
            ),
            "loss_mean_log_scale": float(loss.mean_log_scale.detach().item()),
            "potential_gauge_abs_m": potential_gauge_abs,
            "peak_gpu_gib": peak_gib,
            "wall_seconds": elapsed,
        },
        "claims": {
            "r004h1_registration_authorized": True,
            "learned_uncertainty_established": False,
            "coverage_established": False,
            "gate_b_passed": False,
            "official_calibration_access_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
