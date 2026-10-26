#!/usr/bin/env python
"""One-anchor mechanical qualification of real LTF matrix-free JVPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aorta_drive.vision.ltf_causal_runtime import (
    STAGES,
    LTFCausalRuntime,
    activation_to_npc,
    build_frozen_ltf,
    controlled_sensor_config,
    load_ltf_bindings,
    make_features,
    npc_to_activation,
)
from aorta_drive.vision.planner_tangent import orthonormalize
from run_r006v0s_smoke import apply_condition, select_fit_anchors, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--planner-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sensors", type=Path, required=True)
    parser.add_argument("--group-roles", type=Path, required=True)
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--finite-difference-epsilon", type=float, default=1e-3)
    parser.add_argument("--relative-tolerance", type=float, default=1e-2)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("an explicitly selected CUDA device is required")

    bindings = load_ltf_bindings(args.planner_root)
    model, config = build_frozen_ltf(bindings, args.checkpoint, device)
    runtime = LTFCausalRuntime(model)
    builder = bindings.TransfuserFeatureBuilder(config)
    row = select_fit_anchors(args.group_roles, args.time_set, 1)[0]
    loader = bindings.SceneLoader(
        data_path=args.metadata,
        original_sensor_path=args.sensors,
        scene_filter=bindings.SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            frame_interval=1,
            has_route=True,
            log_names=[row["log_name"]],
            tokens=[row["scene_token"]],
        ),
        sensor_config=controlled_sensor_config(bindings),
    )
    agent_input = loader.get_agent_input_from_token(row["scene_token"])
    clean_features = make_features(builder, agent_input, config, device)
    shifted_features = {key: value.clone() for key, value in clean_features.items()}
    shifted_features["image"] = apply_condition(shifted_features["image"], "low_exposure")
    with torch.no_grad():
        _, clean_capture = runtime.forward(clean_features)
        _, shifted_capture = runtime.forward(shifted_features)

    reports = []
    all_pass = True
    epsilon = float(args.finite_difference_epsilon)
    for stage_index, stage in enumerate(STAGES):
        clean = clean_capture[stage].to(device)
        shifted = shifted_capture[stage].to(device)
        tangent = clean - shifted
        tangent_norm = float(torch.linalg.vector_norm(tangent))
        if tangent_norm == 0.0:
            raise RuntimeError(f"zero clean-shift tangent at {stage}")

        function = lambda value, stage=stage: runtime.trajectory_xy_from_stage(
            shifted_features, stage, value
        )
        _, product = torch.autograd.functional.jvp(
            function, shifted, tangent, create_graph=False, strict=True
        )
        with torch.no_grad():
            upper = function(shifted + epsilon * tangent)
            lower = function(shifted - epsilon * tangent)
            finite_difference = (upper - lower) / (2.0 * epsilon)
        absolute_error = float(torch.linalg.vector_norm(product - finite_difference))
        product_norm = float(torch.linalg.vector_norm(product))
        relative_error = absolute_error / max(product_norm, args.absolute_tolerance)

        npc_tangent = activation_to_npc(tangent)
        channels = npc_tangent.shape[-1]
        rank = min(4, channels)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + stage_index)
        basis = orthonormalize(
            torch.randn(channels, rank, generator=generator, device=device)
        ).requires_grad_(True)
        residual_npc = npc_tangent - (npc_tangent @ basis) @ basis.T
        residual = npc_to_activation(residual_npc, tangent)
        _, differentiable_product = torch.autograd.functional.jvp(
            function, shifted, residual, create_graph=True, strict=True
        )
        objective = torch.mean(differentiable_product * differentiable_product)
        gradient = torch.autograd.grad(objective, basis)[0]
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        finite = bool(
            torch.isfinite(product).all()
            and torch.isfinite(finite_difference).all()
            and torch.isfinite(gradient).all()
        )
        stage_pass = bool(
            finite
            and product_norm > args.absolute_tolerance
            and gradient_norm > 0.0
            and (
                relative_error <= args.relative_tolerance
                or absolute_error <= args.absolute_tolerance
            )
        )
        all_pass = all_pass and stage_pass
        reports.append(
            {
                "stage": stage,
                "activation_shape": list(shifted.shape),
                "tangent_norm": tangent_norm,
                "jvp_norm": product_norm,
                "finite_difference_absolute_error": absolute_error,
                "finite_difference_relative_error": relative_error,
                "basis_gradient_norm": gradient_norm,
                "finite": finite,
                "pass": stage_pass,
            }
        )

    result = {
        "artifact": "R006V0J_REAL_LTF_MATRIX_FREE_JVP_SMOKE_V1",
        "status": "PASS" if all_pass else "FAIL",
        "scientific_metrics_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
        "recording_group": row["recording_group"],
        "scene_token": row["scene_token"],
        "condition": "low_exposure",
        "finite_difference_epsilon": epsilon,
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "runner_sha256": sha256_file(Path(__file__)),
        "stages": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
