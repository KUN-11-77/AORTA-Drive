#!/usr/bin/env python
"""One-anchor mechanical qualification of real LTF matrix-free JVPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aorta_drive.vision.ltf_causal_runtime import (
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

QUALIFIED_STAGES = ("c3_planner_memory", "c4_trajectory_query")


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
    parser.add_argument(
        "--stages", nargs="+", choices=QUALIFIED_STAGES, default=list(QUALIFIED_STAGES)
    )
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--finite-difference-epsilon", type=float, default=1e-3)
    parser.add_argument("--relative-tolerance", type=float, default=1e-2)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-5)
    parser.add_argument("--adjoint-relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--adjoint-absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--backend-output-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # CUDA adaptive_avg_pool2d backward has no officially deterministic kernel
    # in this PyTorch build.  Qualification therefore records warn-only mode
    # and requires an exact same-GPU replay before any formal science run.
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Fused/memory-efficient SDPA in this build has no forward-AD rule.  The
    # mathematically equivalent math backend does and is therefore frozen for
    # every primal, JVP, finite-difference, and replay evaluation.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)
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
        torch.backends.mha.set_fastpath_enabled(True)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            default_clean, _ = runtime.forward(clean_features)
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            math_clean, clean_capture = runtime.forward(clean_features)
            _, shifted_capture = runtime.forward(shifted_features)
    backend_output_max_abs_difference = float(
        torch.max(torch.abs(default_clean - math_clean))
    )
    backend_semantics_pass = bool(
        backend_output_max_abs_difference <= args.backend_output_tolerance
    )
    with torch.no_grad():
        autodiff_head_clean_xy = runtime.trajectory_xy_from_stage(
            clean_features,
            "c4_trajectory_query",
            clean_capture["c4_trajectory_query"].to(device),
        )
    official_clean_xy = math_clean[..., :2].reshape(math_clean.shape[0], -1)
    autodiff_head_max_abs_difference = float(
        torch.max(torch.abs(autodiff_head_clean_xy - official_clean_xy))
    )
    autodiff_head_semantics_pass = bool(
        autodiff_head_max_abs_difference <= args.backend_output_tolerance
    )
    direct_tail_differences = {}
    for qualified_stage in ("c3_planner_memory", "c4_trajectory_query"):
        with torch.no_grad():
            direct = runtime.trajectory_xy_from_qualified_stage(
                qualified_stage, clean_capture[qualified_stage].to(device)
            )
        direct_tail_differences[qualified_stage] = float(
            torch.max(torch.abs(direct - official_clean_xy))
        )
    direct_tail_semantics_pass = all(
        value <= args.backend_output_tolerance
        for value in direct_tail_differences.values()
    )

    reports = []
    all_pass = (
        backend_semantics_pass
        and autodiff_head_semantics_pass
        and direct_tail_semantics_pass
    )
    epsilon = float(args.finite_difference_epsilon)
    selected_stages = tuple(args.stages)
    if len(set(selected_stages)) != len(selected_stages):
        raise ValueError("duplicate stages are not allowed")
    for stage_index, stage in enumerate(selected_stages):
        clean = clean_capture[stage].to(device)
        shifted = shifted_capture[stage].to(device)
        tangent = clean - shifted
        tangent_norm = float(torch.linalg.vector_norm(tangent))
        if tangent_norm == 0.0:
            raise RuntimeError(f"zero clean-shift tangent at {stage}")

        function = lambda value, stage=stage: runtime.trajectory_xy_from_qualified_stage(
            stage, value
        )
        _, product = torch.autograd.functional.jvp(
            function, shifted, tangent, create_graph=False, strict=True
        )
        reverse_base = shifted.detach().clone().requires_grad_(True)
        reverse_output = function(reverse_base)
        probe_generator = torch.Generator(device=device)
        probe_generator.manual_seed(args.seed + 100 + stage_index)
        probe = torch.randn(
            reverse_output.shape,
            generator=probe_generator,
            device=device,
            dtype=reverse_output.dtype,
        )
        probe = probe / torch.linalg.vector_norm(probe)
        (vjp,) = torch.autograd.grad(torch.sum(reverse_output * probe), reverse_base)
        # The C1 identity contracts more than four million terms.  Evaluate the
        # audit scalar in float64 so reduction error is not mistaken for a JVP
        # implementation error; the planner and derivatives remain float32.
        adjoint_left = torch.sum(product.double() * probe.double())
        adjoint_right = torch.sum(tangent.double() * vjp.double())
        adjoint_absolute_error = float(torch.abs(adjoint_left - adjoint_right))
        adjoint_scale = max(
            float(torch.abs(adjoint_left)),
            float(torch.abs(adjoint_right)),
            args.adjoint_absolute_tolerance,
        )
        adjoint_relative_error = adjoint_absolute_error / adjoint_scale
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
                adjoint_relative_error <= args.adjoint_relative_tolerance
                or adjoint_absolute_error <= args.adjoint_absolute_tolerance
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
                "finite_difference_diagnostic_only": True,
                "adjoint_absolute_error": adjoint_absolute_error,
                "adjoint_relative_error": adjoint_relative_error,
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
        "selected_stages": list(selected_stages),
        "finite_difference_epsilon": epsilon,
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "adjoint_relative_tolerance": args.adjoint_relative_tolerance,
        "adjoint_absolute_tolerance": args.adjoint_absolute_tolerance,
        "backend_output_tolerance": args.backend_output_tolerance,
        "backend_output_max_abs_difference": backend_output_max_abs_difference,
        "backend_semantics_pass": backend_semantics_pass,
        "autodiff_head_max_abs_difference": autodiff_head_max_abs_difference,
        "autodiff_head_semantics_pass": autodiff_head_semantics_pass,
        "direct_tail_max_abs_difference": direct_tail_differences,
        "direct_tail_semantics_pass": direct_tail_semantics_pass,
        "deterministic_algorithms_warn_only": True,
        "sdpa_backend": "math_only__flash_false__mem_efficient_false",
        "mha_fastpath_enabled": False,
        "jvp_implementation": "reverse_over_reverse_with_math_sdpa",
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
