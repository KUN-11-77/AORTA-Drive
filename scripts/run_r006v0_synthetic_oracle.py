#!/usr/bin/env python
"""R006V0 synthetic dense-Jacobian qualification oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aorta_drive.vision.planner_tangent import (
    projector_distance,
    solve_task_weighted_basis,
    task_weighted_uncovered_objective,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--objective-relative-tolerance", type=float, default=1e-8)
    parser.add_argument("--gradient-relative-tolerance", type=float, default=1e-7)
    parser.add_argument("--projector-distance-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float64
    samples, positions, channels, outputs, rank = 48, 5, 10, 6, 2
    true_basis, _ = torch.linalg.qr(torch.randn(channels, rank, dtype=dtype))
    coefficients = torch.randn(samples, positions, rank, dtype=dtype)
    deltas = coefficients @ true_basis.transpose(0, 1)
    jacobians = torch.randn(samples, outputs, positions, channels, dtype=dtype)
    metric_factor = torch.randn(outputs, outputs, dtype=dtype)
    output_metric = metric_factor.transpose(0, 1) @ metric_factor + 0.1 * torch.eye(outputs, dtype=dtype)

    probe = torch.randn(channels, rank, dtype=dtype)
    probe, _ = torch.linalg.qr(probe)
    dense_value = task_weighted_uncovered_objective(
        deltas, jacobians, output_metric, probe
    )

    def tail(flattened: torch.Tensor) -> torch.Tensor:
        value = flattened.reshape(samples, positions, channels)
        return torch.einsum("nmpc,npc->nm", jacobians, value)

    residual = deltas - (deltas @ probe) @ probe.transpose(0, 1)
    zeros = torch.zeros_like(residual).requires_grad_(True)
    _, jvp = torch.autograd.functional.jvp(
        tail, (zeros,), (residual,), create_graph=True, strict=False
    )
    matrix_free_value = torch.einsum("nm,mk,nk->", jvp, output_metric, jvp) / samples
    objective_relative_error = float(
        torch.abs(matrix_free_value - dense_value)
        / torch.clamp(torch.abs(dense_value), min=1e-30)
    )

    dense_probe = probe.detach().clone().requires_grad_(True)
    dense_loss = task_weighted_uncovered_objective(
        deltas, jacobians, output_metric, dense_probe
    )
    (dense_gradient,) = torch.autograd.grad(dense_loss, dense_probe)
    matrix_probe = probe.detach().clone().requires_grad_(True)
    matrix_residual = deltas - (deltas @ matrix_probe) @ matrix_probe.transpose(0, 1)
    _, matrix_jvp = torch.autograd.functional.jvp(
        tail,
        (torch.zeros_like(matrix_residual),),
        (matrix_residual,),
        create_graph=True,
        strict=False,
    )
    matrix_loss = torch.einsum(
        "nm,mk,nk->", matrix_jvp, output_metric, matrix_jvp
    ) / samples
    (matrix_gradient,) = torch.autograd.grad(matrix_loss, matrix_probe)
    gradient_relative_error = float(
        torch.linalg.vector_norm(matrix_gradient - dense_gradient)
        / torch.clamp(torch.linalg.vector_norm(dense_gradient), min=1e-30)
    )

    solved = solve_task_weighted_basis(
        deltas,
        jacobians,
        output_metric,
        rank,
        random_seeds=(20261026, 20261027),
    )
    recovered_distance = float(projector_distance(solved.basis, true_basis))
    pass_status = (
        objective_relative_error <= args.objective_relative_tolerance
        and gradient_relative_error <= args.gradient_relative_tolerance
        and recovered_distance <= args.projector_distance_tolerance
    )
    report = {
        "artifact": "R006V0_SYNTHETIC_DENSE_JACOBIAN_ORACLE_V1",
        "seed": args.seed,
        "shape": {
            "samples": samples,
            "positions": positions,
            "channels": channels,
            "outputs": outputs,
            "rank": rank,
        },
        "objective_relative_error": objective_relative_error,
        "gradient_relative_error": gradient_relative_error,
        "recovered_projector_distance": recovered_distance,
        "solver_objective": solved.objective,
        "solver_steps": solved.steps,
        "solver_initialization": solved.initialization,
        "thresholds": {
            "objective_relative_tolerance": args.objective_relative_tolerance,
            "gradient_relative_tolerance": args.gradient_relative_tolerance,
            "projector_distance_tolerance": args.projector_distance_tolerance,
        },
        "pass": pass_status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not pass_status:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
