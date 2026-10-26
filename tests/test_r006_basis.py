from __future__ import annotations

import torch

from aorta_drive.vision.planner_tangent import (
    task_weighted_uncovered_objective,
)
from aorta_drive.vision.r006_basis import (
    evaluate_basis_coverage,
    matrix_free_batch_objective,
    trajectory_xy_jvp_energy,
)


def test_matrix_free_batch_objective_matches_explicit_jacobian_value_and_gradient() -> (
    None
):
    torch.manual_seed(41)
    dtype = torch.float64
    n, p, c, time_steps, rank = 7, 3, 5, 4, 2
    outputs = 2 * time_steps
    bases = torch.randn(n, p, c, dtype=dtype)
    deltas = torch.randn(n, p, c, dtype=dtype)
    jacobians = torch.randn(n, outputs, p, c, dtype=dtype)
    site_mask = torch.tensor([1, 1, 0], dtype=torch.bool)

    def function(value: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nmpc,npc->nm", jacobians, value)

    candidate = torch.randn(c, rank, dtype=dtype, requires_grad=True)
    candidate = torch.linalg.qr(candidate, mode="reduced").Q
    matrix_free = matrix_free_batch_objective(
        function,
        bases,
        deltas,
        candidate,
        site_mask,
    )
    metric = torch.eye(outputs, dtype=dtype) / time_steps
    dense = task_weighted_uncovered_objective(
        deltas,
        jacobians,
        metric,
        candidate,
        site_mask=site_mask,
    )
    matrix_free_gradient = torch.autograd.grad(
        matrix_free, candidate, retain_graph=True
    )[0]
    dense_gradient = torch.autograd.grad(dense, candidate)[0]
    assert torch.allclose(matrix_free, dense, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        matrix_free_gradient,
        dense_gradient,
        atol=1e-9,
        rtol=1e-9,
    )


def test_coverage_uses_fixed_order_and_registered_metric() -> None:
    dtype = torch.float64
    bases = torch.zeros(5, 2, 3, dtype=dtype)
    deltas = torch.zeros_like(bases)
    deltas[:, 0, 0] = torch.arange(1, 6, dtype=dtype)
    basis = torch.tensor([[1.0], [0.0], [0.0]], dtype=dtype)
    mask = torch.tensor([True, False])

    def function(value: torch.Tensor) -> torch.Tensor:
        xy = value[:, 0, :2]
        return torch.cat((xy, 2.0 * xy), dim=1)

    total, uncovered = evaluate_basis_coverage(
        function,
        bases,
        deltas,
        basis,
        mask,
        batch_size=2,
    )
    direct = trajectory_xy_jvp_energy(
        function,
        bases,
        deltas,
        create_graph=False,
    )
    assert torch.equal(total, direct)
    assert torch.equal(uncovered, torch.zeros_like(uncovered))
