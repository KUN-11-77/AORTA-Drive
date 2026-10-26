from __future__ import annotations

import torch

from aorta_drive.vision.planner_tangent import (
    matrix_free_uncovered_objective,
    orthonormalize,
    projector_distance,
    solve_task_weighted_basis,
    task_weighted_uncovered_objective,
)


def test_task_weighted_basis_recovers_known_channel_subspace() -> None:
    torch.manual_seed(7)
    dtype = torch.float64
    n, p, c, m, rank = 24, 3, 7, 4, 2
    truth = orthonormalize(torch.randn(c, rank, dtype=dtype))
    deltas = torch.randn(n, p, rank, dtype=dtype) @ truth.T
    jacobians = torch.randn(n, m, p, c, dtype=dtype)
    metric = torch.eye(m, dtype=dtype)
    result = solve_task_weighted_basis(
        deltas,
        jacobians,
        metric,
        rank,
        max_steps=300,
        patience=10,
    )
    assert result.objective < 1e-10
    assert float(projector_distance(result.basis, truth)) < 1e-4


def test_uncovered_objective_rejects_nonorthogonal_irrelevant_basis() -> None:
    dtype = torch.float64
    deltas = torch.zeros(1, 1, 3, dtype=dtype)
    deltas[..., 0] = 1.0
    jacobian = torch.zeros(1, 1, 1, 3, dtype=dtype)
    jacobian[..., 0] = 1.0
    metric = torch.ones(1, 1, dtype=dtype)
    relevant = torch.tensor([[1.0], [0.0], [0.0]], dtype=dtype)
    irrelevant = torch.tensor([[0.0], [1.0], [0.0]], dtype=dtype)
    assert float(task_weighted_uncovered_objective(deltas, jacobian, metric, relevant)) == 0.0
    assert float(task_weighted_uncovered_objective(deltas, jacobian, metric, irrelevant)) == 1.0


def test_matrix_free_objective_and_gradient_match_dense_oracle() -> None:
    torch.manual_seed(19)
    dtype = torch.float64
    n, p, c, m, rank = 5, 4, 6, 3, 2
    deltas = torch.randn(n, p, c, dtype=dtype)
    jacobians = torch.randn(n, m, p, c, dtype=dtype)
    metric = torch.randn(m, m, dtype=dtype)
    metric = metric.T @ metric
    dense_basis = orthonormalize(torch.randn(c, rank, dtype=dtype)).requires_grad_(True)
    free_basis = dense_basis.detach().clone().requires_grad_(True)

    dense = task_weighted_uncovered_objective(
        deltas, jacobians, metric, dense_basis
    )
    free = matrix_free_uncovered_objective(
        deltas,
        metric,
        free_basis,
        lambda tangent: torch.einsum("nmpc,npc->nm", jacobians, tangent),
    )
    dense_gradient = torch.autograd.grad(dense, dense_basis)[0]
    free_gradient = torch.autograd.grad(free, free_basis)[0]
    torch.testing.assert_close(free, dense, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        free_gradient, dense_gradient, rtol=1e-12, atol=1e-12
    )


def test_autograd_jvp_preserves_basis_gradient_through_tangent() -> None:
    torch.manual_seed(23)
    dtype = torch.float64
    operator = torch.randn(4, 6, dtype=dtype)
    base = torch.randn(6, dtype=dtype)
    delta = torch.randn(6, dtype=dtype)
    basis = orthonormalize(torch.randn(6, 2, dtype=dtype)).requires_grad_(True)
    tangent = delta - (delta @ basis) @ basis.T
    _, product = torch.func.jvp(
        lambda value: operator @ value, (base,), (tangent,)
    )
    loss = torch.sum(product * product)
    gradient = torch.autograd.grad(loss, basis)[0]
    assert torch.isfinite(gradient).all()
    assert float(torch.linalg.vector_norm(gradient)) > 0.0
