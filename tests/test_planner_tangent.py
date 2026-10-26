from __future__ import annotations

import torch

from aorta_drive.vision.planner_tangent import (
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
