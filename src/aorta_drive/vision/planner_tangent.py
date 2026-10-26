"""Task-weighted channel-subspace utilities for R006 planner-tangent repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


@dataclass(frozen=True)
class StiefelSolveResult:
    basis: torch.Tensor
    objective: float
    steps: int
    initialization: str


def _validate_dense_inputs(
    deltas: torch.Tensor,
    jacobians: torch.Tensor,
    output_metric: torch.Tensor,
    basis: torch.Tensor,
) -> None:
    if deltas.ndim != 3:
        raise ValueError("deltas must have shape [N, P, C]")
    if jacobians.ndim != 4:
        raise ValueError("jacobians must have shape [N, M, P, C]")
    if jacobians.shape[0] != deltas.shape[0] or jacobians.shape[2:] != deltas.shape[1:]:
        raise ValueError("jacobian and delta dimensions disagree")
    if output_metric.shape != (jacobians.shape[1], jacobians.shape[1]):
        raise ValueError("output_metric must have shape [M, M]")
    if basis.ndim != 2 or basis.shape[0] != deltas.shape[2]:
        raise ValueError("basis must have shape [C, r]")
    values = (deltas, jacobians, output_metric, basis)
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError("all inputs must be finite")


def task_weighted_uncovered_objective(
    deltas: torch.Tensor,
    jacobians: torch.Tensor,
    output_metric: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Return mean ||J (I_P x (I-BB^T)) delta||_W^2.

    ``deltas`` use the spatial-channel convention ``[N, P, C]`` and the
    Jacobian is the explicit synthetic/reference representation
    ``[N, M, P, C]``. Production code may replace the final contraction by a
    matrix-free JVP, but it must agree with this oracle.
    """

    _validate_dense_inputs(deltas, jacobians, output_metric, basis)
    projected = torch.matmul(torch.matmul(deltas, basis), basis.transpose(0, 1))
    residual = deltas - projected
    output_residual = torch.einsum("nmpc,npc->nm", jacobians, residual)
    weighted = torch.einsum(
        "nm,mk,nk->n", output_residual, output_metric, output_residual
    )
    return weighted.mean()


def matrix_free_uncovered_objective(
    deltas: torch.Tensor,
    output_metric: torch.Tensor,
    basis: torch.Tensor,
    jvp: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Evaluate the same objective using only Jacobian-vector products.

    ``jvp`` receives a tangent with shape ``[N, P, C]`` and must return
    ``J tangent`` with shape ``[N, M]``.  The callable may retain autograd
    dependence on the tangent; production LTF code uses this property to
    differentiate the Stiefel objective without materializing a Jacobian.
    """

    if deltas.ndim != 3 or basis.ndim != 2:
        raise ValueError("expected deltas [N,P,C] and basis [C,r]")
    if basis.shape[0] != deltas.shape[-1]:
        raise ValueError("basis channel dimension disagrees with deltas")
    projected = torch.matmul(torch.matmul(deltas, basis), basis.transpose(0, 1))
    output_residual = jvp(deltas - projected)
    if output_residual.ndim != 2:
        raise ValueError("jvp must return [N,M]")
    if output_metric.shape != (output_residual.shape[1], output_residual.shape[1]):
        raise ValueError("output metric shape disagrees with jvp output")
    if not torch.isfinite(output_residual).all():
        raise ValueError("jvp returned non-finite values")
    weighted = torch.einsum(
        "nm,mk,nk->n", output_residual, output_metric, output_residual
    )
    return weighted.mean()


def orthonormalize(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError("orthonormalize expects a matrix")
    q, r = torch.linalg.qr(value, mode="reduced")
    signs = torch.where(torch.diag(r) < 0, -torch.ones_like(torch.diag(r)), torch.ones_like(torch.diag(r)))
    return q * signs.unsqueeze(0)


def projector_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Normalized Frobenius distance between two subspace projectors."""

    if left.shape != right.shape:
        raise ValueError("basis shapes disagree")
    rank = left.shape[1]
    if rank <= 0:
        raise ValueError("rank must be positive")
    p_left = left @ left.transpose(0, 1)
    p_right = right @ right.transpose(0, 1)
    return torch.linalg.matrix_norm(p_left - p_right) / torch.sqrt(
        torch.as_tensor(float(2 * rank), dtype=left.dtype, device=left.device)
    )


def _pca_initialization(deltas: torch.Tensor, rank: int) -> torch.Tensor:
    flat = deltas.reshape(-1, deltas.shape[-1])
    _, _, vh = torch.linalg.svd(flat, full_matrices=False)
    return vh[:rank].transpose(0, 1).contiguous()


def _random_initialization(
    channels: int, rank: int, *, seed: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return orthonormalize(
        torch.randn(channels, rank, generator=generator, dtype=dtype, device=device)
    )


def solve_task_weighted_basis(
    deltas: torch.Tensor,
    jacobians: torch.Tensor,
    output_metric: torch.Tensor,
    rank: int,
    *,
    random_seeds: Iterable[int] = (20261026, 20261027),
    initial_step: float = 1.0,
    armijo_shrink: float = 0.5,
    armijo_c: float = 1e-4,
    relative_tolerance: float = 1e-6,
    patience: int = 20,
    max_steps: int = 500,
    max_line_search_steps: int = 50,
) -> StiefelSolveResult:
    """Solve the preregistered Stiefel problem using QR-retracted GD."""

    return solve_matrix_free_basis(
        deltas,
        rank,
        lambda basis: task_weighted_uncovered_objective(
            deltas, jacobians, output_metric, basis
        ),
        random_seeds=random_seeds,
        initial_step=initial_step,
        armijo_shrink=armijo_shrink,
        armijo_c=armijo_c,
        relative_tolerance=relative_tolerance,
        patience=patience,
        max_steps=max_steps,
        max_line_search_steps=max_line_search_steps,
    )


def solve_matrix_free_basis(
    deltas: torch.Tensor,
    rank: int,
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    random_seeds: Iterable[int] = (20261026, 20261027),
    initial_step: float = 1.0,
    armijo_shrink: float = 0.5,
    armijo_c: float = 1e-4,
    relative_tolerance: float = 1e-6,
    patience: int = 20,
    max_steps: int = 500,
    max_line_search_steps: int = 50,
) -> StiefelSolveResult:
    """Solve a differentiable matrix-free Stiefel objective.

    The initialization and optimization contract is identical to the dense
    oracle solver.  Only the implementation of ``objective_fn`` changes.
    """

    channels = deltas.shape[-1]
    if rank <= 0 or rank > channels:
        raise ValueError("rank must lie in [1, C]")
    if not (0.0 < armijo_shrink < 1.0):
        raise ValueError("armijo_shrink must lie in (0, 1)")
    if min(initial_step, armijo_c, relative_tolerance) <= 0:
        raise ValueError("solver tolerances and steps must be positive")

    initializations: list[tuple[str, torch.Tensor]] = [
        ("discovery_channel_drift_pca", _pca_initialization(deltas, rank))
    ]
    for seed in random_seeds:
        initializations.append(
            (
                f"random_{seed}",
                _random_initialization(
                    channels,
                    rank,
                    seed=int(seed),
                    dtype=deltas.dtype,
                    device=deltas.device,
                ),
            )
        )

    results: list[StiefelSolveResult] = []
    for name, initial in initializations:
        basis = orthonormalize(initial).detach()
        stale = 0
        completed_steps = 0
        for step_index in range(max_steps):
            variable = basis.detach().requires_grad_(True)
            objective = objective_fn(variable)
            if objective.ndim != 0 or not torch.isfinite(objective):
                raise ValueError("objective_fn must return one finite scalar")
            (euclidean,) = torch.autograd.grad(objective, variable)
            symmetric = 0.5 * (
                variable.transpose(0, 1) @ euclidean
                + euclidean.transpose(0, 1) @ variable
            )
            riemannian = euclidean - variable @ symmetric
            gradient_sq = torch.sum(riemannian * riemannian).detach()
            if float(gradient_sq) == 0.0:
                completed_steps = step_index
                break

            old_value = objective.detach()
            step_size = initial_step
            accepted = False
            candidate = basis
            candidate_value = old_value
            for _ in range(max_line_search_steps):
                candidate = orthonormalize(
                    basis - step_size * riemannian.detach()
                ).detach()
                candidate_value = objective_fn(candidate).detach()
                if candidate_value <= old_value - armijo_c * step_size * gradient_sq:
                    accepted = True
                    break
                step_size *= armijo_shrink
            if not accepted:
                completed_steps = step_index
                break

            relative = float((old_value - candidate_value) / torch.clamp(old_value.abs(), min=1e-30))
            stale = stale + 1 if relative < relative_tolerance else 0
            basis = candidate
            completed_steps = step_index + 1
            if stale >= patience:
                break

        final_value = float(objective_fn(basis).detach())
        results.append(
            StiefelSolveResult(
                basis=basis,
                objective=final_value,
                steps=completed_steps,
                initialization=name,
            )
        )
    return min(results, key=lambda item: (item.objective, item.initialization))
