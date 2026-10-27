"""Task-weighted channel-subspace utilities for R006 planner-tangent repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, cast

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
    site_mask: torch.Tensor | None,
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
    if site_mask is not None:
        if site_mask.ndim != 1 or site_mask.shape[0] != deltas.shape[1]:
            raise ValueError("site_mask must have shape [P]")
        if site_mask.dtype == torch.bool:
            pass
        elif not torch.all((site_mask == 0) | (site_mask == 1)):
            raise ValueError("site_mask must be binary")
    values: tuple[torch.Tensor, ...] = (deltas, jacobians, output_metric, basis)
    if site_mask is not None:
        values = (*values, site_mask)
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError("all inputs must be finite")


def _masked_channel_projection(
    deltas: torch.Tensor,
    basis: torch.Tensor,
    site_mask: torch.Tensor | None,
) -> torch.Tensor:
    projected = torch.matmul(torch.matmul(deltas, basis), basis.transpose(0, 1))
    if site_mask is None:
        return projected
    mask = site_mask.to(dtype=deltas.dtype, device=deltas.device).view(1, -1, 1)
    return projected * mask


def task_weighted_uncovered_objective(
    deltas: torch.Tensor,
    jacobians: torch.Tensor,
    output_metric: torch.Tensor,
    basis: torch.Tensor,
    *,
    site_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the registered-site uncovered task-weighted drift.

    ``deltas`` use the spatial-channel convention ``[N, P, C]`` and the
    Jacobian is the explicit synthetic/reference representation
    ``[N, M, P, C]``. Production code may replace the final contraction by a
    matrix-free JVP, but it must agree with this oracle. ``site_mask`` freezes
    non-repairable sites by excluding their projection while retaining their
    drift in the uncovered residual.
    """

    _validate_dense_inputs(deltas, jacobians, output_metric, basis, site_mask)
    projected = _masked_channel_projection(deltas, basis, site_mask)
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
    *,
    site_mask: torch.Tensor | None = None,
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
    if site_mask is not None:
        if site_mask.ndim != 1 or site_mask.shape[0] != deltas.shape[1]:
            raise ValueError("site_mask must have shape [P]")
        if site_mask.dtype != torch.bool and not torch.all(
            (site_mask == 0) | (site_mask == 1)
        ):
            raise ValueError("site_mask must be binary")
        if not torch.isfinite(site_mask).all():
            raise ValueError("site_mask must be finite")
    projected = _masked_channel_projection(deltas, basis, site_mask)
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
    signs = torch.where(
        torch.diag(r) < 0,
        -torch.ones_like(torch.diag(r)),
        torch.ones_like(torch.diag(r)),
    )
    return cast(torch.Tensor, q * signs.unsqueeze(0))


def projector_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Normalized Frobenius distance between two subspace projectors."""

    if left.shape != right.shape:
        raise ValueError("basis shapes disagree")
    rank = left.shape[1]
    if rank <= 0:
        raise ValueError("rank must be positive")
    p_left = left @ left.transpose(0, 1)
    p_right = right @ right.transpose(0, 1)
    return cast(
        torch.Tensor,
        torch.linalg.matrix_norm(p_left - p_right)
        / torch.sqrt(
            torch.as_tensor(float(2 * rank), dtype=left.dtype, device=left.device)
        ),
    )


def _pca_initialization(deltas: torch.Tensor, rank: int) -> torch.Tensor:
    flat = deltas.reshape(-1, deltas.shape[-1])
    _, _, vh = torch.linalg.svd(flat, full_matrices=False)
    return cast(torch.Tensor, vh[:rank].transpose(0, 1).contiguous())


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
    site_mask: torch.Tensor | None = None,
    include_pca_initialization: bool = True,
) -> StiefelSolveResult:
    """Solve the preregistered Stiefel problem using QR-retracted GD."""

    return solve_matrix_free_basis(
        deltas,
        rank,
        lambda basis: task_weighted_uncovered_objective(
            deltas,
            jacobians,
            output_metric,
            basis,
            site_mask=site_mask,
        ),
        random_seeds=random_seeds,
        initial_step=initial_step,
        armijo_shrink=armijo_shrink,
        armijo_c=armijo_c,
        relative_tolerance=relative_tolerance,
        patience=patience,
        max_steps=max_steps,
        max_line_search_steps=max_line_search_steps,
        include_pca_initialization=include_pca_initialization,
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
    include_pca_initialization: bool = True,
) -> StiefelSolveResult:
    """Solve a differentiable matrix-free Stiefel objective.

    The initialization and optimization contract is identical to the dense
    oracle solver.  Only the implementation of ``objective_fn`` changes.
    """

    def value_and_gradient(basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        variable = basis.detach().requires_grad_(True)
        objective = objective_fn(variable)
        if objective.ndim != 0 or not torch.isfinite(objective):
            raise ValueError("objective_fn must return one finite scalar")
        (gradient,) = torch.autograd.grad(objective, variable)
        return objective.detach(), gradient.detach()

    def value_only(basis: torch.Tensor) -> torch.Tensor:
        value = objective_fn(basis).detach()
        if value.ndim != 0 or not torch.isfinite(value):
            raise ValueError("objective_fn must return one finite scalar")
        return value

    return _solve_stiefel_with_oracle(
        deltas,
        rank,
        value_and_gradient,
        value_only,
        random_seeds=random_seeds,
        initial_step=initial_step,
        armijo_shrink=armijo_shrink,
        armijo_c=armijo_c,
        relative_tolerance=relative_tolerance,
        patience=patience,
        max_steps=max_steps,
        max_line_search_steps=max_line_search_steps,
        include_pca_initialization=include_pca_initialization,
    )


def solve_streaming_matrix_free_basis(
    deltas: torch.Tensor,
    rank: int,
    batch_objective_fn: Callable[[torch.Tensor, int, int], torch.Tensor],
    *,
    batch_size: int,
    random_seeds: Iterable[int] = (20261026, 20261027),
    initial_step: float = 1.0,
    armijo_shrink: float = 0.5,
    armijo_c: float = 1e-4,
    relative_tolerance: float = 1e-6,
    patience: int = 20,
    max_steps: int = 500,
    max_line_search_steps: int = 50,
    include_pca_initialization: bool = True,
) -> StiefelSolveResult:
    """Solve the same Stiefel objective with exact fixed-order batch reduction.

    ``batch_objective_fn`` returns the mean objective for ``[start:end]``.
    Values and Euclidean gradients are weighted by batch cardinality, so the
    result optimizes the full-sample mean without retaining all JVP graphs at
    once.  Batch order and size are part of formal provenance.
    """

    if deltas.ndim != 3 or deltas.shape[0] <= 0:
        raise ValueError("deltas must have shape [N>0,P,C]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sample_count = deltas.shape[0]

    def batches() -> Iterable[tuple[int, int]]:
        for start in range(0, sample_count, batch_size):
            yield start, min(start + batch_size, sample_count)

    def value_only(basis: torch.Tensor) -> torch.Tensor:
        total = basis.new_zeros(())
        for start, end in batches():
            value = batch_objective_fn(basis, start, end)
            if value.ndim != 0 or not torch.isfinite(value):
                raise ValueError("batch objective must return one finite scalar")
            total = total + value.detach() * (end - start)
        return total / sample_count

    def value_and_gradient(basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        variable = basis.detach().requires_grad_(True)
        total_value = basis.new_zeros(())
        total_gradient = torch.zeros_like(basis)
        for start, end in batches():
            value = batch_objective_fn(variable, start, end)
            if value.ndim != 0 or not torch.isfinite(value):
                raise ValueError("batch objective must return one finite scalar")
            (gradient,) = torch.autograd.grad(value, variable)
            weight = end - start
            total_value = total_value + value.detach() * weight
            total_gradient = total_gradient + gradient.detach() * weight
        return total_value / sample_count, total_gradient / sample_count

    return _solve_stiefel_with_oracle(
        deltas,
        rank,
        value_and_gradient,
        value_only,
        random_seeds=random_seeds,
        initial_step=initial_step,
        armijo_shrink=armijo_shrink,
        armijo_c=armijo_c,
        relative_tolerance=relative_tolerance,
        patience=patience,
        max_steps=max_steps,
        max_line_search_steps=max_line_search_steps,
        include_pca_initialization=include_pca_initialization,
    )


def _solve_stiefel_with_oracle(
    deltas: torch.Tensor,
    rank: int,
    value_and_gradient_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    value_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    random_seeds: Iterable[int],
    initial_step: float,
    armijo_shrink: float,
    armijo_c: float,
    relative_tolerance: float,
    patience: int,
    max_steps: int,
    max_line_search_steps: int,
    include_pca_initialization: bool,
) -> StiefelSolveResult:
    """Shared deterministic QR-retracted optimization loop."""

    channels = deltas.shape[-1]
    if rank <= 0 or rank > channels:
        raise ValueError("rank must lie in [1, C]")
    if not (0.0 < armijo_shrink < 1.0):
        raise ValueError("armijo_shrink must lie in (0, 1)")
    if min(initial_step, armijo_c, relative_tolerance) <= 0:
        raise ValueError("solver tolerances and steps must be positive")

    initializations: list[tuple[str, torch.Tensor]] = []
    if include_pca_initialization:
        initializations.append(
            ("discovery_channel_drift_pca", _pca_initialization(deltas, rank))
        )
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
    if not initializations:
        raise ValueError("at least one PCA or random initialization is required")

    results: list[StiefelSolveResult] = []
    for name, initial in initializations:
        basis = orthonormalize(initial).detach()
        stale = 0
        completed_steps = 0
        for step_index in range(max_steps):
            old_value, euclidean = value_and_gradient_fn(basis)
            if euclidean.shape != basis.shape or not torch.isfinite(euclidean).all():
                raise ValueError("gradient oracle returned an invalid gradient")
            symmetric = 0.5 * (
                basis.transpose(0, 1) @ euclidean + euclidean.transpose(0, 1) @ basis
            )
            riemannian = euclidean - basis @ symmetric
            gradient_sq = torch.sum(riemannian * riemannian).detach()
            if float(gradient_sq) == 0.0:
                completed_steps = step_index
                break

            step_size = initial_step
            accepted = False
            candidate = basis
            candidate_value = old_value
            for _ in range(max_line_search_steps):
                candidate = orthonormalize(
                    basis - step_size * riemannian.detach()
                ).detach()
                candidate_value = value_fn(candidate).detach()
                if candidate_value <= old_value - armijo_c * step_size * gradient_sq:
                    accepted = True
                    break
                step_size *= armijo_shrink
            if not accepted:
                completed_steps = step_index
                break

            relative = float(
                (old_value - candidate_value) / torch.clamp(old_value.abs(), min=1e-30)
            )
            stale = stale + 1 if relative < relative_tolerance else 0
            basis = candidate
            completed_steps = step_index + 1
            if stale >= patience:
                break

        final_value = float(value_fn(basis).detach())
        results.append(
            StiefelSolveResult(
                basis=basis,
                objective=final_value,
                steps=completed_steps,
                initialization=name,
            )
        )
    return min(results, key=lambda item: (item.objective, item.initialization))
