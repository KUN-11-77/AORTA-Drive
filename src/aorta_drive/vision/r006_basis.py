"""Exact matrix-free batch primitives for formal R006V0 basis discovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch


TailFunction = Callable[[torch.Tensor], torch.Tensor]


def _validate_batch(
    bases: torch.Tensor,
    deltas: torch.Tensor,
    basis: torch.Tensor,
    site_mask: torch.Tensor,
) -> None:
    if bases.ndim != 3 or deltas.shape != bases.shape:
        raise ValueError("bases and deltas must have equal [N,P,C] shape")
    if basis.ndim != 2 or basis.shape[0] != bases.shape[-1]:
        raise ValueError("basis must have shape [C,r]")
    if site_mask.ndim != 1 or site_mask.shape[0] != bases.shape[1]:
        raise ValueError("site_mask must have shape [P]")
    if site_mask.dtype != torch.bool and not torch.all(
        (site_mask == 0) | (site_mask == 1)
    ):
        raise ValueError("site_mask must be binary")
    if any(
        not torch.isfinite(value).all() for value in (bases, deltas, basis, site_mask)
    ):
        raise ValueError("all basis-discovery inputs must be finite")


def trajectory_xy_jvp_energy(
    function: TailFunction,
    bases: torch.Tensor,
    tangents: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Return per-sample ``mean_t ||J tangent||_2^2``.

    The output of ``function`` is flattened XY with shape ``[N,2T]``.  Thus
    this is exactly the quadratic form ``W=I_(2T)/T`` frozen in amendment 6.
    """

    if bases.ndim != 3 or tangents.shape != bases.shape:
        raise ValueError("bases and tangents must have equal [N,P,C] shape")
    _, product = torch.autograd.functional.jvp(
        function,
        bases,
        tangents,
        create_graph=create_graph,
        strict=True,
    )
    if product.ndim != 2 or product.shape[0] != bases.shape[0]:
        raise ValueError("tail JVP must have shape [N,2T]")
    if product.shape[1] <= 0 or product.shape[1] % 2:
        raise ValueError("tail JVP output dimension must equal 2T")
    if not torch.isfinite(product).all():
        raise ValueError("tail JVP returned non-finite values")
    time_steps = product.shape[1] // 2
    return cast(torch.Tensor, torch.sum(product * product, dim=1) / time_steps)


def matrix_free_batch_objective(
    function: TailFunction,
    bases: torch.Tensor,
    deltas: torch.Tensor,
    basis: torch.Tensor,
    site_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the complete-mean objective contribution for one fixed batch."""

    _validate_batch(bases, deltas, basis, site_mask)
    mask = site_mask.to(device=deltas.device, dtype=deltas.dtype).view(1, -1, 1)
    projected = ((deltas @ basis) @ basis.transpose(0, 1)) * mask
    residual = deltas - projected
    return trajectory_xy_jvp_energy(
        function,
        bases,
        residual,
        create_graph=basis.requires_grad,
    ).mean()


def evaluate_basis_coverage(
    function: TailFunction,
    bases: torch.Tensor,
    deltas: torch.Tensor,
    basis: torch.Tensor,
    site_mask: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample total and uncovered task drift in fixed order."""

    _validate_batch(bases, deltas, basis, site_mask)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    mask = site_mask.to(device=deltas.device, dtype=deltas.dtype).view(1, -1, 1)
    totals = []
    uncovered = []
    for start in range(0, bases.shape[0], batch_size):
        end = min(start + batch_size, bases.shape[0])
        current_bases = bases[start:end]
        current_deltas = deltas[start:end]
        projected = ((current_deltas @ basis) @ basis.transpose(0, 1)) * mask
        totals.append(
            trajectory_xy_jvp_energy(
                function,
                current_bases,
                current_deltas,
                create_graph=False,
            ).detach()
        )
        uncovered.append(
            trajectory_xy_jvp_energy(
                function,
                current_bases,
                current_deltas - projected,
                create_graph=False,
            ).detach()
        )
    return torch.cat(totals), torch.cat(uncovered)
