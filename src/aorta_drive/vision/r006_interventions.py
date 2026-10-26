"""Registered low-rank and matched-control interventions for R006V0."""

from __future__ import annotations

import math

import torch

from .planner_tangent import orthonormalize


def validate_npc_basis_mask(
    delta: torch.Tensor,
    basis: torch.Tensor,
    site_mask: torch.Tensor,
) -> None:
    if delta.ndim != 3:
        raise ValueError("delta must have shape [N,P,C]")
    if basis.ndim != 2 or basis.shape[0] != delta.shape[-1]:
        raise ValueError("basis must have shape [C,r]")
    if site_mask.ndim != 1 or site_mask.shape[0] != delta.shape[1]:
        raise ValueError("site_mask must have shape [P]")
    if site_mask.dtype != torch.bool and not torch.all(
        (site_mask == 0) | (site_mask == 1)
    ):
        raise ValueError("site_mask must be binary")
    if any(not torch.isfinite(value).all() for value in (delta, basis, site_mask)):
        raise ValueError("intervention tensors must be finite")


def project_registered_delta(
    delta: torch.Tensor,
    basis: torch.Tensor,
    site_mask: torch.Tensor,
) -> torch.Tensor:
    """Project a clean-minus-shift delta without changing frozen sites."""

    validate_npc_basis_mask(delta, basis, site_mask)
    correction = (delta @ basis) @ basis.transpose(0, 1)
    mask = site_mask.to(device=delta.device, dtype=delta.dtype).view(1, -1, 1)
    return correction * mask


def energy_match(
    control: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Match Frobenius energy exactly and fail on a degenerate control."""

    if control.shape != target.shape:
        raise ValueError("control and target corrections must share a shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not torch.isfinite(control).all() or not torch.isfinite(target).all():
        raise ValueError("energy-matching inputs must be finite")
    control_norm = torch.linalg.vector_norm(control)
    target_norm = torch.linalg.vector_norm(target)
    if float(target_norm) <= epsilon:
        return torch.zeros_like(control)
    if float(control_norm) <= epsilon:
        raise ValueError("nonzero target cannot be matched by zero-energy control")
    return control * (target_norm / control_norm)


def seeded_haar_basis(
    channels: int,
    rank: int,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Draw one deterministic Haar-distributed channel subspace."""

    if channels <= 0 or rank <= 0 or rank > channels:
        raise ValueError("rank must lie in [1, channels]")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        channels,
        rank,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    return orthonormalize(matrix)


def energy_matched_random_subspace_correction(
    delta: torch.Tensor,
    target_correction: torch.Tensor,
    site_mask: torch.Tensor,
    *,
    rank: int,
    seed: int,
) -> torch.Tensor:
    """Project the same-scene delta into a seeded random subspace and match A1 energy."""

    basis = seeded_haar_basis(
        delta.shape[-1],
        rank,
        seed=seed,
        dtype=delta.dtype,
        device=delta.device,
    )
    raw = project_registered_delta(delta, basis, site_mask)
    return energy_match(raw, target_correction)


def empirical_gaussian_coefficient_field(
    training_coefficients: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Sample the full empirical coefficient-field covariance without materializing it.

    ``training_coefficients`` has shape ``[K,P,r]``.  Centering followed by
    Gaussian weights with variance ``1/(K-1)`` produces a zero-mean sample
    whose covariance is exactly the empirical covariance of the flattened
    outer-train coefficient fields, including cross-site correlations.
    """

    if training_coefficients.ndim != 3 or training_coefficients.shape[0] < 2:
        raise ValueError("training_coefficients must have shape [K>=2,P,r]")
    if not torch.isfinite(training_coefficients).all():
        raise ValueError("training coefficients must be finite")
    centered = training_coefficients - training_coefficients.mean(dim=0, keepdim=True)
    generator = torch.Generator(device=training_coefficients.device)
    generator.manual_seed(int(seed))
    weights = torch.randn(
        centered.shape[0],
        generator=generator,
        dtype=centered.dtype,
        device=centered.device,
    ) / math.sqrt(centered.shape[0] - 1)
    return torch.einsum("k,kpr->pr", weights, centered)


def energy_matched_gaussian_noise_correction(
    training_coefficients: torch.Tensor,
    basis: torch.Tensor,
    target_correction: torch.Tensor,
    site_mask: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Draw an empirical-covariance coefficient field and match A1 energy."""

    coefficients = empirical_gaussian_coefficient_field(
        training_coefficients,
        seed=seed,
    )
    if coefficients.shape != (
        target_correction.shape[1],
        basis.shape[1],
    ):
        raise ValueError("coefficient-field shape disagrees with target and basis")
    raw = (coefficients @ basis.transpose(0, 1)).unsqueeze(0)
    mask = site_mask.to(
        device=raw.device,
        dtype=raw.dtype,
    ).view(1, -1, 1)
    raw = raw * mask
    return energy_match(raw, target_correction)
