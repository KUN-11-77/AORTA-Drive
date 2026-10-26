from __future__ import annotations

import pytest
import torch

from aorta_drive.vision.r006_interventions import (
    empirical_gaussian_coefficient_field,
    energy_match,
    energy_matched_gaussian_noise_correction,
    energy_matched_random_subspace_correction,
    project_registered_delta,
    seeded_haar_basis,
)


def test_registered_projection_preserves_frozen_site() -> None:
    delta = torch.ones(1, 3, 4, dtype=torch.float64)
    basis = torch.eye(4, dtype=torch.float64)[:, :2]
    mask = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    correction = project_registered_delta(delta, basis, mask)
    torch.testing.assert_close(correction[:, 2], torch.zeros(1, 4, dtype=torch.float64))
    torch.testing.assert_close(
        correction[:, :2, :2], torch.ones(1, 2, 2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        correction[:, :2, 2:], torch.zeros(1, 2, 2, dtype=torch.float64)
    )


def test_energy_match_is_exact_and_fail_closed() -> None:
    control = torch.tensor([[[3.0, 4.0]]], dtype=torch.float64)
    target = torch.tensor([[[0.0, 10.0]]], dtype=torch.float64)
    matched = energy_match(control, target)
    torch.testing.assert_close(
        torch.linalg.vector_norm(matched),
        torch.linalg.vector_norm(target),
    )
    with pytest.raises(ValueError, match="zero-energy"):
        energy_match(torch.zeros_like(control), target)


def test_seeded_haar_basis_is_orthonormal_and_deterministic() -> None:
    first = seeded_haar_basis(
        8,
        3,
        seed=20261026,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    second = seeded_haar_basis(
        8,
        3,
        seed=20261026,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.T @ first, torch.eye(3, dtype=torch.float64))


def test_random_subspace_control_matches_target_energy_and_mask() -> None:
    delta = torch.arange(24, dtype=torch.float64).reshape(1, 3, 8)
    mask = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    target = torch.zeros_like(delta)
    target[:, :2, :2] = 2.0
    control = energy_matched_random_subspace_correction(
        delta,
        target,
        mask,
        rank=2,
        seed=20261026,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(control),
        torch.linalg.vector_norm(target),
    )
    torch.testing.assert_close(control[:, 2], torch.zeros(1, 8, dtype=torch.float64))


def test_empirical_gaussian_field_preserves_shape_seed_and_centered_covariance() -> (
    None
):
    training = torch.arange(60, dtype=torch.float64).reshape(5, 3, 4)
    first = empirical_gaussian_coefficient_field(training, seed=20261026)
    second = empirical_gaussian_coefficient_field(training, seed=20261026)
    assert first.shape == (3, 4)
    torch.testing.assert_close(first, second)


def test_gaussian_control_matches_energy_and_freezes_status_site() -> None:
    torch.manual_seed(4)
    training = torch.randn(8, 3, 2, dtype=torch.float64)
    training[:, 2] = 0.0
    basis = torch.eye(4, dtype=torch.float64)[:, :2]
    target = torch.zeros(1, 3, 4, dtype=torch.float64)
    target[:, :2, :2] = 1.0
    mask = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    control = energy_matched_gaussian_noise_correction(
        training,
        basis,
        target,
        mask,
        seed=20261026,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(control),
        torch.linalg.vector_norm(target),
    )
    torch.testing.assert_close(control[:, 2], torch.zeros(1, 4, dtype=torch.float64))
