from __future__ import annotations

import pytest
import torch

from aorta_drive.models.heterogeneous_envelope import (
    BoundedEdgeScaleHead,
    HeterogeneousEnvelopePredictor,
    uncentered_global_rms_normalize,
)
from aorta_drive.models.quotient_field import QuotientFieldPredictor
from aorta_drive.training.r004h2d import (
    analytic_adamw_first_step,
    initial_scale_gradient,
    rms_only_normalize,
)


def test_analytic_initial_gradient_matches_autograd() -> None:
    torch.manual_seed(7)
    spatial = torch.randn(1, 3, 4, 4, dtype=torch.float64)
    residual = torch.rand(1, 24, dtype=torch.float64) * 2.0
    analytic = initial_scale_gradient(spatial, residual)
    head = BoundedEdgeScaleHead(input_channels=3, grid_size=4).to(torch.float64)
    scales = head(spatial)
    # Direct residuals are used because arbitrary grid edges need not form an
    # integrable potential; the expression is the production proper loss at s=1.
    loss = torch.mean(residual / scales + torch.log(scales))
    torch.autograd.backward(loss)
    projection_bias = head.projection.bias
    assert head.projection.weight.grad is not None
    assert projection_bias is not None and projection_bias.grad is not None
    torch.testing.assert_close(
        head.projection.weight.grad[:, :, 0, 0], analytic.weight, rtol=1e-6, atol=2e-8
    )
    torch.testing.assert_close(
        projection_bias.grad, analytic.bias, rtol=1e-6, atol=2e-8
    )


def test_rms_normalization_changes_only_global_scale() -> None:
    spatial = torch.arange(1, 49, dtype=torch.float64).reshape(1, 3, 4, 4) * 1000.0
    normalized, rms = rms_only_normalize(spatial)
    assert rms > 1000.0
    assert torch.sqrt(torch.mean(normalized.square())).item() == pytest.approx(1.0)
    torch.testing.assert_close(normalized * rms, spatial)


def test_normalization_reduces_first_adamw_logit_jump() -> None:
    spatial = torch.full((1, 3, 4, 4), 10000.0, dtype=torch.float64)
    residual = torch.full((1, 24), 2.0, dtype=torch.float64)
    raw_gradient = initial_scale_gradient(spatial, residual)
    normalized, _ = rms_only_normalize(spatial)
    normalized_gradient = initial_scale_gradient(normalized, residual)
    raw_step = analytic_adamw_first_step(spatial, raw_gradient)
    normalized_step = analytic_adamw_first_step(normalized, normalized_gradient)
    assert raw_step.maximum_absolute_logit_change > 1.0
    assert normalized_step.maximum_absolute_logit_change < 0.01
    assert raw_step.safe_learning_rate < normalized_step.safe_learning_rate


def test_bounded_parameterization_does_not_amplify_initial_log_scale_gradient() -> None:
    spatial = torch.ones(1, 2, 3, 3)
    residual = torch.ones(1, 12)
    gradient = initial_scale_gradient(spatial, residual)
    assert 0.9 < gradient.bounded_to_log_scale_derivative_ratio < 1.0


def test_uncentered_rms_normalization_is_positive_scale_invariant() -> None:
    torch.manual_seed(11)
    spatial = torch.randn(2, 5, 4, 4)
    normalized = uncentered_global_rms_normalize(spatial)
    rescaled = uncentered_global_rms_normalize(spatial * 2017.0)
    torch.testing.assert_close(normalized, rescaled, rtol=2e-6, atol=2e-6)
    rms = torch.sqrt(torch.mean(normalized.square(), dim=(1, 2, 3)))
    torch.testing.assert_close(rms, torch.ones_like(rms))


def test_scale_branch_normalization_does_not_change_potential_path() -> None:
    torch.manual_seed(13)
    potential = QuotientFieldPredictor(
        input_dimension=8,
        token_dimension=8,
        grid_size=8,
        spatial_widths=(8, 8, 8),
    )
    raw = HeterogeneousEnvelopePredictor(potential, scale_input_normalization="none")
    normalized = HeterogeneousEnvelopePredictor(
        potential,
        scale_input_normalization="uncentered_global_rms",
    )
    normalized.scale_head.load_state_dict(raw.scale_head.state_dict(), strict=True)
    features = torch.randn(1, 64, 12, 8)
    mask = torch.ones(1, 64, 12, dtype=torch.bool)
    raw_output = raw(features, mask)
    normalized_output = normalized(features, mask)
    torch.testing.assert_close(raw_output.potential, normalized_output.potential)
    assert sum(
        parameter.numel()
        for parameter in normalized.parameters()
        if parameter.requires_grad
    ) == 18


def test_uncentered_rms_normalization_rejects_zero_gauge() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        uncentered_global_rms_normalize(torch.zeros(1, 3, 4, 4))
