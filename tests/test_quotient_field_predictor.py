from __future__ import annotations

import math

import pytest
import torch

from aorta_drive.models.quotient_field import (
    DevelopmentCheckpoint,
    QuotientFieldPredictor,
    oriented_grid_gradients,
    quotient_gradient_loss,
    select_development_checkpoint,
)


def _small_model() -> QuotientFieldPredictor:
    torch.manual_seed(7)
    return QuotientFieldPredictor(
        input_dimension=16,
        token_dimension=8,
        grid_size=8,
        roi_half_extent_m=4.0,
        spatial_widths=(16, 24, 32),
    )


def test_predictor_is_finite_zero_gauge_and_differentiable() -> None:
    model = _small_model()
    features = torch.randn(2, 64, 12, 16)
    mask = torch.rand(2, 64, 12) > 0.35
    mask[:, 0, :] = False
    output = model(features, mask)
    assert output.shape == (2, 8, 8)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.mean(dim=(-2, -1)), torch.zeros(2), atol=2e-7)

    labels = torch.randn_like(output)
    loss = quotient_gradient_loss(output, labels)
    loss.total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_invalid_slot_values_cannot_change_prediction() -> None:
    model = _small_model().eval()
    features = torch.randn(1, 64, 12, 16)
    mask = torch.rand(1, 64, 12) > 0.5
    mask[:, :3, :] = False
    changed = torch.where(mask.unsqueeze(-1), features, torch.full_like(features, 1000.0))
    with torch.inference_mode():
        baseline = model(features, mask)
        alternative = model(changed, mask)
    torch.testing.assert_close(baseline, alternative, rtol=0.0, atol=0.0)


def test_predictor_rejects_non_boolean_mask_and_nonfinite_storage() -> None:
    model = _small_model()
    features = torch.zeros(1, 64, 12, 16)
    mask = torch.zeros(1, 64, 12, dtype=torch.bool)
    with pytest.raises(TypeError, match="boolean"):
        model(features, mask.float())
    features[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(features, mask)


def test_oriented_grid_gradient_has_frozen_edge_order() -> None:
    potential = torch.tensor([[[0.0, 1.0], [3.0, 7.0]]])
    gradient = oriented_grid_gradients(potential)
    torch.testing.assert_close(gradient, torch.tensor([[1.0, 4.0, 3.0, 6.0]]))


def test_loss_is_common_shift_invariant_and_exact_at_zero() -> None:
    prediction = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    label = prediction + 99.0
    loss = quotient_gradient_loss(prediction, label)
    assert loss.total.item() == pytest.approx(0.0, abs=1e-6)
    assert loss.maximum_residual.item() == pytest.approx(0.0, abs=1e-6)


def test_checkpoint_rule_uses_tail_then_mean_then_earliest_epoch() -> None:
    base = tuple(float(value) for value in range(24))
    candidates = [
        DevelopmentCheckpoint(epoch=3, session_max_residuals_m=base),
        DevelopmentCheckpoint(epoch=2, session_max_residuals_m=base),
        DevelopmentCheckpoint(
            epoch=1,
            session_max_residuals_m=tuple(value + 1.0 for value in base),
        ),
    ]
    selected = select_development_checkpoint(candidates)
    assert selected.epoch == 2
    assert selected.tail_mean_m == pytest.approx(sum((23, 22, 21, 20, 19)) / 5)
    assert selected.mean_m == pytest.approx(sum(base) / 24)


@pytest.mark.parametrize("tail_fraction", [0.0, -0.1, 1.1, math.inf])
def test_checkpoint_rule_rejects_invalid_tail_fraction(tail_fraction: float) -> None:
    candidate = DevelopmentCheckpoint(epoch=1, session_max_residuals_m=(0.0,) * 24)
    with pytest.raises(ValueError, match="tail_fraction"):
        select_development_checkpoint([candidate], tail_fraction=tail_fraction)
