"""R004H0: structural qualification for the frozen-potential scale interface."""

from __future__ import annotations

import inspect
import math

import pytest
import torch
from torch import Tensor
import torch.nn.functional as functional

from aorta_drive.models.heterogeneous_envelope import (
    BoundedEdgeScaleHead,
    HeterogeneousEnvelopePredictor,
    bounded_laplace_scale_loss,
)
from aorta_drive.models.quotient_field import QuotientFieldPredictor


def _small_predictor() -> QuotientFieldPredictor:
    torch.manual_seed(20261024)
    return QuotientFieldPredictor(
        input_dimension=8,
        token_dimension=8,
        grid_size=8,
        roi_half_extent_m=4.0,
        spatial_widths=(16, 24, 32),
    )


def _inputs() -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(20261025)
    features = torch.randn(2, 64, 12, 8, generator=generator)
    mask = torch.rand(2, 64, 12, generator=generator) > 0.25
    mask[:, 0, :] = False
    return features, mask


def _legacy_forward(
    model: QuotientFieldPredictor,
    node_slot_features: Tensor,
    node_slot_mask: Tensor,
) -> Tensor:
    """Frozen pre-refactor forward expression used as an independent replay oracle."""

    compute_dtype = model.missing_evidence_token.dtype
    features = node_slot_features.to(dtype=compute_dtype)
    projected = model.feature_projection(features)
    coordinates = model.normalized_coordinates.to(device=features.device, dtype=compute_dtype)
    coordinate_context = model.coordinate_projection(coordinates)
    slot_context = model.camera_embedding(model.slot_camera_indices)
    slot_context = slot_context + model.history_embedding(model.slot_history_indices)
    logits = model.attention_score(
        projected + slot_context[None, None, :, :] + coordinate_context[None, :, None, :]
    ).squeeze(-1)
    weights = model._masked_softmax(logits, node_slot_mask)
    fused = torch.sum(weights.unsqueeze(-1) * projected, dim=2)
    has_evidence = node_slot_mask.any(dim=-1)
    missing = model.missing_evidence_token[None, None, :].expand_as(fused)
    fused = torch.where(has_evidence.unsqueeze(-1), fused, missing)
    evidence_fraction = node_slot_mask.to(compute_dtype).mean(dim=-1, keepdim=True)
    fused = model.fused_norm(
        fused
        + coordinate_context[None, :, :]
        + model.evidence_count_projection(evidence_fraction)
    )
    batch = fused.shape[0]
    spatial = fused.transpose(1, 2).reshape(
        batch,
        model.token_dimension,
        model.grid_size,
        model.grid_size,
    )
    level0 = model.encoder0(model.spatial_stem(spatial))
    level1 = model.encoder1(model.down1(level0))
    level2 = model.bottleneck(model.down2(level1))
    decoded1 = functional.interpolate(
        level2,
        size=level1.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    decoded1 = model.decoder1(model.up1(decoded1) + level1)
    decoded0 = functional.interpolate(
        decoded1,
        size=level0.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    decoded0 = model.decoder0(model.up0(decoded0) + level0)
    potential = model.output_head(decoded0).squeeze(1)
    return potential - potential.mean(dim=(-2, -1), keepdim=True)


def test_spatial_refactor_exactly_replays_the_legacy_potential_expression() -> None:
    model = _small_predictor().eval()
    features, mask = _inputs()
    with torch.no_grad():
        expected = _legacy_forward(model, features, mask)
        actual = model(features, mask)
        spatial = model.spatial_features(features, mask)
        from_spatial = model.potential_from_spatial_features(spatial)
    assert torch.equal(actual, expected)
    assert torch.equal(from_spatial, expected)
    assert torch.equal(actual.mean(dim=(-2, -1)), torch.zeros(2))


def test_default_scale_head_has_exactly_194_trainable_parameters() -> None:
    head = BoundedEdgeScaleHead()
    assert sum(parameter.numel() for parameter in head.parameters()) == 194
    assert all(parameter.requires_grad for parameter in head.parameters())
    assert head.edge_count == 8064


def test_scale_head_initializes_every_edge_to_one_metre_within_registered_tolerance() -> None:
    head = BoundedEdgeScaleHead(input_channels=3, grid_size=4)
    output = head(torch.randn(2, 3, 4, 4))
    assert output.shape == (2, 24)
    assert torch.max(torch.abs(output - 1.0)).item() <= 1.0e-6
    assert torch.all(output >= 2.0**-10)
    assert torch.all(output <= 2.0**4)


def test_scale_head_uses_all_rightward_then_all_downward_edge_order() -> None:
    head = BoundedEdgeScaleHead(
        input_channels=1,
        grid_size=3,
        scale_min_m=0.5,
        scale_max_m=2.0,
        initial_scale_m=1.0,
    )
    with torch.no_grad():
        head.projection.weight[0, 0, 0, 0] = 1.0
        head.projection.weight[1, 0, 0, 0] = -1.0
        head.projection.bias.zero_()
    spatial = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    output = head(spatial)
    logits = head.projection(spatial)
    full = 0.5 + 1.5 * torch.sigmoid(logits)
    expected_right = full[:, 0, :, :-1].reshape(1, -1)
    expected_down = full[:, 1, :-1, :].reshape(1, -1)
    assert torch.equal(output[:, :6], expected_right)
    assert torch.equal(output[:, 6:], expected_down)


def test_wrapper_freezes_potential_and_only_scale_head_receives_gradients() -> None:
    predictor = _small_predictor()
    model = HeterogeneousEnvelopePredictor(predictor).train()
    features, mask = _inputs()
    output = model(features, mask)
    label = torch.zeros_like(output.potential)
    loss = bounded_laplace_scale_loss(
        output.potential,
        label,
        output.edge_scales_m,
    )
    loss.total.backward()
    assert math.isfinite(float(loss.total.detach()))
    assert not predictor.training
    assert all(not parameter.requires_grad for parameter in predictor.parameters())
    assert all(parameter.grad is None for parameter in predictor.parameters())
    scale_gradients = [parameter.grad for parameter in model.scale_head.parameters()]
    assert all(gradient is not None for gradient in scale_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in scale_gradients if gradient is not None)
    assert any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in scale_gradients
        if gradient is not None
    )


def test_wrapper_forward_signature_has_no_candidate_or_planner_input() -> None:
    parameters = tuple(inspect.signature(HeterogeneousEnvelopePredictor.forward).parameters)
    assert parameters == ("self", "node_slot_features", "node_slot_mask")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale_min_m": 0.0},
        {"scale_min_m": 1.0, "initial_scale_m": 1.0},
        {"scale_max_m": 1.0, "initial_scale_m": 1.0},
        {"scale_min_m": 2.0, "scale_max_m": 1.0},
    ],
)
def test_scale_head_rejects_invalid_registered_bounds(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BoundedEdgeScaleHead(**kwargs)


def test_scale_loss_rejects_nonpositive_or_misaligned_scales() -> None:
    potential = torch.zeros(1, 4, 4)
    label = torch.zeros_like(potential)
    with pytest.raises(ValueError, match="match"):
        bounded_laplace_scale_loss(potential, label, torch.ones(1, 2))
    with pytest.raises(ValueError, match="positive"):
        bounded_laplace_scale_loss(potential, label, torch.zeros(1, 24))
