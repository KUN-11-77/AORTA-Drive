from __future__ import annotations

import pytest
import torch

from aorta_drive.models.full_context_scorer import (
    FullContextSetScorer,
    _trajectory_features,
    equal_anchor_challenger_margin_mse,
)


def _model() -> FullContextSetScorer:
    torch.manual_seed(7)
    model = FullContextSetScorer(
        visual_input_dimension=8,
        hidden_dimension=32,
        attention_heads=4,
        trajectory_layers=2,
        set_layers=2,
        patch_height=3,
        patch_width=4,
        dropout=0.0,
    )
    return model.eval()


def _inputs() -> tuple[torch.Tensor, ...]:
    batch, candidates, time = 2, 4, 6
    clock = torch.linspace(0.0, 1.0, time)
    poses = torch.zeros(batch, candidates, time, 3)
    for batch_index in range(batch):
        for candidate_index in range(candidates):
            poses[batch_index, candidate_index, :, 0] = 10.0 * clock
            poses[batch_index, candidate_index, :, 1] = candidate_index * clock
            poses[batch_index, candidate_index, :, 2] = 0.1 * candidate_index * clock
    candidate_mask = torch.tensor(
        [[True, True, False, True], [True, False, False, False]], dtype=torch.bool
    )
    previous = poses[:, 0].clone()
    generator = torch.Generator().manual_seed(11)
    visual = torch.randn(batch, 12, 3, 4, 8, generator=generator)
    visual_mask = torch.ones(batch, 12, dtype=torch.bool)
    visual_mask[0, 2] = False
    return poses, candidate_mask, previous, visual, visual_mask


def test_forward_is_finite_bounded_and_native_relative() -> None:
    model = _model()
    output = model(*_inputs())
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, 0], torch.zeros(2))
    assert torch.equal(output[0, 2:3], torch.zeros(1))
    assert torch.equal(output[1, 1:], torch.zeros(3))
    assert bool(torch.all(output.abs() <= 1.0))


def test_trajectory_time_feature_preserves_physical_overlap() -> None:
    current = _trajectory_features(torch.zeros(1, 41, 3))
    previous_overlap = _trajectory_features(torch.zeros(1, 36, 3))
    assert current[0, 0, -1].item() == 0.0
    assert current[0, -1, -1].item() == pytest.approx(1.0)
    assert previous_overlap[0, -1, -1].item() == pytest.approx(0.875)
    assert torch.equal(current[:, :36, -1], previous_overlap[:, :, -1])


def test_invalid_candidate_storage_cannot_change_valid_outputs() -> None:
    model = _model()
    inputs = list(_inputs())
    baseline = model(*inputs)
    poses = inputs[0].clone()
    poses[0, 2] = 99_999.0
    changed = model(poses, *inputs[1:])
    mask = inputs[1]
    assert torch.equal(baseline[mask], changed[mask])


def test_masked_visual_slot_storage_cannot_change_output() -> None:
    model = _model()
    inputs = list(_inputs())
    baseline = model(*inputs)
    visual = inputs[3].clone()
    visual[0, 2] = -99_999.0
    changed = model(inputs[0], inputs[1], inputs[2], visual, inputs[4])
    assert torch.equal(baseline, changed)


def test_challenger_permutation_equivariance() -> None:
    model = _model()
    poses, mask, previous, visual, visual_mask = _inputs()
    original = model(poses, mask, previous, visual, visual_mask)
    permutation = torch.tensor([0, 3, 1, 2])
    permuted = model(
        poses[:, permutation],
        mask[:, permutation],
        previous,
        visual,
        visual_mask,
    )
    assert torch.allclose(permuted, original[:, permutation], atol=1.0e-7, rtol=0.0)


def test_seed_replay_and_identical_arm_parameter_contract() -> None:
    first = _model()
    second = _model()
    assert sum(parameter.numel() for parameter in first.parameters()) == sum(
        parameter.numel() for parameter in second.parameters()
    )
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters()
    ):
        assert torch.equal(first_parameter, second_parameter)
    assert torch.equal(first(*_inputs()), second(*_inputs()))


def test_loss_is_anchor_then_challenger_equal_and_keeps_zero_challenger_anchor() -> (
    None
):
    predicted = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True], [True, False, False]])
    loss = equal_anchor_challenger_margin_mse(predicted, target, mask)
    assert torch.equal(loss.valid_challenger_count, torch.tensor([2, 0]))
    assert torch.allclose(loss.per_anchor, torch.tensor([1.0, 0.0]))
    assert torch.allclose(loss.total, torch.tensor(0.5))


def test_backward_is_finite() -> None:
    model = _model().train()
    poses, mask, previous, visual, visual_mask = _inputs()
    predicted = model(poses, mask, previous, visual, visual_mask)
    target = torch.zeros_like(predicted)
    loss = equal_anchor_challenger_margin_mse(predicted, target, mask).total
    loss.backward()  # type: ignore[no-untyped-call]
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("failure", ["native", "visual", "nan", "shape"])
def test_forward_fails_closed_on_invalid_contract(failure: str) -> None:
    model = _model()
    inputs = list(_inputs())
    if failure == "native":
        inputs[1] = inputs[1].clone()
        inputs[1][0, 0] = False
    elif failure == "visual":
        inputs[4] = torch.zeros_like(inputs[4])
    elif failure == "nan":
        inputs[3] = inputs[3].clone()
        inputs[3][0, 0, 0, 0, 0] = float("nan")
    else:
        inputs[3] = inputs[3][:, :, :, :-1]
    with pytest.raises((ValueError, TypeError)):
        model(*inputs)


def test_loss_rejects_nonzero_native_margin() -> None:
    predicted = torch.tensor([[0.1, 0.0]])
    target = torch.zeros_like(predicted)
    mask = torch.ones_like(predicted, dtype=torch.bool)
    with pytest.raises(ValueError, match="native"):
        equal_anchor_challenger_margin_mse(predicted, target, mask)


def test_constructor_rejects_architecture_drift() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        FullContextSetScorer(visual_cross_attention_layers=2)
    with pytest.raises(ValueError, match="divisible"):
        FullContextSetScorer(hidden_dimension=30, attention_heads=8)
