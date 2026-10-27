"""Data-free counterexamples for R006 task-pullback rank geometry."""

import torch


def test_nested_projector_can_increase_planner_pullback_residual() -> None:
    """A larger nested rank need not improve task coverage after a Jacobian."""

    # The third direction is planner-null.  Rank 1 removes it and preserves the
    # cancellation between the first two channels.  Rank 2 additionally removes
    # channel 1, destroying that cancellation and increasing output residual.
    delta = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    jacobian = torch.tensor([[1.0, -0.8, 0.0]], dtype=torch.float64)
    rank_one = torch.diag(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    rank_two = torch.diag(torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64))

    residual_one = jacobian @ ((torch.eye(3) - rank_one) @ delta)
    residual_two = jacobian @ ((torch.eye(3) - rank_two) @ delta)

    assert torch.allclose(rank_two @ rank_one, rank_one)
    assert torch.sum(residual_two.square()) > torch.sum(residual_one.square())
    assert torch.allclose(
        torch.sum(residual_one.square()), torch.tensor(0.04, dtype=torch.float64)
    )
    assert torch.allclose(
        torch.sum(residual_two.square()), torch.tensor(0.64, dtype=torch.float64)
    )


def test_pullback_residual_increment_identity_is_exact() -> None:
    """Verify the identity used by the pre-result interpretation contract."""

    delta = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    jacobian = torch.tensor([[1.0, -0.8, 0.0]], dtype=torch.float64)
    projector = torch.diag(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    increment = torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    identity = torch.eye(3, dtype=torch.float64)

    a = jacobian @ ((identity - projector) @ delta)
    b = jacobian @ (increment @ delta)
    left = torch.sum((a - b).square()) - torch.sum(a.square())
    right = torch.sum(b.square()) - 2.0 * torch.sum(a * b)

    assert torch.allclose(left, right)
    assert left > 0.0
