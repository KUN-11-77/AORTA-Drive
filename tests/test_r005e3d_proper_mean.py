from __future__ import annotations

import numpy as np
import pytest
import torch

from aorta_drive.models.full_context_scorer import (
    equal_anchor_challenger_margin_mse,
)
from audit_r005e3d_proper_mean import _classify, _failure_reasons, _summary
from run_r005e3d_proper_mean import _decision_summary


def test_proper_mean_loss_keeps_gradient_on_missed_positive_tail() -> None:
    prediction = torch.tensor([[0.0, -20.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.25]])
    mask = torch.ones_like(prediction, dtype=torch.bool)
    loss = equal_anchor_challenger_margin_mse(prediction, target, mask).total
    loss.backward()
    assert prediction.grad is not None
    assert float(prediction.grad[0, 1]) == pytest.approx(-40.5)


def test_decision_summary_reports_relation_mse_and_gain() -> None:
    prediction = np.asarray([[0.0, 0.3], [0.0, -0.2]], dtype=np.float32)
    target = np.asarray([[0.0, 0.2], [0.0, -0.1]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    summary = _decision_summary(prediction, target, mask, temperature=0.125)
    assert summary["relation_mse"] == pytest.approx(0.01)
    assert summary["mean_gain_over_native"] == pytest.approx(0.1)
    assert summary["beneficial_nonnative_count"] == 1


def test_frozen_gate_passes_only_jointly() -> None:
    prediction = np.asarray([[0.0, 0.3], [0.0, -0.2]], dtype=np.float32)
    target = np.asarray([[0.0, 0.2], [0.0, -0.1]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    summary = _summary(prediction, target, mask)
    assert _classify(summary) == "PASS_PROPER_MEAN_OUTER_TRAIN_QUALIFICATION"
    assert _failure_reasons(summary) == []


def test_failure_label_does_not_claim_a_unique_cause() -> None:
    prediction = np.asarray([[0.0, -1.0], [0.0, -1.0]], dtype=np.float32)
    target = np.asarray([[0.0, 0.2], [0.0, -0.1]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    summary = _summary(prediction, target, mask)
    assert _classify(summary) == "FAIL_FROZEN_QUALIFICATION_GATE"
    assert _failure_reasons(summary) == [
        "NO_POSITIVE_GAIN",
        "NO_BENEFICIAL_NONNATIVE_SELECTION",
        "NO_NONNATIVE_SELECTION",
    ]


def test_conditional_mean_ranking_recovers_expected_utility_order() -> None:
    outcomes = np.asarray(
        [
            [0.0, 1.0, 0.59],
            [0.0, 1.0, 0.59],
            [0.0, 0.0, 0.59],
            [0.0, 0.0, 0.59],
            [0.0, 1.0, 0.59],
        ],
        dtype=np.float64,
    )
    conditional_mean = outcomes.mean(axis=0)
    assert conditional_mean.tolist() == pytest.approx([0.0, 0.6, 0.59])
    assert int(np.argmax(conditional_mean)) == 1
