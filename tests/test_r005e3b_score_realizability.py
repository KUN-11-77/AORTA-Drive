from __future__ import annotations

import numpy as np
import pytest

from audit_r005e3b_score_realizability import _classify, _summary
from run_r005e3b_score_realizability import _decision_summary


def test_gain_decomposition_and_absolute_capture() -> None:
    prediction = np.asarray([[0.0, 0.8, -0.2], [0.0, -0.1, 0.9]], dtype=np.float32)
    target = np.asarray([[0.0, 0.3, -0.1], [0.0, 0.4, 0.2]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    result = _decision_summary(prediction, target, mask, temperature=0.125)
    assert result["mean_gain_over_native"] == pytest.approx(0.25)
    assert result["mean_oracle_gain_over_native"] == pytest.approx(0.35)
    assert result["mean_hard_regret"] == pytest.approx(0.10)
    assert result["absolute_oracle_capture"] == pytest.approx(5.0 / 7.0)
    assert result["beneficial_nonnative_count"] == 2
    assert result["harmful_nonnative_count"] == 0


def test_native_only_cannot_pass_absolute_gain_gate() -> None:
    prediction = np.asarray([[0.0, -1.0], [0.0, -1.0]], dtype=np.float32)
    target = np.asarray([[0.0, 0.2], [0.0, -0.1]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    result = _decision_summary(prediction, target, mask, temperature=0.125)
    assert result["mean_gain_over_native"] == 0.0
    assert result["absolute_oracle_capture"] == 0.0
    assert result["native_choice_fraction"] == 1.0
    assert result["nonnative_choice_count"] == 0


def test_missed_positive_candidate_has_nonzero_identity_score_gradient() -> None:
    prediction = np.asarray([[0.0, -1.0]], dtype=np.float32)
    target = np.asarray([[0.0, 0.2]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    result = _decision_summary(prediction, target, mask, temperature=0.125)
    assert float(result["positive_oracle_score_gradient_abs_min"]) > 0.0


def test_independent_audit_recomputation_and_gate() -> None:
    prediction = np.asarray([[0.0, 0.8, -0.2], [0.0, -0.1, 0.9]], dtype=np.float32)
    target = np.asarray([[0.0, 0.3, -0.1], [0.0, 0.4, 0.2]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    summary = _summary(prediction, target, mask)
    assert summary["mean_gain_over_native"] == pytest.approx(0.25)
    assert summary["absolute_oracle_capture"] == pytest.approx(5.0 / 7.0)
    assert _classify(summary) == "PASS_LINEAR_REALIZABILITY"


def test_independent_audit_rejects_native_fallback() -> None:
    prediction = np.asarray([[0.0, -1.0], [0.0, -1.0]], dtype=np.float32)
    target = np.asarray([[0.0, 0.2], [0.0, -0.1]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    assert _classify(_summary(prediction, target, mask)) == "FAIL_LINEAR_REALIZABILITY"
