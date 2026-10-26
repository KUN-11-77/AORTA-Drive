from __future__ import annotations

import numpy as np
import pytest

from diagnose_r005e3a_collapse import summarize_panel_predictions


def test_native_collapse_has_zero_absolute_oracle_capture() -> None:
    prediction = np.asarray([[0.0, -1.0, -1.0], [0.0, -1.0, 9.0]], dtype=np.float32)
    target = np.asarray([[0.0, 0.2, -0.1], [0.0, -0.2, 8.0]], dtype=np.float32)
    mask = np.asarray([[True, True, True], [True, True, False]])
    summary = summarize_panel_predictions(prediction, target, mask)
    assert summary["native_choice_fraction"] == 1.0
    assert summary["nonnative_choice_count"] == 0
    assert summary["challenger_score_minus_one_fraction"] == 1.0
    assert summary["mean_selected_gain_over_native"] == 0.0
    assert float(summary["mean_oracle_gain_over_native"]) > 0.0
    assert summary["absolute_oracle_capture"] == 0.0


def test_non_native_selection_decomposes_oracle_gain() -> None:
    prediction = np.asarray([[0.0, 0.8, -0.2], [0.0, -0.1, 0.9]], dtype=np.float32)
    target = np.asarray([[0.0, 0.3, -0.1], [0.0, 0.4, 0.2]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    summary = summarize_panel_predictions(prediction, target, mask)
    assert summary["native_choice_fraction"] == 0.0
    assert summary["mean_selected_gain_over_native"] == pytest.approx(0.25)
    assert summary["mean_oracle_gain_over_native"] == pytest.approx(0.35)
    assert summary["mean_hard_regret"] == pytest.approx(0.10)
    assert summary["absolute_oracle_capture"] == pytest.approx(5.0 / 7.0)


def test_invalid_panel_contract_fails_closed() -> None:
    prediction = np.zeros((2, 2), dtype=np.float32)
    target = np.zeros((2, 2), dtype=np.float32)
    mask = np.asarray([[True, True], [False, True]])
    with pytest.raises(ValueError, match="native"):
        summarize_panel_predictions(prediction, target, mask)
