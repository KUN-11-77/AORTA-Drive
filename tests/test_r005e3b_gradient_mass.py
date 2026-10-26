from __future__ import annotations

import numpy as np
import pytest
import torch

from diagnose_r005e3b_gradient_mass import (
    _gradient_geometry,
    _score_diagnostics,
    _score_gradient_arrays,
)


def test_score_gradient_identifies_positive_and_nonpositive_challengers() -> None:
    prediction = np.asarray([[0.0, -1.0, 0.2]], dtype=np.float32)
    target = np.asarray([[0.0, 0.3, -0.2]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    _, gradient, positive, nonpositive = _score_gradient_arrays(
        prediction, target, mask, temperature=0.125
    )
    assert positive.tolist() == [[False, True, False]]
    assert nonpositive.tolist() == [[False, False, True]]
    assert gradient[0, 1] < 0.0
    assert gradient[0, 2] > 0.0


def test_score_diagnostic_reports_sparse_opportunity_and_gradient_mass() -> None:
    prediction = np.asarray([[0.0, -1.0, 0.2], [0.0, -0.3, -0.4]], dtype=np.float32)
    target = np.asarray([[0.0, 0.3, -0.2], [0.0, -0.1, -0.2]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    result = _score_diagnostics(prediction, target, mask, temperature=0.125)
    assert result["anchor_count"] == 2
    assert result["valid_challenger_count"] == 4
    assert result["opportunity_anchor_count"] == 1
    assert result["opportunity_anchor_fraction"] == pytest.approx(0.5)
    assert result["positive_challenger_fraction"] == pytest.approx(0.25)
    assert result["positive"]["count"] == 1
    assert result["nonpositive"]["count"] == 3
    assert result["positive_raise_direction_fraction"] == 1.0


def test_parameter_gradient_geometry_exposes_exact_conflict() -> None:
    result = _gradient_geometry(
        {"weight": torch.tensor([1.0, 0.0])},
        {"weight": torch.tensor([-2.0, 0.0])},
    )
    assert result["positive_parameter_gradient_norm"] == pytest.approx(1.0)
    assert result["nonpositive_parameter_gradient_norm"] == pytest.approx(2.0)
    assert result["nonpositive_to_positive_norm_ratio"] == pytest.approx(2.0)
    assert result["positive_nonpositive_cosine"] == pytest.approx(-1.0)
    assert result["combined_parameter_gradient_norm"] == pytest.approx(1.0)
