from __future__ import annotations

import numpy as np
import pytest
import torch

from aorta_drive.calibration.heterogeneous_proxy import (
    CONTROL_NAMES,
    calibrate_one_control_session_scores,
    calibrate_control_session_scores,
    canonical_grid_edges,
)
from aorta_drive.models.quotient_field import oriented_grid_gradients


def test_canonical_grid_edges_match_model_gradient_order() -> None:
    potential = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4)
    sources, targets = canonical_grid_edges(4)
    flattened = potential.numpy().reshape(-1)
    endpoint_differences = flattened[targets] - flattened[sources]
    model_differences = oriented_grid_gradients(potential).numpy().reshape(-1)
    np.testing.assert_array_equal(endpoint_differences, model_differences)


def test_proxy_controls_receive_independent_rank_16_thresholds() -> None:
    scores = {
        name: tuple(float(index + offset) for index in range(16))
        for offset, name in enumerate(CONTROL_NAMES)
    }
    calibrated = calibrate_control_session_scores(
        scores,
        alpha=0.10,
        expected_sessions=16,
    )
    assert set(calibrated) == set(CONTROL_NAMES)
    for offset, name in enumerate(CONTROL_NAMES):
        assert calibrated[name].rank == 16
        assert calibrated[name].threshold == pytest.approx(15.0 + offset)


@pytest.mark.parametrize(
    "bad_scores",
    [
        tuple(float(index) for index in range(15)),
        tuple([0.0] * 15 + [float("inf")]),
        tuple([0.0] * 15 + [-1.0]),
    ],
)
def test_proxy_calibration_rejects_missing_or_invalid_sessions(
    bad_scores: tuple[float, ...],
) -> None:
    scores = {name: tuple(float(index) for index in range(16)) for name in CONTROL_NAMES}
    scores["learned"] = bad_scores
    with pytest.raises(ValueError):
        calibrate_control_session_scores(scores, alpha=0.10, expected_sessions=16)


def test_proxy_calibration_rejects_control_filtering() -> None:
    scores = {name: tuple(float(index) for index in range(16)) for name in CONTROL_NAMES}
    del scores["evidence_only"]
    with pytest.raises(ValueError, match="exactly"):
        calibrate_control_session_scores(scores, alpha=0.10, expected_sessions=16)


def test_single_control_calibration_supports_explicit_invalid_variant_handling() -> None:
    calibration = calibrate_one_control_session_scores(
        tuple(float(index) for index in range(16)),
        alpha=0.10,
        expected_sessions=16,
        control_name="learned",
    )
    assert calibration.rank == 16
    assert calibration.threshold == 15.0
