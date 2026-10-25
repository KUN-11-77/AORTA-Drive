from __future__ import annotations

import numpy as np

from audit_r005e2_scientific_run import _anchor_metrics, _bootstrap


def test_anchor_metrics_use_ragged_native_relative_domain() -> None:
    prediction = np.asarray([0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0], np.float32)
    target = np.asarray([0.0, -0.1, 0.4, 0.0, 0.0, 0.0, 0.0], np.float32)
    mask = np.asarray([True, True, True, False, False, False, False])
    relation_mse, regret, sign_error, boundary_error = _anchor_metrics(
        prediction, target, mask
    )
    assert np.isclose(relation_mse, (0.4**2 + 0.6**2) / 2)
    assert np.isclose(regret, 0.5)
    assert sign_error == 1.0
    assert boundary_error == 1.0


def test_anchor_metrics_keep_zero_challenger_anchor() -> None:
    prediction = np.zeros(7, np.float32)
    target = np.zeros(7, np.float32)
    mask = np.asarray([True, False, False, False, False, False, False])
    assert _anchor_metrics(prediction, target, mask) == (0.0, 0.0, 0.0, 0.0)


def test_zero_prediction_is_sign_error_for_non_tie_target() -> None:
    prediction = np.zeros(7, np.float32)
    target = np.asarray([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    mask = np.asarray([True, True, False, False, False, False, False])
    assert _anchor_metrics(prediction, target, mask)[2] == 1.0


def test_group_bootstrap_is_deterministic() -> None:
    values = np.linspace(-1.0, 1.0, 42, dtype=np.float64)
    assert _bootstrap(values) == _bootstrap(values)
