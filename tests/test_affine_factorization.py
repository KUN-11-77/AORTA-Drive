import numpy as np
import pytest

from aorta_drive.operators.affine import factorize_margin_panel


def test_factorization_reconstructs_non_degenerate_panel() -> None:
    margins = np.array([-0.4, 0.1, 0.7, 1.2])
    operator = factorize_margin_panel(margins)

    assert not operator.degenerate
    np.testing.assert_allclose(operator.compose(), margins, atol=1e-12)
    assert np.dot(operator.weights, operator.relative_shape) == pytest.approx(0.0)
    assert np.dot(operator.weights, operator.relative_shape**2) == pytest.approx(1.0)


def test_factorization_marks_constant_panel_degenerate() -> None:
    margins = np.full(4, 0.25)
    operator = factorize_margin_panel(margins)

    assert operator.degenerate
    np.testing.assert_allclose(operator.relative_shape, 0.0)
    np.testing.assert_allclose(operator.compose(), margins)


def test_factorization_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        factorize_margin_panel([0.0, 1.0], weights=[1.0, 0.0])
