"""R004H4 preflight: continuous teacher opportunity arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from aorta_drive.certification.opportunity import (
    continuous_teacher_improvement_lower_bound,
)


def test_teacher_improvement_subtracts_cover_and_both_pointwise_errors() -> None:
    result = continuous_teacher_improvement_lower_bound(
        np.array([3.0, 4.0, 1.0, 2.0]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        (0, 1),
        (2, 3),
        challenger_cover_radius_m=0.5,
    )
    assert result.challenger_margin_lower_m <= 2.4
    assert result.native_margin_upper_m >= 1.3
    assert result.improvement_lower_m <= 1.1
    assert result.improvement_lower_m > 1.099999999999


def test_teacher_improvement_is_monotone_in_declared_uncertainty() -> None:
    labels = np.array([3.0, 4.0, 1.0, 2.0])
    exact = continuous_teacher_improvement_lower_bound(
        labels, np.zeros(4), (0, 1), (2, 3), challenger_cover_radius_m=0.0
    )
    uncertain = continuous_teacher_improvement_lower_bound(
        labels, np.full(4, 0.2), (0, 1), (2, 3), challenger_cover_radius_m=0.5
    )
    assert uncertain.improvement_lower_m <= exact.improvement_lower_m


@pytest.mark.parametrize(
    ("errors", "challenger", "native", "radius"),
    [
        (np.array([0.0, -0.1]), (0,), (1,), 0.0),
        (np.zeros(2), (), (1,), 0.0),
        (np.zeros(2), (0,), (), 0.0),
        (np.zeros(2), (0,), (1,), -0.1),
    ],
)
def test_teacher_improvement_fails_closed_on_invalid_contracts(
    errors: np.ndarray,
    challenger: tuple[int, ...],
    native: tuple[int, ...],
    radius: float,
) -> None:
    with pytest.raises(ValueError):
        continuous_teacher_improvement_lower_bound(
            np.array([1.0, 2.0]),
            errors,
            challenger,
            native,
            challenger_cover_radius_m=radius,
        )
