"""R004H4 preflight: frozen intent/progress/deviation qualification."""

from __future__ import annotations

import numpy as np
import pytest

from aorta_drive.evaluation.h4 import IntentEligibilityLimits, qualify_intent_match


LIMITS = IntentEligibilityLimits(
    minimum_native_terminal_displacement_m=2.0,
    minimum_progress_ratio=0.8,
    maximum_progress_ratio=1.2,
    maximum_terminal_cross_track_m=3.5,
    maximum_terminal_heading_difference_rad=0.35,
    maximum_terminal_speed_difference_mps=4.0,
)


def _trajectory(x: float, y: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [x, y, yaw]], dtype=np.float64)


def test_identical_intent_is_eligible() -> None:
    result = qualify_intent_match(
        _trajectory(10.0),
        _trajectory(10.0),
        challenger_terminal_speed_mps=5.0,
        native_terminal_speed_mps=5.0,
        limits=LIMITS,
    )
    assert result.eligible
    assert result.reason == "eligible"
    assert result.progress_ratio == 1.0


@pytest.mark.parametrize(
    ("challenger", "challenger_speed", "reason"),
    [
        (_trajectory(7.0), 5.0, "progress_below_budget"),
        (_trajectory(13.0), 5.0, "progress_above_budget"),
        (_trajectory(10.0, 4.0), 5.0, "cross_track_above_budget"),
        (_trajectory(10.0, 0.0, 0.5), 5.0, "heading_above_budget"),
        (_trajectory(10.0), 10.0, "speed_above_budget"),
    ],
)
def test_each_intent_budget_fails_closed(
    challenger: np.ndarray, challenger_speed: float, reason: str
) -> None:
    result = qualify_intent_match(
        challenger,
        _trajectory(10.0),
        challenger_terminal_speed_mps=challenger_speed,
        native_terminal_speed_mps=5.0,
        limits=LIMITS,
    )
    assert not result.eligible
    assert result.reason == reason


def test_low_progress_native_has_no_well_defined_intent_frame() -> None:
    result = qualify_intent_match(
        _trajectory(1.0),
        _trajectory(1.0),
        challenger_terminal_speed_mps=0.0,
        native_terminal_speed_mps=0.0,
        limits=LIMITS,
    )
    assert not result.eligible
    assert result.reason == "native_progress_too_small"


def test_origin_and_time_contracts_are_exact() -> None:
    with pytest.raises(ValueError, match="origin"):
        qualify_intent_match(
            np.array([[0.1, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            _trajectory(10.0),
            challenger_terminal_speed_mps=5.0,
            native_terminal_speed_mps=5.0,
            limits=LIMITS,
        )
