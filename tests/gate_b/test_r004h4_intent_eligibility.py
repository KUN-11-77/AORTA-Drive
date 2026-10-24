"""R004H4 preflight: frozen intent/progress/deviation qualification."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aorta_drive.evaluation.h4 import (
    IntentEligibilityLimits,
    audit_endpoint_invariant,
    qualify_intent_match,
    terminal_segment_average_speed,
)


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


def test_terminal_segment_average_speed_uses_only_final_position_interval() -> None:
    poses = np.array([[0.0, 0.0, 7.0], [3.0, 4.0, -2.0], [3.0, 8.0, 1.0]])
    assert terminal_segment_average_speed(poses, interval_s=0.5) == 8.0


@pytest.mark.parametrize("interval", [0.0, -0.5, math.inf, math.nan])
def test_terminal_segment_average_speed_rejects_invalid_interval(interval: float) -> None:
    with pytest.raises(ValueError):
        terminal_segment_average_speed(np.zeros((2, 3)), interval_s=interval)


def test_endpoint_invariant_accepts_structural_equality_with_one_ulp_progress_roundoff() -> None:
    native = np.array([[0.0, 0.0, 0.0], [17.3, -4.2, 0.4], [31.7, 8.9, 0.7]])
    outcome = qualify_intent_match(
        native,
        native,
        challenger_terminal_speed_mps=5.0,
        native_terminal_speed_mps=5.0,
        limits=LIMITS,
    )
    audited = audit_endpoint_invariant(
        native, native, outcome, numeric_tolerance=64.0 * np.finfo(np.float64).eps
    )
    assert audited.verified
    assert audited.final_segment_bit_exact
    assert audited.maximum_numeric_residual <= 64.0 * np.finfo(np.float64).eps


def test_endpoint_invariant_rejects_nonidentical_final_segment() -> None:
    native = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    challenger = native.copy()
    challenger[-2, 1] = 0.1
    outcome = qualify_intent_match(
        challenger,
        native,
        challenger_terminal_speed_mps=5.0,
        native_terminal_speed_mps=5.0,
        limits=LIMITS,
    )
    audited = audit_endpoint_invariant(
        challenger,
        native,
        outcome,
        numeric_tolerance=64.0 * np.finfo(np.float64).eps,
    )
    assert not audited.verified
    assert not audited.final_segment_bit_exact


def test_origin_and_time_contracts_are_exact() -> None:
    with pytest.raises(ValueError, match="origin"):
        qualify_intent_match(
            np.array([[0.1, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            _trajectory(10.0),
            challenger_terminal_speed_mps=5.0,
            native_terminal_speed_mps=5.0,
            limits=LIMITS,
        )
