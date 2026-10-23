"""Tests for the deterministic source-only kinematic candidate library."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aorta_drive.candidates import generate_kinematic_candidates


def _candidate(batch, acceleration: float, curvature: float):
    return next(
        candidate
        for candidate in batch.candidates
        if candidate.control.acceleration_mps2 == acceleration
        and candidate.control.curvature_per_m == curvature
    )


def test_straight_constant_speed_has_exact_samples() -> None:
    batch = generate_kinematic_candidates(5.0)
    candidate = _candidate(batch, 0.0, 0.0)
    assert np.allclose(candidate.poses[:, 0], np.arange(1, 9) * 2.5, atol=1e-14)
    assert np.array_equal(candidate.poses[:, 1:], np.zeros((8, 2)))
    assert candidate.terminal_speed_mps == 5.0


def test_constant_curvature_matches_analytic_circle() -> None:
    batch = generate_kinematic_candidates(5.0)
    candidate = _candidate(batch, 0.0, 0.04)
    distance = 20.0
    expected_heading = 0.04 * distance
    assert candidate.poses[-1, 0] == pytest.approx(math.sin(expected_heading) / 0.04)
    assert candidate.poses[-1, 1] == pytest.approx((1.0 - math.cos(expected_heading)) / 0.04)
    assert candidate.poses[-1, 2] == pytest.approx(expected_heading)


def test_braking_stops_without_reverse_and_deduplicates_stationary_controls() -> None:
    batch = generate_kinematic_candidates(0.5)
    candidate = _candidate(batch, -1.0, 0.0)
    assert candidate.terminal_speed_mps == 0.0
    assert np.all(np.diff(candidate.poses[:, 0]) >= 0.0)
    assert candidate.poses[-1, 0] == pytest.approx(0.125)
    stationary = generate_kinematic_candidates(0.0)
    assert stationary.rejection_counts["duplicate"] > 0


def test_dynamic_envelopes_reject_high_speed_controls() -> None:
    batch = generate_kinematic_candidates(19.0)
    assert batch.rejection_counts["speed_envelope"] == 7
    assert batch.rejection_counts["lateral_acceleration_envelope"] > 0
    assert all(
        max(19.0, candidate.terminal_speed_mps) ** 2
        * abs(candidate.control.curvature_per_m)
        <= 4.0
        for candidate in batch.candidates
    )


def test_custom_longitudinal_acceleration_beyond_envelope_is_rejected() -> None:
    batch = generate_kinematic_candidates(
        5.0,
        accelerations_mps2=(-3.0, -4.0),
        curvatures_per_m=(0.0,),
    )
    assert len(batch.candidates) == 1
    assert batch.candidates[0].control.acceleration_mps2 == -3.0
    assert batch.rejection_counts["longitudinal_acceleration_envelope"] == 1


def test_generation_is_deterministic_and_provenance_is_unique() -> None:
    first = generate_kinematic_candidates(7.0)
    second = generate_kinematic_candidates(7.0)
    assert [candidate.control.provenance for candidate in first.candidates] == [
        candidate.control.provenance for candidate in second.candidates
    ]
    assert len({candidate.control.provenance for candidate in first.candidates}) == len(
        first.candidates
    )
    for left, right in zip(first.candidates, second.candidates, strict=True):
        assert np.array_equal(left.poses, right.poses)


@pytest.mark.parametrize("speed", [-0.1, 20.1, math.nan])
def test_invalid_initial_speed_fails_closed(speed: float) -> None:
    with pytest.raises(ValueError):
        generate_kinematic_candidates(speed)
