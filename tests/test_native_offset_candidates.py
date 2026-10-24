"""Qualification tests for the endpoint-invariant native-conditioned provider."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from aorta_drive.candidates import (
    endpoint_invariant_bump,
    generate_native_offset_candidates,
)
from aorta_drive.candidates.native_offset import _certify_nonpositive


TIMES = np.arange(9, dtype=np.float64) * 0.5


def _straight(speed_mps: float = 5.0) -> np.ndarray:
    return np.column_stack((speed_mps * TIMES, np.zeros(9), np.zeros(9)))


def _circle(radius_m: float = 50.0, speed_mps: float = 5.0, sign: float = 1.0) -> np.ndarray:
    arc = speed_mps * TIMES
    angle = sign * arc / radius_m
    return np.column_stack(
        (
            radius_m * np.sin(arc / radius_m),
            sign * radius_m * (1.0 - np.cos(arc / radius_m)),
            angle,
        )
    )


def test_bump_has_registered_endpoint_and_peak_values() -> None:
    values = endpoint_invariant_bump(np.array([0.0, 0.5, 1.0]))
    assert np.array_equal(values, np.array([0.0, 1.0, 0.0]))


@pytest.mark.parametrize("native", [_straight(), _circle(), _circle(sign=-1.0)])
def test_final_segment_is_bit_exact_and_mid_horizon_differs(native: np.ndarray) -> None:
    batch = generate_native_offset_candidates(native, amplitudes_m=(-0.5, 0.5))
    assert len(batch.candidates) == 2
    for candidate in batch.candidates:
        full = np.vstack((np.zeros((1, 3)), candidate.poses))
        assert np.array_equal(full[7:], native[7:])
        assert not np.array_equal(full[1:7, :2], native[1:7, :2])
        assert candidate.bernstein_subdivision_count >= 0


def test_straight_offset_heading_comes_from_perturbed_tangent() -> None:
    native = _straight()
    candidate = generate_native_offset_candidates(native, amplitudes_m=(0.5,)).candidates[0]
    assert np.any(np.abs(candidate.poses[:6, 2]) > 1e-4)
    assert candidate.poses[-1, 2] == native[-1, 2]


def test_small_terminal_displacement_fails_closed() -> None:
    native = _straight(speed_mps=0.2)
    batch = generate_native_offset_candidates(native)
    assert not batch.candidates
    assert batch.rejection_counts["native_reference_displacement"] == 6


def test_excessive_offset_is_rejected_by_continuous_envelope() -> None:
    batch = generate_native_offset_candidates(_straight(), amplitudes_m=(20.0,))
    assert not batch.candidates
    assert batch.rejection_counts["continuous_dynamic_envelope"] == 1


def test_bernstein_screen_accepts_strictly_negative_interior_maximum() -> None:
    passed, _ = _certify_nonpositive(
        Polynomial([-0.3, 1.0, -1.0]), strict_margin=1e-9, maximum_depth=24
    )
    assert passed


def test_bernstein_screen_rejects_positive_interior_counterexample() -> None:
    passed, _ = _certify_nonpositive(
        Polynomial([-0.2, 1.0, -1.0]), strict_margin=1e-9, maximum_depth=24
    )
    assert not passed


def test_provider_is_deterministic_with_unique_provenance() -> None:
    first = generate_native_offset_candidates(_circle(), amplitudes_m=(-0.5, 0.5))
    second = generate_native_offset_candidates(_circle(), amplitudes_m=(-0.5, 0.5))
    assert [item.control.provenance for item in first.candidates] == [
        item.control.provenance for item in second.candidates
    ]
    for left, right in zip(first.candidates, second.candidates, strict=True):
        assert np.array_equal(left.poses, right.poses)


@pytest.mark.parametrize(
    "native",
    [
        np.full((9, 3), math.nan),
        np.ones((9, 3)),
        np.zeros((2, 3)),
    ],
)
def test_invalid_native_input_is_rejected(native: np.ndarray) -> None:
    with pytest.raises(ValueError):
        generate_native_offset_candidates(native)
