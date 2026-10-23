"""R002: deterministic geometry and map-error lifting checks."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from aorta_drive.certification.geometry import (
    AxisAlignedRectangle,
    ContinuousCorrection,
    certify_rectangle_internal_net,
)


def _cartesian_points(xs: list[float], ys: list[float]) -> np.ndarray:
    return np.asarray(list(itertools.product(xs, ys)), dtype=np.float64)


def _linear_infimum(rectangle: AxisAlignedRectangle, gradient: np.ndarray, offset: float) -> float:
    x = rectangle.x_min if gradient[0] >= 0 else rectangle.x_max
    y = rectangle.y_min if gradient[1] >= 0 else rectangle.y_max
    return float(gradient @ np.array([x, y]) + offset)


def test_rectangle_grid_has_exact_internal_covering_radius() -> None:
    rectangle = AxisAlignedRectangle(0.0, 2.0, -1.0, 1.0)
    points = _cartesian_points([0.0, 1.0, 2.0], [-1.0, 0.0, 1.0])
    certificate = certify_rectangle_internal_net(rectangle, points)
    assert certificate.point_count == 9
    assert np.isclose(certificate.x_radius, 0.5)
    assert np.isclose(certificate.y_radius, 0.5)
    assert np.isclose(certificate.h, math.sqrt(0.5))


def test_checker_rejects_outside_duplicate_and_incomplete_grids() -> None:
    rectangle = AxisAlignedRectangle(0.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="inside"):
        certify_rectangle_internal_net(rectangle, [[0.0, 0.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="duplicate-free"):
        certify_rectangle_internal_net(rectangle, [[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="complete Cartesian"):
        certify_rectangle_internal_net(rectangle, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])


def test_continuous_bound_is_sound_for_adversarial_linear_teachers() -> None:
    rng = np.random.default_rng(20261023)
    challenger = AxisAlignedRectangle(-1.0, 1.5, -0.5, 2.0)
    native = AxisAlignedRectangle(-2.0, 0.5, -1.5, 0.75)
    challenger_points = _cartesian_points([-0.8, 0.2, 1.2], [-0.3, 0.7, 1.7])
    native_points = _cartesian_points([-1.8, -0.7, 0.3], [-1.3, -0.2, 0.6])
    h = certify_rectangle_internal_net(challenger, challenger_points).h
    epsilon_challenger = 0.07
    epsilon_native = 0.11
    correction = ContinuousCorrection(h, epsilon_challenger, epsilon_native, 0.0)

    for _ in range(500):
        gradient = rng.normal(size=2)
        gradient /= max(1.0, np.linalg.norm(gradient))
        offset = float(rng.normal())
        challenger_labels = (
            challenger_points @ gradient
            + offset
            + rng.uniform(-epsilon_challenger, epsilon_challenger, challenger_points.shape[0])
        )
        native_labels = (
            native_points @ gradient
            + offset
            + rng.uniform(-epsilon_native, epsilon_native, native_points.shape[0])
        )
        discrete_contrast = float(np.min(challenger_labels) - np.min(native_labels))
        certified = correction.apply(discrete_contrast)
        true_continuous = _linear_infimum(challenger, gradient, offset) - _linear_infimum(
            native, gradient, offset
        )
        assert certified <= true_continuous + 1e-12


def test_numerical_optimism_is_subtracted_in_the_safe_direction() -> None:
    mathematical_bound = 0.25
    optimism = 1e-6
    computed_bound = mathematical_bound + optimism
    correction = ContinuousCorrection(0.0, 0.0, 0.0, optimism)
    assert correction.apply(computed_bound) <= mathematical_bound
