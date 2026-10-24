"""Proof-carrying ego-local to global coordinate transforms.

The input binary64 values are interpreted as exact real numbers.  Sine and
cosine are enclosed with exact-rational Taylor polynomials and a Lagrange
remainder, so the certificate does not assume a correctly rounded libm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np

from aorta_drive.certification.geometry import Pose2D

_TAYLOR_DEGREE: Final = 40
_MAX_ABS_HEADING: Final = Fraction(4)


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(value)


@dataclass(frozen=True)
class RationalInterval:
    """Closed exact-rational interval."""

    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("interval endpoints are reversed")

    def add(self, other: RationalInterval) -> RationalInterval:
        return RationalInterval(self.lower + other.lower, self.upper + other.upper)

    def scale(self, scalar: Fraction) -> RationalInterval:
        endpoints = (scalar * self.lower, scalar * self.upper)
        return RationalInterval(min(endpoints), max(endpoints))


@dataclass(frozen=True)
class CoordinateTransformCertificate:
    """Stored global point and radius enclosing the exact rigid transform."""

    x: float
    y: float
    error_radius: float
    x_interval: RationalInterval
    y_interval: RationalInterval
    sine_interval: RationalInterval
    cosine_interval: RationalInterval
    taylor_degree: int = _TAYLOR_DEGREE


def _sqrt_fraction_upper(value: Fraction) -> float:
    if value < 0:
        raise ValueError("cannot take the square root of a negative value")
    if value == 0:
        return 0.0
    estimate = math.sqrt(float(value))
    if not math.isfinite(estimate):
        raise ValueError("transform error exceeds finite binary64 range")
    while _fraction(estimate) ** 2 < value:
        estimate = math.nextafter(estimate, math.inf)
    while True:
        predecessor = math.nextafter(estimate, -math.inf)
        if predecessor <= 0 or _fraction(predecessor) ** 2 < value:
            return estimate
        estimate = predecessor


def _sin_cos_intervals(angle: Fraction) -> tuple[RationalInterval, RationalInterval]:
    """Enclose exact sin/cos on the registered heading domain."""

    if abs(angle) > _MAX_ABS_HEADING:
        raise ValueError("absolute heading exceeds the certified 4-radian domain")
    sine_polynomial = Fraction(0)
    cosine_polynomial = Fraction(0)
    power = Fraction(1)
    factorial = 1
    for degree in range(_TAYLOR_DEGREE + 1):
        if degree > 0:
            power *= angle
            factorial *= degree
        term = power / factorial
        phase = degree % 4
        if phase == 0:
            cosine_polynomial += term
        elif phase == 1:
            sine_polynomial += term
        elif phase == 2:
            cosine_polynomial -= term
        else:
            sine_polynomial -= term
    remainder = abs(angle) ** (_TAYLOR_DEGREE + 1) / math.factorial(_TAYLOR_DEGREE + 1)
    return (
        RationalInterval(sine_polynomial - remainder, sine_polynomial + remainder),
        RationalInterval(cosine_polynomial - remainder, cosine_polynomial + remainder),
    )


def _stored_midpoint(interval: RationalInterval) -> float:
    value = float((interval.lower + interval.upper) / 2)
    if not math.isfinite(value):
        raise ValueError("transformed coordinate is outside finite binary64 range")
    return value


def certify_ego_to_global(
    local_point: tuple[float, float],
    ego_rear_axle: Pose2D,
) -> CoordinateTransformCertificate:
    """Certify a 2D rigid transform without trusting floating trigonometry.

    The exact target is

    ``global = translation + R(heading) @ local``.

    The returned binary64 point is the midpoint of an exact-rational enclosure.
    ``error_radius`` is an outward-rounded Euclidean radius containing every
    point in that enclosure and therefore the exact transformed point.
    """

    if len(local_point) != 2 or not all(np.isfinite(value) for value in local_point):
        raise ValueError("local point must contain two finite coordinates")
    qx, qy = map(_fraction, local_point)
    tx = _fraction(ego_rear_axle.x)
    ty = _fraction(ego_rear_axle.y)
    sine, cosine = _sin_cos_intervals(_fraction(ego_rear_axle.yaw))
    x_interval = RationalInterval(tx, tx).add(cosine.scale(qx)).add(sine.scale(-qy))
    y_interval = RationalInterval(ty, ty).add(sine.scale(qx)).add(cosine.scale(qy))
    stored_x = _stored_midpoint(x_interval)
    stored_y = _stored_midpoint(y_interval)
    stored_x_fraction = _fraction(stored_x)
    stored_y_fraction = _fraction(stored_y)
    x_error = max(
        abs(stored_x_fraction - x_interval.lower),
        abs(stored_x_fraction - x_interval.upper),
    )
    y_error = max(
        abs(stored_y_fraction - y_interval.lower),
        abs(stored_y_fraction - y_interval.upper),
    )
    error_radius = _sqrt_fraction_upper(x_error * x_error + y_error * y_error)
    return CoordinateTransformCertificate(
        x=stored_x,
        y=stored_y,
        error_radius=error_radius,
        x_interval=x_interval,
        y_interval=y_interval,
        sine_interval=sine,
        cosine_interval=cosine,
    )


def certify_global_to_ego(
    global_point: tuple[float, float],
    ego_rear_axle: Pose2D,
) -> CoordinateTransformCertificate:
    """Certify the inverse rigid transform into the current ego frame.

    The exact target is ``local = R(-heading) @ (global - translation)``.
    This is the transform needed to express a cached global planner trajectory
    on AORTA's fixed ego-centric graph without trusting platform trigonometry.
    """

    if len(global_point) != 2 or not all(np.isfinite(value) for value in global_point):
        raise ValueError("global point must contain two finite coordinates")
    dx = _fraction(global_point[0]) - _fraction(ego_rear_axle.x)
    dy = _fraction(global_point[1]) - _fraction(ego_rear_axle.y)
    sine, cosine = _sin_cos_intervals(_fraction(ego_rear_axle.yaw))
    x_interval = cosine.scale(dx).add(sine.scale(dy))
    y_interval = cosine.scale(dy).add(sine.scale(-dx))
    stored_x = _stored_midpoint(x_interval)
    stored_y = _stored_midpoint(y_interval)
    stored_x_fraction = _fraction(stored_x)
    stored_y_fraction = _fraction(stored_y)
    x_error = max(
        abs(stored_x_fraction - x_interval.lower),
        abs(stored_x_fraction - x_interval.upper),
    )
    y_error = max(
        abs(stored_y_fraction - y_interval.lower),
        abs(stored_y_fraction - y_interval.upper),
    )
    error_radius = _sqrt_fraction_upper(x_error * x_error + y_error * y_error)
    return CoordinateTransformCertificate(
        x=stored_x,
        y=stored_y,
        error_radius=error_radius,
        x_interval=x_interval,
        y_interval=y_interval,
        sine_interval=sine,
        cosine_interval=cosine,
    )
