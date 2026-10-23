"""R002D: exact polygon teacher and deterministic map-error ledger."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from aorta_drive.certification import teacher as teacher_module
from aorta_drive.certification.teacher import (
    ExactPolygonTeacher,
    MapErrorLedger,
    PolygonComponent,
    SignedDistanceInterval,
)


def _rectangle(x_min: float, x_max: float, y_min: float, y_max: float) -> PolygonComponent:
    return PolygonComponent(outer=((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))


def _assert_contains(interval: SignedDistanceInterval, exact: float) -> None:
    assert interval.lower <= exact <= interval.upper


def test_rectangle_teacher_has_correct_sign_boundary_and_distance() -> None:
    teacher = ExactPolygonTeacher((_rectangle(0.0, 4.0, 0.0, 3.0),))
    cases = {
        (2.0, 1.0): 1.0,
        (0.0, 2.0): 0.0,
        (-2.0, 1.0): -2.0,
        (5.0, 4.0): -math.sqrt(2.0),
    }
    for point, exact in cases.items():
        _assert_contains(teacher.signed_distance_interval(point), exact)


def test_concave_hole_and_disjoint_union_semantics() -> None:
    concave = PolygonComponent(
        outer=((0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0))
    )
    donut = PolygonComponent(
        outer=((5.0, 0.0), (9.0, 0.0), (9.0, 4.0), (5.0, 4.0)),
        holes=(((6.0, 1.0), (8.0, 1.0), (8.0, 3.0), (6.0, 3.0)),),
    )
    island = _rectangle(11.0, 13.0, 0.0, 2.0)
    teacher = ExactPolygonTeacher((concave, donut, island))
    assert teacher.signed_distance_interval((0.5, 2.0)).lower > 0
    assert teacher.signed_distance_interval((2.0, 2.0)).upper < 0
    assert teacher.signed_distance_interval((5.5, 2.0)).lower > 0
    assert teacher.signed_distance_interval((7.0, 2.0)).upper < 0
    assert teacher.signed_distance_interval((12.0, 1.0)).lower > 0


def test_overlapping_polygon_cover_keeps_internal_boundaries_conservatively() -> None:
    teacher = ExactPolygonTeacher(
        (
            _rectangle(0.0, 2.0, 0.0, 2.0),
            _rectangle(1.0, 3.0, 0.0, 2.0),
        )
    )
    internal_seam = teacher.signed_distance_interval((1.0, 1.0))
    overlap_interior = teacher.signed_distance_interval((1.5, 1.0))
    outside = teacher.signed_distance_interval((4.0, 1.0))
    assert internal_seam.lower == internal_seam.upper == 0.0
    _assert_contains(overlap_interior, 0.5)
    _assert_contains(outside, -1.0)


def test_binary64_interval_encloses_exact_rational_segment_distance() -> None:
    teacher = ExactPolygonTeacher((_rectangle(0.0, 1.0, 0.0, 1.0),))
    point = (1.1, 1.2)
    interval = teacher.signed_distance_interval(point)
    dx = Fraction.from_float(point[0]) - Fraction(1)
    dy = Fraction.from_float(point[1]) - Fraction(1)
    exact_squared = dx * dx + dy * dy
    assert Fraction.from_float(interval.lower) ** 2 >= exact_squared
    assert Fraction.from_float(interval.upper) ** 2 <= exact_squared
    assert interval.lower <= interval.upper < 0


def test_certified_node_label_error_contains_entire_interval() -> None:
    teacher = ExactPolygonTeacher((_rectangle(-1.0, 2.0, -2.0, 3.0),))
    rng = np.random.default_rng(20261027)
    for point in rng.uniform(-4.0, 4.0, size=(500, 2)):
        label = teacher.certified_node_label((float(point[0]), float(point[1])))
        assert label.value - label.distance_error <= label.interval.lower
        assert label.value + label.distance_error >= label.interval.upper


def test_signed_distance_is_one_lipschitz_on_registered_polygon_cases() -> None:
    teacher = ExactPolygonTeacher((_rectangle(-2.0, 2.0, -1.0, 1.0),))
    rng = np.random.default_rng(20261028)
    points = rng.uniform(-4.0, 4.0, size=(300, 2))
    intervals = [
        teacher.signed_distance_interval((float(point[0]), float(point[1]))) for point in points
    ]
    for first in range(0, len(points), 2):
        second = first + 1
        worst_difference = max(
            abs(intervals[first].lower - intervals[second].upper),
            abs(intervals[first].upper - intervals[second].lower),
        )
        separation = float(np.linalg.norm(points[first] - points[second]))
        assert worst_difference <= separation + 1e-12


def test_error_ledger_rounds_up_and_rejects_invalid_terms() -> None:
    ledger = MapErrorLedger(0.01, 0.02, 0.03, 0.04)
    exact = sum(Fraction.from_float(value) for value in (0.01, 0.02, 0.03, 0.04))
    assert Fraction.from_float(ledger.total) >= exact
    assert MapErrorLedger(0.0, 0.0, 0.0, 0.0).total == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        MapErrorLedger(0.0, -1.0, 0.0, 0.0)


def test_invalid_rings_and_empty_teacher_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least three"):
        PolygonComponent(outer=((0.0, 0.0), (1.0, 0.0)))
    with pytest.raises(ValueError, match="repeated closing"):
        PolygonComponent(outer=((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)))
    with pytest.raises(ValueError, match="at least one"):
        ExactPolygonTeacher(())


def test_exact_teacher_reports_frozen_object_size() -> None:
    teacher = ExactPolygonTeacher(
        (
            _rectangle(0.0, 1.0, 0.0, 1.0),
            _rectangle(2.0, 4.0, 2.0, 5.0),
        )
    )

    assert teacher.component_count == 2
    assert teacher.segment_count == 8


def test_accelerated_segment_screening_never_drops_the_exact_nearest() -> None:
    components = (
        _rectangle(-10.0, -2.0, -7.0, 6.0),
        _rectangle(-1.5, 4.0, -3.0, 9.0),
        _rectangle(5.0, 11.0, -8.0, 2.0),
    )
    teacher = ExactPolygonTeacher(components)
    rng = np.random.default_rng(20261031)
    for sample in rng.uniform(-15.0, 15.0, size=(500, 2)):
        point = (float(sample[0]), float(sample[1]))
        candidates = set(int(index) for index in teacher._candidate_segment_indices(point))
        exact_distances = tuple(
            teacher_module._point_segment_squared_distance(point, start, end)
            for start, end in teacher._segments
        )
        exact_minimum = min(exact_distances)
        exact_nearest = {
            index for index, distance in enumerate(exact_distances) if distance == exact_minimum
        }
        assert exact_nearest <= candidates
        interval = teacher.signed_distance_interval(point)
        magnitude_squared = min(
            Fraction.from_float(interval.lower) ** 2,
            Fraction.from_float(interval.upper) ** 2,
        )
        assert magnitude_squared <= exact_minimum
