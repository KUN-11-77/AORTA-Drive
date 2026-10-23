from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from types import SimpleNamespace

from shapely.geometry import MultiPolygon, Polygon

from aorta_drive.protocol.cache_fingerprint import teacher_input_sha256


class _Layer(Enum):
    ROADBLOCK = auto()
    DRIVABLE_AREA = auto()
    LANE = auto()


@dataclass(frozen=True)
class _DrivableMap:
    _geometries: tuple[object, ...]
    _map_types: tuple[_Layer, ...]


def _cache(
    geometries: tuple[object, ...], layers: tuple[_Layer, ...]
) -> SimpleNamespace:
    return SimpleNamespace(
        ego_state=SimpleNamespace(
            rear_axle=SimpleNamespace(x=1.25, y=-2.5, heading=0.125)
        ),
        drivable_area_map=_DrivableMap(geometries, layers),
    )


def test_fingerprint_is_invariant_to_component_ring_and_part_order() -> None:
    first = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    first_rotated_reversed = Polygon([(2, 0), (0, 0), (0, 2), (2, 2)])
    second = Polygon([(4, 0), (5, 0), (5, 1), (4, 1)])
    left = _cache(
        (MultiPolygon([first, second]),),
        (_Layer.DRIVABLE_AREA,),
    )
    right = _cache(
        (MultiPolygon([second, first_rotated_reversed]),),
        (_Layer.DRIVABLE_AREA,),
    )
    assert teacher_input_sha256(left) == teacher_input_sha256(right)


def test_fingerprint_changes_with_teacher_geometry_or_pose() -> None:
    polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    changed_polygon = Polygon([(0, 0), (3, 0), (2, 2), (0, 2)])
    baseline = _cache((polygon,), (_Layer.ROADBLOCK,))
    changed = _cache((changed_polygon,), (_Layer.ROADBLOCK,))
    assert teacher_input_sha256(baseline) != teacher_input_sha256(changed)

    moved = _cache((polygon,), (_Layer.ROADBLOCK,))
    moved.ego_state.rear_axle.x = 1.5
    assert teacher_input_sha256(baseline) != teacher_input_sha256(moved)


def test_unselected_layer_does_not_change_teacher_fingerprint() -> None:
    selected = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    ignored = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
    baseline = _cache((selected,), (_Layer.ROADBLOCK,))
    with_ignored = _cache((selected, ignored), (_Layer.ROADBLOCK, _Layer.LANE))
    assert teacher_input_sha256(baseline) == teacher_input_sha256(with_ignored)
