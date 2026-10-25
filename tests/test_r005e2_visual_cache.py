from __future__ import annotations

from copy import deepcopy

import pytest

from aorta_drive.protocol.r005e2_visual_cache import (
    anchors_from_panel_roster,
    exact_slot_order,
)


def _roster() -> dict[str, object]:
    rows = [
        {
            "recording_group": f"g{group:02d}",
            "anchor_index": anchor,
            "scene_token": f"token-{group:02d}-{anchor:02d}",
            "log_name": f"log-{group:02d}",
        }
        for group in range(42)
        for anchor in range(16)
    ]
    return {
        "schema": "aorta.r005e2.panel_identity_roster.v1",
        "utility_candidate_or_visual_fields_used": False,
        "development_calibration_test_navdream_target_accessed": False,
        "anchor_count": 672,
        "group_count": 42,
        "anchors": rows,
    }


def test_panel_roster_builds_exact_train_anchor_contract() -> None:
    anchors, audit = anchors_from_panel_roster(_roster())
    assert len(anchors) == 672
    assert audit.recording_group_count == 42
    assert {anchor.split for anchor in anchors} == {"train"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("utility_candidate_or_visual_fields_used", True),
        ("development_calibration_test_navdream_target_accessed", True),
    ],
)
def test_panel_roster_rejects_role_or_content_contamination(
    field: str, value: bool
) -> None:
    payload = _roster()
    payload[field] = value
    with pytest.raises(ValueError):
        anchors_from_panel_roster(payload)


def test_panel_roster_rejects_duplicate_scene_token() -> None:
    payload = deepcopy(_roster())
    rows = payload["anchors"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict) and isinstance(rows[1], dict)
    rows[1]["scene_token"] = rows[0]["scene_token"]
    with pytest.raises(ValueError):
        anchors_from_panel_roster(payload)


def test_exact_slot_order_is_strict() -> None:
    slots = [
        {"slot_index": index, "history_offset": offset, "camera": camera}
        for index, (offset, camera) in enumerate(
            (offset, camera)
            for offset in (-3, -2, -1, 0)
            for camera in ("left", "front", "right")
        )
    ]
    assert exact_slot_order(slots)
    swapped = deepcopy(slots)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert not exact_slot_order(swapped)
