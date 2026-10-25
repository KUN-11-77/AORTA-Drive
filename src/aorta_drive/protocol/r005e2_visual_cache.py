"""Fail-closed identity loading for the R005E2 frozen visual cache."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from aorta_drive.protocol.training_join import TrainingAnchor


@dataclass(frozen=True)
class VisualCacheRosterAudit:
    anchor_count: int
    recording_group_count: int
    anchors_per_group: int


def anchors_from_panel_roster(
    payload: Mapping[str, Any],
    *,
    expected_anchor_count: int = 672,
    expected_group_count: int = 42,
    expected_anchors_per_group: int = 16,
) -> tuple[tuple[TrainingAnchor, ...], VisualCacheRosterAudit]:
    """Validate the outcome-free roster and return exact source-fit anchors."""

    if payload.get("schema") != "aorta.r005e2.panel_identity_roster.v1":
        raise ValueError("unexpected R005E2 panel-roster schema")
    if payload.get("utility_candidate_or_visual_fields_used") is not False:
        raise ValueError("panel roster is not outcome/content free")
    if payload.get("development_calibration_test_navdream_target_accessed") is not False:
        raise ValueError("panel roster accessed a sealed data role")
    raw_rows = payload.get("anchors")
    if not isinstance(raw_rows, list):
        raise ValueError("panel roster must contain an anchors list")

    anchors: list[TrainingAnchor] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("every panel-roster anchor must be an object")
        anchor = TrainingAnchor(
            recording_group=str(raw["recording_group"]),
            anchor_index=int(raw["anchor_index"]),
            split="train",
            scene_token=str(raw["scene_token"]),
            log_name=str(raw["log_name"]),
        )
        if not anchor.recording_group or not anchor.scene_token or not anchor.log_name:
            raise ValueError("panel-roster identity fields must be nonempty")
        anchors.append(anchor)

    anchors.sort(key=lambda item: (item.recording_group, item.anchor_index))
    keys = {(anchor.recording_group, anchor.anchor_index) for anchor in anchors}
    tokens = {anchor.scene_token for anchor in anchors}
    if len(anchors) != expected_anchor_count:
        raise ValueError("R005E2 panel-roster anchor count mismatch")
    if len(keys) != len(anchors) or len(tokens) != len(anchors):
        raise ValueError("R005E2 anchor keys and scene tokens must be unique")
    counts = Counter(anchor.recording_group for anchor in anchors)
    if len(counts) != expected_group_count:
        raise ValueError("R005E2 recording-group count mismatch")
    if set(counts.values()) != {expected_anchors_per_group}:
        raise ValueError("every R005E2 recording group must contain exactly 16 anchors")
    if int(payload.get("anchor_count", -1)) != len(anchors):
        raise ValueError("panel-roster declared anchor count mismatch")
    if int(payload.get("group_count", -1)) != len(counts):
        raise ValueError("panel-roster declared group count mismatch")

    audit = VisualCacheRosterAudit(
        anchor_count=len(anchors),
        recording_group_count=len(counts),
        anchors_per_group=expected_anchors_per_group,
    )
    return tuple(anchors), audit


def exact_slot_order(slots: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether twelve visual slots use the frozen history/camera order."""

    expected = [
        (slot_index, history_offset, camera)
        for slot_index, (history_offset, camera) in enumerate(
            (offset, camera)
            for offset in (-3, -2, -1, 0)
            for camera in ("left", "front", "right")
        )
    ]
    observed = [
        (int(slot["slot_index"]), int(slot["history_offset"]), str(slot["camera"]))
        for slot in slots
    ]
    return observed == expected
