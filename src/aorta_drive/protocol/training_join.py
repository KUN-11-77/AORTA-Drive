"""Pure validation helpers for the R004E anchor/cache/label join."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class TrainingAnchor:
    recording_group: str
    anchor_index: int
    split: str
    scene_token: str
    log_name: str


@dataclass(frozen=True)
class JoinAudit:
    anchor_count: int
    recording_group_count: int
    label_token_count: int
    matched_anchor_count: int
    missing_anchor_count: int
    unexpected_label_count: int
    duplicate_label_token_count: int

    @property
    def complete(self) -> bool:
        return (
            self.anchor_count > 0
            and self.matched_anchor_count == self.anchor_count
            and self.missing_anchor_count == 0
            and self.unexpected_label_count == 0
            and self.duplicate_label_token_count == 0
        )


def load_training_anchors(
    path: Path,
    *,
    allowed_splits: Sequence[str] = ("train", "development"),
) -> tuple[TrainingAnchor, ...]:
    allowed = frozenset(allowed_splits)
    if not allowed or not allowed <= {"train", "development"}:
        raise ValueError("R004E may load only train and development anchors")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    anchors = tuple(
        sorted(
            (
                TrainingAnchor(
                    recording_group=row["recording_group"],
                    anchor_index=int(row["anchor_index"]),
                    split=row["split"],
                    scene_token=row["scene_token"],
                    log_name=row["log_name"],
                )
                for row in rows
                if row["split"] in allowed
            ),
            key=lambda anchor: (anchor.recording_group, anchor.anchor_index),
        )
    )
    if not anchors:
        raise ValueError("no permitted anchors found")
    keys = {(anchor.recording_group, anchor.anchor_index) for anchor in anchors}
    tokens = {anchor.scene_token for anchor in anchors}
    if len(keys) != len(anchors) or len(tokens) != len(anchors):
        raise ValueError("anchor keys and scene tokens must each be unique")
    group_counts = Counter(anchor.recording_group for anchor in anchors)
    if any(count != 16 for count in group_counts.values()):
        raise ValueError("every selected recording group must have exactly 16 anchors")
    return anchors


def select_preflight_anchors(
    anchors: Sequence[TrainingAnchor],
    *,
    maximum_anchors: int | None = None,
    maximum_recording_groups: int | None = None,
) -> tuple[TrainingAnchor, ...]:
    if maximum_anchors is not None and maximum_recording_groups is not None:
        raise ValueError("choose either an anchor limit or a recording-group limit")
    if maximum_anchors is not None:
        if maximum_anchors <= 0:
            raise ValueError("maximum_anchors must be positive")
        return tuple(anchors[:maximum_anchors])
    if maximum_recording_groups is not None:
        if maximum_recording_groups <= 0:
            raise ValueError("maximum_recording_groups must be positive")
        groups = sorted({anchor.recording_group for anchor in anchors})[
            :maximum_recording_groups
        ]
        selected = set(groups)
        return tuple(anchor for anchor in anchors if anchor.recording_group in selected)
    return tuple(anchors)


def label_manifest_tokens(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("label manifest must contain a records list")
    tokens: list[str] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("cache_id"), str):
            raise ValueError("every label record must contain a string cache_id")
        parts = record["cache_id"].split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError("cache_id must have the form log_name/scene_token")
        tokens.append(parts[1])
    return tuple(tokens)


def audit_label_join(
    anchors: Sequence[TrainingAnchor], label_tokens: Iterable[str]
) -> JoinAudit:
    anchor_tokens = {anchor.scene_token for anchor in anchors}
    counts = Counter(label_tokens)
    unique_labels = set(counts)
    matched = anchor_tokens & unique_labels
    return JoinAudit(
        anchor_count=len(anchors),
        recording_group_count=len({anchor.recording_group for anchor in anchors}),
        label_token_count=len(unique_labels),
        matched_anchor_count=len(matched),
        missing_anchor_count=len(anchor_tokens - unique_labels),
        unexpected_label_count=len(unique_labels - anchor_tokens),
        duplicate_label_token_count=sum(count - 1 for count in counts.values() if count > 1),
    )


def expected_metric_cache_path(root: Path, anchor: TrainingAnchor) -> Path:
    return root / anchor.log_name / "unknown" / anchor.scene_token / "metric_cache.pkl"
