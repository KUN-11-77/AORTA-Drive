"""Source-session splitting and deterministic time-set helpers.

The independent unit is a complete NAVSIM recording, not an exported log
fragment or a frame.  This module deliberately contains no outcome-dependent
allocation logic: a split assignment is frozen externally and is only audited
here against the source roster.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

SPLIT_NAMES = ("train", "development", "calibration", "test")
_LOG_PATTERN = re.compile(r"^(?P<group>.*_veh-\d+)_(?P<start>\d+)_(?P<end>\d+)$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceLogRecord:
    """One official source log and its result-independent provenance."""

    log_name: str
    recording_group: str
    scene_count: int
    token_sha256: str
    log_sha256: str
    sensor_dir_exists: bool


@dataclass(frozen=True)
class SplitAudit:
    """Deterministic summary of an audited recording-group split."""

    group_counts: Mapping[str, int]
    log_counts: Mapping[str, int]
    scene_counts: Mapping[str, int]
    total_groups: int
    total_logs: int
    total_scenes: int
    logical_sha256: str


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value with one canonical serialization."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recording_group_for_log(log_name: str) -> str:
    """Return the physical recording identifier for an exported log fragment."""

    match = _LOG_PATTERN.fullmatch(log_name)
    if match is None:
        raise ValueError(f"invalid NAVSIM log name: {log_name!r}")
    start = int(match.group("start"))
    end = int(match.group("end"))
    if end < start:
        raise ValueError(f"log end precedes start: {log_name!r}")
    return match.group("group")


def midpoint_anchor_positions(scene_count: int, anchor_count: int) -> tuple[int, ...]:
    """Select fixed midpoint-quantile positions from a chronological session.

    Position ``j`` is ``floor((2j + 1) N / (2m))`` for session length ``N``
    and requested anchor count ``m``.  The definition is deterministic and
    uses no labels, predictions, or downstream scores.
    """

    if scene_count <= 0:
        raise ValueError("scene_count must be positive")
    if anchor_count <= 0:
        raise ValueError("anchor_count must be positive")
    if anchor_count > scene_count:
        raise ValueError("anchor_count cannot exceed scene_count")
    positions = tuple(
        math.floor((2 * index + 1) * scene_count / (2 * anchor_count))
        for index in range(anchor_count)
    )
    if len(set(positions)) != anchor_count:
        raise AssertionError("midpoint rule produced duplicate positions")
    return positions


def _validate_record(record: SourceLogRecord) -> None:
    if record.recording_group != recording_group_for_log(record.log_name):
        raise ValueError(f"recording group mismatch for {record.log_name}")
    if record.scene_count <= 0:
        raise ValueError(f"non-positive scene count for {record.log_name}")
    if not record.sensor_dir_exists:
        raise ValueError(f"missing sensor directory for {record.log_name}")
    for label, digest in (
        ("token_sha256", record.token_sha256),
        ("log_sha256", record.log_sha256),
    ):
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"invalid {label} for {record.log_name}")


def build_split_audit(
    records: Iterable[SourceLogRecord],
    assignments: Mapping[str, str],
    expected_group_counts: Mapping[str, int],
) -> SplitAudit:
    """Validate and summarize a frozen recording-group assignment.

    Every recording group must occur in exactly one split, every roster log
    must have sensors, and the assignment cannot contain missing or extra
    groups.  The returned hash commits to every log-level row and split label.
    """

    rows = tuple(records)
    if not rows:
        raise ValueError("source roster is empty")
    names = [record.log_name for record in rows]
    if len(names) != len(set(names)):
        raise ValueError("source roster contains duplicate log names")
    for record in rows:
        _validate_record(record)

    roster_groups = {record.recording_group for record in rows}
    assignment_groups = set(assignments)
    if roster_groups != assignment_groups:
        missing = sorted(roster_groups - assignment_groups)
        extra = sorted(assignment_groups - roster_groups)
        raise ValueError(f"assignment group mismatch: missing={missing}, extra={extra}")
    invalid_splits = sorted(set(assignments.values()) - set(SPLIT_NAMES))
    if invalid_splits:
        raise ValueError(f"invalid split names: {invalid_splits}")
    if set(expected_group_counts) != set(SPLIT_NAMES):
        raise ValueError("expected_group_counts must name all and only the frozen splits")

    group_counts = Counter(assignments.values())
    if dict(group_counts) != dict(expected_group_counts):
        raise ValueError(f"group counts do not match the frozen protocol: {dict(group_counts)}")

    log_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    canonical_rows: list[dict[str, object]] = []
    for record in sorted(rows, key=lambda item: item.log_name):
        split = assignments[record.recording_group]
        log_counts[split] += 1
        scene_counts[split] += record.scene_count
        row = asdict(record)
        row["split"] = split
        canonical_rows.append(row)

    ordered_groups = {name: int(group_counts[name]) for name in SPLIT_NAMES}
    ordered_logs = {name: int(log_counts[name]) for name in SPLIT_NAMES}
    ordered_scenes = {name: int(scene_counts[name]) for name in SPLIT_NAMES}
    return SplitAudit(
        group_counts=ordered_groups,
        log_counts=ordered_logs,
        scene_counts=ordered_scenes,
        total_groups=len(roster_groups),
        total_logs=len(rows),
        total_scenes=sum(record.scene_count for record in rows),
        logical_sha256=canonical_sha256(canonical_rows),
    )


def choose_anchor_tokens(tokens: Sequence[str], anchor_count: int) -> tuple[str, ...]:
    """Apply the frozen midpoint rule to an ordered session token sequence."""

    if len(tokens) != len(set(tokens)):
        raise ValueError("session token sequence contains duplicates")
    positions = midpoint_anchor_positions(len(tokens), anchor_count)
    return tuple(tokens[position] for position in positions)
