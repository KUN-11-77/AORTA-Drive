"""Outcome-free temporal-context roster checks for full NAVSIM scoring.

The official two-frame comfort term is defined only when an evaluated current
frame has an evaluated predecessor within the frozen time-gap contract.  The
helpers in this module resolve that predecessor from the original log order;
an experiment anchor index is deliberately never treated as time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TemporalContextRosterProtocol:
    """Frozen identity and adjacency requirements."""

    maximum_gap_s: float
    expected_anchor_count: int
    expected_group_count: int
    expected_anchors_per_group: int

    def __post_init__(self) -> None:
        if not 0.0 < self.maximum_gap_s <= 1.0:
            raise ValueError("maximum temporal gap must lie in (0, 1]")
        if self.expected_anchor_count <= 0:
            raise ValueError("expected anchor count must be positive")
        if self.expected_group_count <= 0 or self.expected_anchors_per_group <= 0:
            raise ValueError("group dimensions must be positive")
        if (
            self.expected_group_count * self.expected_anchors_per_group
            != self.expected_anchor_count
        ):
            raise ValueError("group dimensions must multiply to anchor count")


def resolve_previous_context(
    frames: Sequence[Mapping[str, Any]],
    *,
    current_token: str,
    maximum_gap_s: float,
) -> dict[str, Any]:
    """Resolve the immediately preceding raw-log row for one current token.

    This is intentionally stricter than searching for any convenient earlier
    row.  Duplicate tokens, log-boundary crossings, nonpositive gaps, and gaps
    larger than the official adjacency threshold all fail closed.
    """

    positions = [
        index for index, frame in enumerate(frames) if str(frame.get("token")) == current_token
    ]
    if len(positions) != 1:
        return {
            "status": "TOKEN_MISSING" if not positions else "TOKEN_DUPLICATE",
            "previous_token": None,
            "gap_s": None,
        }
    index = positions[0]
    if index == 0:
        return {"status": "NO_PREVIOUS_ROW", "previous_token": None, "gap_s": None}

    current = frames[index]
    previous = frames[index - 1]
    if str(current.get("log_name")) != str(previous.get("log_name")):
        return {
            "status": "LOG_BOUNDARY",
            "previous_token": str(previous.get("token")),
            "gap_s": None,
        }
    try:
        gap_s = (int(current["timestamp"]) - int(previous["timestamp"])) / 1e6
    except (KeyError, TypeError, ValueError):
        return {
            "status": "TIMESTAMP_INVALID",
            "previous_token": str(previous.get("token")),
            "gap_s": None,
        }
    status = "AVAILABLE" if 0.0 < gap_s <= maximum_gap_s else "GAP_INVALID"
    return {
        "status": status,
        "previous_token": str(previous.get("token")),
        "gap_s": float(gap_s),
    }


def summarize_temporal_context_roster(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_names: Sequence[str],
    protocol: TemporalContextRosterProtocol,
) -> dict[str, Any]:
    """Validate the complete group-by-anchor roster and summarize availability."""

    if len(rows) != protocol.expected_anchor_count:
        raise ValueError("temporal roster anchor count mismatch")
    if len(group_names) != protocol.expected_group_count or len(set(group_names)) != len(
        group_names
    ):
        raise ValueError("temporal roster group identities mismatch")
    expected = {
        (group, anchor)
        for group in group_names
        for anchor in range(protocol.expected_anchors_per_group)
    }
    observed = {
        (str(row["recording_group"]), int(row["anchor_index"])) for row in rows
    }
    if observed != expected or len(observed) != len(rows):
        raise ValueError("temporal roster is not the complete group-by-anchor product")

    statuses = Counter(str(row["status"]) for row in rows)
    group_available: dict[str, int] = {group: 0 for group in group_names}
    available_gaps: list[float] = []
    for row in rows:
        if row["status"] == "AVAILABLE":
            previous_token = row.get("previous_token")
            gap = row.get("gap_s")
            if not previous_token or gap is None:
                raise ValueError("available context lacks predecessor identity or gap")
            gap_value = float(gap)
            if not 0.0 < gap_value <= protocol.maximum_gap_s:
                raise ValueError("available context violates the frozen gap")
            group_available[str(row["recording_group"])] += 1
            available_gaps.append(gap_value)

    histogram = Counter(group_available.values())
    all_statuses_available = statuses == Counter(
        {"AVAILABLE": protocol.expected_anchor_count}
    )
    all_groups_complete = histogram == Counter(
        {protocol.expected_anchors_per_group: protocol.expected_group_count}
    )
    passed = all_statuses_available and all_groups_complete
    return {
        "decision": (
            "TEMPORAL_CONTEXT_ROSTER_PASS"
            if passed
            else "TEMPORAL_CONTEXT_ROSTER_INVALID"
        ),
        "anchor_count": len(rows),
        "group_count": len(group_names),
        "status_counts": dict(sorted(statuses.items())),
        "group_available_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
        "groups_all_context_available": sum(
            value == protocol.expected_anchors_per_group
            for value in group_available.values()
        ),
        "available_gap_min_s": min(available_gaps) if available_gaps else None,
        "available_gap_max_s": max(available_gaps) if available_gaps else None,
        "scientific_outcomes_accessed": False,
    }

