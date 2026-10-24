from __future__ import annotations

import pytest

from aorta_drive.evaluation.temporal_context_roster import (
    TemporalContextRosterProtocol,
    resolve_previous_context,
    summarize_temporal_context_roster,
)


def _frame(token: str, timestamp: int, log_name: str = "log") -> dict[str, object]:
    return {"token": token, "timestamp": timestamp, "log_name": log_name}


def test_resolve_previous_context_uses_raw_order_not_anchor_index() -> None:
    frames = [_frame("previous", 1_000_000), _frame("current", 1_500_000)]
    result = resolve_previous_context(
        frames, current_token="current", maximum_gap_s=0.55
    )
    assert result == {
        "status": "AVAILABLE",
        "previous_token": "previous",
        "gap_s": 0.5,
    }


@pytest.mark.parametrize(
    ("frames", "token", "status"),
    [
        ([_frame("current", 1_000_000)], "current", "NO_PREVIOUS_ROW"),
        ([_frame("x", 1_000_000)], "current", "TOKEN_MISSING"),
        ([_frame("current", 1), _frame("current", 2)], "current", "TOKEN_DUPLICATE"),
        (
            [_frame("previous", 1_000_000, "a"), _frame("current", 1_500_000, "b")],
            "current",
            "LOG_BOUNDARY",
        ),
        (
            [_frame("previous", 1_000_000), _frame("current", 1_600_000)],
            "current",
            "GAP_INVALID",
        ),
    ],
)
def test_resolve_previous_context_fails_closed(
    frames: list[dict[str, object]], token: str, status: str
) -> None:
    assert (
        resolve_previous_context(frames, current_token=token, maximum_gap_s=0.55)[
            "status"
        ]
        == status
    )


def test_summarize_requires_complete_group_product() -> None:
    protocol = TemporalContextRosterProtocol(0.55, 4, 2, 2)
    rows = [
        {
            "recording_group": group,
            "anchor_index": anchor,
            "status": "AVAILABLE",
            "previous_token": f"p-{group}-{anchor}",
            "gap_s": 0.5,
        }
        for group in ("g0", "g1")
        for anchor in range(2)
    ]
    summary = summarize_temporal_context_roster(
        rows, group_names=["g0", "g1"], protocol=protocol
    )
    assert summary["decision"] == "TEMPORAL_CONTEXT_ROSTER_PASS"
    assert summary["group_available_histogram"] == {"2": 2}

    rows[-1] = {**rows[-1], "status": "GAP_INVALID"}
    summary = summarize_temporal_context_roster(
        rows, group_names=["g0", "g1"], protocol=protocol
    )
    assert summary["decision"] == "TEMPORAL_CONTEXT_ROSTER_INVALID"


def test_protocol_dimensions_must_close() -> None:
    with pytest.raises(ValueError, match="multiply"):
        TemporalContextRosterProtocol(0.55, 5, 2, 2)

