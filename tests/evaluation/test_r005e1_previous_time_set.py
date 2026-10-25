from __future__ import annotations

import pytest

from build_r005e1_previous_time_set import build_rows


def _inputs():
    current = []
    context = []
    for group_index in range(42):
        group = f"g{group_index:02d}"
        for anchor in range(16):
            current.append(
                {
                    "recording_group": group,
                    "split": "train",
                    "anchor_index": str(anchor),
                    "session_position": str(anchor + 10),
                    "scene_token": f"c-{group}-{anchor}",
                    "log_name": f"log-{group}",
                }
            )
            context.append(
                {
                    "recording_group": group,
                    "anchor_index": anchor,
                    "current_token": f"c-{group}-{anchor}",
                    "previous_token": f"p-{group}-{anchor}",
                    "log_name": f"log-{group}",
                    "status": "AVAILABLE",
                }
            )
    return current, {"anchor_ledger": context}


def test_builds_exact_previous_rows() -> None:
    current, roster = _inputs()
    rows = build_rows(current, roster)
    assert len(rows) == 672
    assert rows[0]["scene_token"].startswith("p-")
    assert int(rows[0]["session_position"]) == 9
    assert {row["split"] for row in rows} == {"train"}


def test_rejects_context_mismatch() -> None:
    current, roster = _inputs()
    roster["anchor_ledger"][0]["current_token"] = "wrong"
    with pytest.raises(ValueError, match="current token mismatch"):
        build_rows(current, roster)

