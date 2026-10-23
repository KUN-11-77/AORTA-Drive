from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from aorta_drive.protocol.training_join import (
    audit_label_join,
    expected_metric_cache_path,
    label_manifest_tokens,
    load_training_anchors,
    select_preflight_anchors,
)


def _write_time_set(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "recording_group",
                "split",
                "anchor_index",
                "session_position",
                "scene_token",
                "log_name",
            ),
        )
        writer.writeheader()
        for group, split in (("group-a", "train"), ("group-b", "development")):
            for index in range(16):
                writer.writerow(
                    {
                        "recording_group": group,
                        "split": split,
                        "anchor_index": index,
                        "session_position": index,
                        "scene_token": f"{group}-token-{index}",
                        "log_name": f"{group}-log",
                    }
                )
        for index in range(16):
            writer.writerow(
                {
                    "recording_group": "group-c",
                    "split": "calibration",
                    "anchor_index": index,
                    "session_position": index,
                    "scene_token": f"group-c-token-{index}",
                    "log_name": "group-c-log",
                }
            )


def test_load_and_preflight_selection_preserve_recording_groups(tmp_path: Path) -> None:
    time_set = tmp_path / "time_set.csv"
    _write_time_set(time_set)
    anchors = load_training_anchors(time_set)
    assert len(anchors) == 32
    assert {anchor.split for anchor in anchors} == {"train", "development"}
    assert len(select_preflight_anchors(anchors, maximum_anchors=1)) == 1
    group_preflight = select_preflight_anchors(anchors, maximum_recording_groups=1)
    assert len(group_preflight) == 16
    assert len({anchor.recording_group for anchor in group_preflight}) == 1


def test_label_join_requires_exact_token_coverage(tmp_path: Path) -> None:
    time_set = tmp_path / "time_set.csv"
    _write_time_set(time_set)
    anchors = load_training_anchors(time_set)
    complete = audit_label_join(anchors, (anchor.scene_token for anchor in anchors))
    assert complete.complete
    incomplete = audit_label_join(anchors, [anchors[0].scene_token, "unexpected"])
    assert not incomplete.complete
    assert incomplete.matched_anchor_count == 1
    assert incomplete.missing_anchor_count == 31
    assert incomplete.unexpected_label_count == 1


def test_manifest_parser_and_expected_cache_path(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"records": [{"cache_id": "log-a/token-a"}]}), encoding="utf-8"
    )
    assert label_manifest_tokens(manifest) == ("token-a",)
    time_set = tmp_path / "time_set.csv"
    _write_time_set(time_set)
    anchor = load_training_anchors(time_set)[0]
    assert expected_metric_cache_path(Path("root"), anchor) == (
        Path("root") / anchor.log_name / "unknown" / anchor.scene_token / "metric_cache.pkl"
    )


def test_anchor_loader_rejects_pseudoreplicated_partial_group(tmp_path: Path) -> None:
    time_set = tmp_path / "bad.csv"
    with time_set.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "recording_group",
                "split",
                "anchor_index",
                "session_position",
                "scene_token",
                "log_name",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "recording_group": "group-a",
                "split": "train",
                "anchor_index": 0,
                "session_position": 0,
                "scene_token": "token-a",
                "log_name": "log-a",
            }
        )
    with pytest.raises(ValueError, match="exactly 16"):
        load_training_anchors(time_set)
