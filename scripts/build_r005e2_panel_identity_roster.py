"""Extract the outcome-free R005E2 panel identity and hash roster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from aorta_drive.protocol.visual_recoverability import canonical_sha256
from aorta_drive.vision.feature_cache import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = cast(
        dict[str, Any], json.loads(args.context_roster.read_text(encoding="utf-8"))
    )
    result = cast(dict[str, Any], json.loads(args.r005e1b_result.read_text(encoding="utf-8")))
    b4_result = cast(dict[str, Any], json.loads(args.b4_result.read_text(encoding="utf-8")))
    if args.expected_result_sha256 and sha256_file(args.r005e1b_result) != args.expected_result_sha256:
        raise ValueError("R005E1B result hash mismatch")
    if args.expected_b4_sha256 and sha256_file(args.b4_result) != args.expected_b4_sha256:
        raise ValueError("B4R1 result hash mismatch")
    context_rows = {
        (str(row["recording_group"]), int(row["anchor_index"])): row
        for row in cast(list[dict[str, Any]], context["anchor_ledger"])
    }
    result_rows = {
        (str(row["recording_group"]), int(row["anchor_index"])): row
        for row in cast(list[dict[str, Any]], result["anchor_ledgers"])
    }
    b4_rows = {
        (str(row["recording_group"]), int(row["anchor_index"])): row
        for row in cast(list[dict[str, Any]], b4_result["anchor_ledgers"])
    }
    if not (set(context_rows) == set(result_rows) == set(b4_rows)) or len(context_rows) != 672:
        raise ValueError("context, R005E1B, and B4R1 identity rosters differ")

    anchors: list[dict[str, Any]] = []
    for key in sorted(context_rows):
        context_row = context_rows[key]
        result_row = result_rows[key]
        b4_row = b4_rows[key]
        if str(context_row["current_token"]) != str(result_row["scene_token"]):
            raise ValueError("current scene-token join mismatch")
        if str(context_row["previous_token"]) != str(result_row["previous_token"]):
            raise ValueError("previous scene-token join mismatch")
        if str(context_row["log_name"]) != str(result_row["log_name"]):
            raise ValueError("log identity join mismatch")
        if str(context_row["current_token"]) != str(b4_row["scene_token"]):
            raise ValueError("B4R1 current scene-token join mismatch")
        anchors.append(
            {
                "recording_group": key[0],
                "anchor_index": key[1],
                "log_name": str(context_row["log_name"]),
                "scene_token": str(context_row["current_token"]),
                "previous_scene_token": str(context_row["previous_token"]),
                "panel_artifact_sha256": str(b4_row["panel_artifact_sha256"]),
            }
        )
    payload: dict[str, Any] = {
        "schema": "aorta.r005e2.panel_identity_roster.v1",
        "source_context_roster_sha256": sha256_file(args.context_roster),
        "source_r005e1b_result_sha256": sha256_file(args.r005e1b_result),
        "source_b4r1_result_sha256": sha256_file(args.b4_result),
        "source_result_fields_used": [
            "recording_group",
            "anchor_index",
            "scene_token",
            "previous_token",
            "log_name",
            "panel_artifact_sha256",
        ],
        "utility_candidate_or_visual_fields_used": False,
        "development_calibration_test_navdream_target_accessed": False,
        "anchor_count": len(anchors),
        "group_count": len({str(row["recording_group"]) for row in anchors}),
        "anchors": anchors,
    }
    payload["manifest_sha256_without_self"] = canonical_sha256(payload)
    _write_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-roster", type=Path, required=True)
    parser.add_argument("--r005e1b-result", type=Path, required=True)
    parser.add_argument("--b4-result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256")
    parser.add_argument("--expected-b4-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(json.dumps({key: value for key, value in output.items() if key != "anchors"}, indent=2))
