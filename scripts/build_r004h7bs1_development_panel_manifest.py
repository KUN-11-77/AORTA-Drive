"""Extract an outcome-free development panel manifest from the audited H7B bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, cast

from aorta_drive.vision.feature_cache import sha256_file

EXPECTED_H7B_RESULT_SHA256 = "64e80d90124a2d88b60cd55339970d03677d04262ddb74ec3ba4b019196f7488"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.h7b_result) != EXPECTED_H7B_RESULT_SHA256:
        raise ValueError("H7B result is not the audited insufficient-coverage run")
    result = json.loads(args.h7b_result.read_text(encoding="utf-8"))
    if result.get("decision") != "INSUFFICIENT_PANEL_COVERAGE":
        raise ValueError("H7B result does not have the registered coverage decision")
    ledgers = result.get("anchor_ledgers")
    if not isinstance(ledgers, list) or len(ledgers) != 384:
        raise ValueError("H7B result does not contain exactly 384 anchor ledgers")
    records = [
        {
            "recording_group": str(row["recording_group"]),
            "anchor_index": int(row["anchor_index"]),
            "scene_token": str(row["scene_token"]),
            "panel_artifact": str(row["panel_artifact"]),
            "panel_artifact_sha256": str(row["panel_artifact_sha256"]),
        }
        for raw in ledgers
        for row in (cast(dict[str, Any], raw),)
    ]
    records.sort(key=lambda row: (row["recording_group"], row["anchor_index"]))
    if len({(row["recording_group"], row["anchor_index"]) for row in records}) != 384:
        raise ValueError("H7B development panel identities are not unique")
    payload = {
        "manifest_id": "R004H7BS1-development-panel-artifacts-v1",
        "source_result_sha256": EXPECTED_H7B_RESULT_SHA256,
        "record_count": 384,
        "permitted_fields": [
            "recording_group",
            "anchor_index",
            "scene_token",
            "panel_artifact",
            "panel_artifact_sha256",
        ],
        "outcome_fields_included": False,
        "records": records,
    }
    _write_json(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h7b-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args)
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
