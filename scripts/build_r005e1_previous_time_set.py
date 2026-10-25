"""Build a frozen source-fit time-set for the causal previous tokens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, cast


FIELDS = (
    "recording_group",
    "split",
    "anchor_index",
    "session_position",
    "scene_token",
    "log_name",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_rows(current_rows: list[dict[str, str]], roster: dict[str, Any]) -> list[dict[str, str]]:
    current_by_key = {
        (row["recording_group"], int(row["anchor_index"])): row
        for row in current_rows
        if row["split"] == "train"
    }
    context_rows = cast(list[dict[str, Any]], roster["anchor_ledger"])
    if len(context_rows) != 672:
        raise ValueError("expected 672 context rows")
    output: list[dict[str, str]] = []
    for context in context_rows:
        key = (str(context["recording_group"]), int(context["anchor_index"]))
        current = current_by_key.get(key)
        if current is None:
            raise ValueError(f"current train anchor missing: {key}")
        if current["scene_token"] != str(context["current_token"]):
            raise ValueError(f"current token mismatch: {key}")
        if current["log_name"] != str(context["log_name"]):
            raise ValueError(f"log mismatch: {key}")
        if context.get("status") != "AVAILABLE":
            raise ValueError(f"previous context unavailable: {key}")
        position = int(current["session_position"])
        if position <= 0:
            raise ValueError(f"current token has no causal predecessor position: {key}")
        output.append(
            {
                "recording_group": key[0],
                "split": "train",
                "anchor_index": str(key[1]),
                "session_position": str(position - 1),
                "scene_token": str(context["previous_token"]),
                "log_name": str(context["log_name"]),
            }
        )
    if len(output) != 672 or len({row["scene_token"] for row in output}) != 672:
        raise ValueError("previous-token time set is incomplete or non-unique")
    return sorted(output, key=lambda row: (row["recording_group"], int(row["anchor_index"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-time-set", type=Path, required=True)
    parser.add_argument("--context-roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with args.current_time_set.open("r", encoding="utf-8", newline="") as stream:
        current_rows = list(csv.DictReader(stream))
    roster = cast(dict[str, Any], json.loads(args.context_roster.read_text(encoding="utf-8")))
    rows = build_rows(current_rows, roster)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "PASS",
        "role": "source_fit_causal_previous_tokens_only",
        "row_count": len(rows),
        "recording_group_count": len({row["recording_group"] for row in rows}),
        "current_time_set_sha256": _sha256(args.current_time_set),
        "context_roster_sha256": _sha256(args.context_roster),
        "output_sha256": _sha256(args.output),
        "development_calibration_test_navdream_target_accessed": False,
        "scientific_outcomes_accessed": False,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

