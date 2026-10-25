"""Build exact 672-anchor inputs after the frozen one-anchor context repair."""

from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.evaluation.eligible_preference_oracle import (
    summarize_eligible_preference_oracle,
)
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4b4_preference_oracle import _groups, _write_json
from run_r004h7bs4b4r1_eligible_preference_oracle import _protocol


REPAIR_KEY = ("2021.06.07.19.29.59_veh-38", 4)
OLD_CURRENT = "2c18d1604abe52b8"
NEW_CURRENT = "e25e0f03413553f9"
NEW_GAP_S = 0.500173


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["recording_group"]), int(row["anchor_index"])


def _link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def _cache_path(root: Path, record: dict[str, Any]) -> Path:
    return root / str(record["relative_cache_path"])


def _record_for_file(template: dict[str, Any], source: Path) -> dict[str, Any]:
    record = deepcopy(template)
    record["cache_bytes"] = source.stat().st_size
    record["cache_sha256"] = sha256_file(source)
    record.pop("teacher_input_sha256", None)
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    current_output = args.output_root / "current"
    previous_output = args.output_root / "previous"

    original_manifest = cast(
        dict[str, Any], json.loads(args.original_current_manifest.read_text("utf-8"))
    )
    replacement_manifest = cast(
        dict[str, Any], json.loads(args.replacement_current_manifest.read_text("utf-8"))
    )
    original_records = cast(list[dict[str, Any]], original_manifest["records"])
    replacement_records = {
        _key(row): row for row in cast(list[dict[str, Any]], replacement_manifest["records"])
    }
    if REPAIR_KEY not in replacement_records:
        raise ValueError("replacement current manifest lacks the frozen anchor")

    combined_current_records: list[dict[str, Any]] = []
    original_by_key = {_key(row): row for row in original_records}
    for original in original_records:
        key = _key(original)
        source_record = replacement_records[key] if key == REPAIR_KEY else original
        source_root = (
            args.replacement_current_root if key == REPAIR_KEY else args.original_current_root
        )
        if key == REPAIR_KEY and source_record["scene_token"] != NEW_CURRENT:
            raise ValueError("replacement current token changed")
        source = _cache_path(source_root, source_record)
        destination = current_output / str(source_record["relative_cache_path"])
        _link(source, destination)
        combined_current_records.append(_record_for_file(source_record, source))

    with args.previous_time_set.open("r", encoding="utf-8", newline="") as stream:
        previous_rows = [dict(row) for row in csv.DictReader(stream)]
    if len(previous_rows) != 672:
        raise ValueError("previous time set is not 672 anchors")
    previous_by_key = {_key(row): row for row in previous_rows}
    combined_previous_records: list[dict[str, Any]] = []
    for key in sorted(previous_by_key):
        template = previous_by_key[key]
        template["anchor_index"] = int(template["anchor_index"])
        if key == REPAIR_KEY:
            source_record = original_by_key[key]
            if source_record["scene_token"] != OLD_CURRENT:
                raise ValueError("repair predecessor is not the original current token")
            source = _cache_path(args.original_current_root, source_record)
            template["scene_token"] = OLD_CURRENT
            template["log_name"] = source_record["log_name"]
        else:
            relative = (
                Path("metric_cache")
                / str(template["log_name"])
                / "unknown"
                / str(template["scene_token"])
                / "metric_cache.pkl"
            )
            source = args.partial_previous_root / relative
        relative = (
            Path("metric_cache")
            / str(template["log_name"])
            / "unknown"
            / str(template["scene_token"])
            / "metric_cache.pkl"
        )
        template["relative_cache_path"] = relative.as_posix()
        destination = previous_output / relative
        _link(source, destination)
        combined_previous_records.append(_record_for_file(template, source))

    current_manifest = {
        **{key: value for key, value in original_manifest.items() if key != "records"},
        "experiment_id": "R005E1-repaired-current-cache-manifest",
        "status": "PASS",
        "anchor_count": 672,
        "recording_group_count": 42,
        "repair_key": list(REPAIR_KEY),
        "records": combined_current_records,
    }
    previous_manifest = {
        "experiment_id": "R005E1-repaired-causal-previous-cache-manifest",
        "status": "PASS",
        "anchor_count": 672,
        "recording_group_count": 42,
        "repair_key": list(REPAIR_KEY),
        "records": combined_previous_records,
        "development_calibration_test_navdream_target_accessed": False,
    }
    _write_json(current_output / "manifest.json", current_manifest)
    _write_json(previous_output / "manifest.json", previous_manifest)

    b4 = cast(dict[str, Any], json.loads(args.b4_result.read_text("utf-8")))
    replacement = cast(
        dict[str, Any], json.loads(args.replacement_one_frame.read_text("utf-8"))
    )
    replacement_row = cast(dict[str, Any], replacement["anchor_ledger"])
    if _key(replacement_row) != REPAIR_KEY or replacement_row["scene_token"] != NEW_CURRENT:
        raise ValueError("replacement one-frame row is not the frozen repair")
    b4_rows = cast(list[dict[str, Any]], b4["anchor_ledgers"])
    repaired_rows = [replacement_row if _key(row) == REPAIR_KEY else row for row in b4_rows]
    b4_config = cast(dict[str, Any], yaml.safe_load(args.b4_config.read_text("utf-8")))
    b4_stage = cast(dict[str, Any], b4_config["H7BS4B4R1"])
    groups = _groups(args.group_role_manifest)
    summary = summarize_eligible_preference_oracle(
        repaired_rows,
        group_names=groups,
        protocol=_protocol(b4_stage),
        diagnostic_margins=tuple(
            float(value)
            for value in b4_stage["decision_protocol"]["diagnostic_absolute_margins"]
        ),
    )
    repaired_b4 = {
        **b4,
        "experiment_id": "R005E1A-R1-repaired-legal-domain-one-frame-oracle-v1",
        "parent_b4r1_sha256": sha256_file(args.b4_result),
        "replacement_one_frame_sha256": sha256_file(args.replacement_one_frame),
        "anchor_ledgers": repaired_rows,
        "summary": summary,
        "decision": summary["decision"],
        "context_panel_repair_applied": True,
    }
    _write_json(args.output_root / "repaired_b4_result.json", repaired_b4)

    context = cast(dict[str, Any], json.loads(args.context_roster.read_text("utf-8")))
    context_rows = cast(list[dict[str, Any]], context["anchor_ledger"])
    repaired_context_rows = []
    for row in context_rows:
        if _key(row) != REPAIR_KEY:
            repaired_context_rows.append(row)
            continue
        repaired = deepcopy(row)
        repaired["current_token"] = NEW_CURRENT
        repaired["previous_token"] = OLD_CURRENT
        repaired["gap_s"] = NEW_GAP_S
        repaired_context_rows.append(repaired)
    repaired_context = {
        **context,
        "experiment_id": "R005E1A-R1-repaired-temporal-context-roster-v1",
        "parent_context_roster_sha256": sha256_file(args.context_roster),
        "anchor_ledger": repaired_context_rows,
        "context_panel_repair_applied": True,
    }
    _write_json(args.output_root / "repaired_context_roster.json", repaired_context)
    result = {
        "status": "PASS",
        "repair_key": list(REPAIR_KEY),
        "current_manifest_sha256": sha256_file(current_output / "manifest.json"),
        "previous_manifest_sha256": sha256_file(previous_output / "manifest.json"),
        "repaired_b4_result_sha256": sha256_file(args.output_root / "repaired_b4_result.json"),
        "repaired_context_roster_sha256": sha256_file(
            args.output_root / "repaired_context_roster.json"
        ),
        "b4_decision": summary["decision"],
        "development_calibration_test_navdream_target_accessed": False,
    }
    _write_json(args.output_root / "build_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-current-root", type=Path, required=True)
    parser.add_argument("--original-current-manifest", type=Path, required=True)
    parser.add_argument("--partial-previous-root", type=Path, required=True)
    parser.add_argument("--previous-time-set", type=Path, required=True)
    parser.add_argument("--replacement-current-root", type=Path, required=True)
    parser.add_argument("--replacement-current-manifest", type=Path, required=True)
    parser.add_argument("--b4-result", type=Path, required=True)
    parser.add_argument("--replacement-one-frame", type=Path, required=True)
    parser.add_argument("--context-roster", type=Path, required=True)
    parser.add_argument("--b4-config", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, allow_nan=False))
