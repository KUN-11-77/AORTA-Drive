"""Audit full-score previous-frame availability without reading outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.evaluation.temporal_context_roster import (
    TemporalContextRosterProtocol,
    resolve_previous_context,
    summarize_temporal_context_roster,
)
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4b4_preference_oracle import _groups, _metric_records


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(dict[str, Any], config["H7BS4B4T0"])
    frozen = cast(Mapping[str, str], stage["frozen_input_sha256"])
    for name, path in {
        "group_role_manifest": args.group_role_manifest,
        "metric_manifest": args.metric_manifest,
    }.items():
        if sha256_file(path) != frozen[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")

    groups = _groups(args.group_role_manifest)
    records = _metric_records(args.metric_manifest, groups)
    by_log: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_log[str(record["log_name"])].append(record)

    protocol = TemporalContextRosterProtocol(
        maximum_gap_s=float(stage["adjacency"]["maximum_gap_s"]),
        expected_anchor_count=int(stage["scope"]["total_anchors"]),
        expected_group_count=int(stage["scope"]["recording_groups"]),
        expected_anchors_per_group=int(stage["scope"]["anchors_per_group"]),
    )
    ledger: list[dict[str, Any]] = []
    raw_log_hashes: dict[str, str] = {}
    for log_name, log_records in sorted(by_log.items()):
        path = args.raw_log_root / f"{log_name}.pkl"
        if not path.is_file():
            frames: list[dict[str, Any]] = []
            raw_log_hashes[log_name] = "MISSING"
        else:
            payload = path.read_bytes()
            raw_log_hashes[log_name] = hashlib.sha256(payload).hexdigest()
            loaded = pickle.loads(payload)  # noqa: S301 - official frozen NAVSIM logs
            if not isinstance(loaded, list):
                raise ValueError(f"raw log is not a frame list: {log_name}")
            frames = cast(list[dict[str, Any]], loaded)
        for record in log_records:
            resolved = resolve_previous_context(
                frames,
                current_token=str(record["scene_token"]),
                maximum_gap_s=protocol.maximum_gap_s,
            )
            if raw_log_hashes[log_name] == "MISSING":
                resolved = {
                    "status": "LOG_MISSING",
                    "previous_token": None,
                    "gap_s": None,
                }
            ledger.append(
                {
                    "recording_group": str(record["recording_group"]),
                    "anchor_index": int(record["anchor_index"]),
                    "log_name": log_name,
                    "current_token": str(record["scene_token"]),
                    "raw_log_sha256": raw_log_hashes[log_name],
                    **resolved,
                }
            )

    summary = summarize_temporal_context_roster(
        ledger, group_names=groups, protocol=protocol
    )
    result = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETE",
        "config_sha256": sha256_file(args.config),
        "frozen_input_sha256": dict(frozen),
        "unique_raw_log_count": len(by_log),
        "summary": summary,
        "anchor_ledger": sorted(
            ledger, key=lambda row: (row["recording_group"], row["anchor_index"])
        ),
        "scientific_outcomes_accessed": False,
        "development_calibration_test_navdream_target_accessed": False,
    }
    _write_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--raw-log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

