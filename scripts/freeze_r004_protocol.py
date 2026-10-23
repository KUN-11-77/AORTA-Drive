"""Freeze and audit the R004 source split and finite session time set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from aorta_drive.protocol.source_split import (
    SourceLogRecord,
    build_split_audit,
    canonical_sha256,
    midpoint_anchor_positions,
)

EXPECTED_GROUP_COUNTS = {
    "train": 66,
    "development": 24,
    "calibration": 40,
    "test": 32,
}
_LOG_PATTERN = re.compile(r"^(?P<group>.*_veh-\d+)_(?P<start>\d+)_(?P<end>\d+)$")
_GROUP_PATTERN = re.compile(r"^(?P<year>\d{4})\.(?P<month>\d{2})\.\d{2}.*_(?P<vehicle>veh-\d+)$")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def freeze_protocol(
    roster_path: Path,
    token_index_path: Path,
    assignment_path: Path,
    output_dir: Path,
    *,
    anchors_per_session: int,
) -> dict[str, Any]:
    """Validate source provenance and write the three frozen R004 manifests."""

    roster = _load_json(roster_path)
    token_index = _load_json(token_index_path)
    allocation = _load_json(assignment_path)
    if roster["source_config_sha256"] != token_index["source_config_sha256"]:
        raise ValueError("roster and token index use different source configurations")
    if token_index["official_log_count"] != 1192:
        raise ValueError("unexpected official source log count")
    if token_index["official_token_count"] != 103288:
        raise ValueError("unexpected official source token count")
    if allocation["source_roster_logical_sha256"] != roster["logical_sha256"]:
        raise ValueError("split assignment does not commit to this source roster")

    source_records = tuple(SourceLogRecord(**record) for record in roster["records"])
    assignments = {str(key): str(value) for key, value in allocation["assignments"].items()}
    split_audit = build_split_audit(source_records, assignments, EXPECTED_GROUP_COUNTS)

    index_by_log = {record["log_name"]: record for record in token_index["records"]}
    roster_by_log = {record.log_name: record for record in source_records}
    if set(index_by_log) != set(roster_by_log):
        raise ValueError("token index and roster log sets differ")

    token_owner: dict[str, str] = {}
    logs_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for log_name in sorted(roster_by_log):
        roster_record = roster_by_log[log_name]
        index_record = index_by_log[log_name]
        if index_record["recording_group"] != roster_record.recording_group:
            raise ValueError(f"recording group mismatch for {log_name}")
        tokens = [str(token) for token in index_record["tokens"]]
        if len(tokens) != roster_record.scene_count:
            raise ValueError(f"scene count mismatch for {log_name}")
        match = _LOG_PATTERN.fullmatch(log_name)
        if match is None:
            raise ValueError(f"invalid source log name: {log_name}")
        index_record = {
            **index_record,
            "fragment_start": int(match.group("start")),
            "fragment_end": int(match.group("end")),
            "tokens": tokens,
        }
        logs_by_group[roster_record.recording_group].append(index_record)
        for token in tokens:
            prior = token_owner.setdefault(token, log_name)
            if prior != log_name:
                raise ValueError(f"token {token} occurs in both {prior} and {log_name}")
    if len(token_owner) != token_index["official_token_count"]:
        raise ValueError("token index cardinality does not match official source config")

    group_rows: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    for group in sorted(logs_by_group):
        match = _GROUP_PATTERN.fullmatch(group)
        if match is None:
            raise ValueError(f"invalid recording group: {group}")
        split = assignments[group]
        ordered_logs = sorted(
            logs_by_group[group],
            key=lambda item: (item["fragment_start"], item["fragment_end"], item["log_name"]),
        )
        ordered_tokens: list[str] = []
        token_to_log: dict[str, str] = {}
        for item in ordered_logs:
            for token in item["tokens"]:
                ordered_tokens.append(token)
                token_to_log[token] = item["log_name"]
        positions = midpoint_anchor_positions(len(ordered_tokens), anchors_per_session)
        for anchor_index, position in enumerate(positions):
            token = ordered_tokens[position]
            anchor_rows.append(
                {
                    "recording_group": group,
                    "split": split,
                    "anchor_index": anchor_index,
                    "session_position": position,
                    "scene_token": token,
                    "log_name": token_to_log[token],
                }
            )
        group_rows.append(
            {
                "recording_group": group,
                "split": split,
                "scene_count": len(ordered_tokens),
                "log_count": len(ordered_logs),
                "month": f"{match.group('year')}.{match.group('month')}",
                "vehicle": match.group("vehicle"),
            }
        )
        for item in ordered_logs:
            record = roster_by_log[item["log_name"]]
            log_rows.append(
                {
                    "log_name": record.log_name,
                    "recording_group": group,
                    "split": split,
                    "scene_count": record.scene_count,
                    "token_sha256": item["token_sha256"],
                    "log_sha256": record.log_sha256,
                    "sensor_dir_exists": str(record.sensor_dir_exists).lower(),
                }
            )

    anchor_counts = Counter(str(row["split"]) for row in anchor_rows)
    expected_anchor_counts = {
        split: count * anchors_per_session for split, count in EXPECTED_GROUP_COUNTS.items()
    }
    if dict(anchor_counts) != expected_anchor_counts:
        raise AssertionError("anchor counts do not match the frozen group allocation")
    if len({row["scene_token"] for row in anchor_rows}) != len(anchor_rows):
        raise ValueError("frozen time set contains duplicate scene tokens")

    group_path = output_dir / "r004_recording_group_split.csv"
    log_path = output_dir / "r004_source_log_manifest.csv"
    anchor_path = output_dir / "r004_session_time_set.csv"
    _write_csv(
        group_path,
        ["recording_group", "split", "scene_count", "log_count", "month", "vehicle"],
        group_rows,
    )
    _write_csv(
        log_path,
        [
            "log_name",
            "recording_group",
            "split",
            "scene_count",
            "token_sha256",
            "log_sha256",
            "sensor_dir_exists",
        ],
        log_rows,
    )
    _write_csv(
        anchor_path,
        [
            "recording_group",
            "split",
            "anchor_index",
            "session_position",
            "scene_token",
            "log_name",
        ],
        anchor_rows,
    )

    summary = {
        "schema_version": 1,
        "independent_unit": "recording_group",
        "time_set_rule": "floor((2*j+1)*N/(2*m)); zero-based; chronological",
        "anchors_per_session": anchors_per_session,
        "source_config_sha256": roster["source_config_sha256"],
        "source_roster_logical_sha256": roster["logical_sha256"],
        "source_token_index_logical_sha256": token_index["logical_sha256"],
        "allocation_logical_sha256": allocation["logical_sha256"],
        "allocation_solver_status": allocation["solver_status"],
        "split_audit": {
            "group_counts": dict(split_audit.group_counts),
            "log_counts": dict(split_audit.log_counts),
            "scene_counts": dict(split_audit.scene_counts),
            "total_groups": split_audit.total_groups,
            "total_logs": split_audit.total_logs,
            "total_scenes": split_audit.total_scenes,
            "logical_sha256": split_audit.logical_sha256,
        },
        "anchor_counts": expected_anchor_counts,
        "total_anchors": len(anchor_rows),
        "manifests": {
            group_path.name: _file_sha256(group_path),
            log_path.name: _file_sha256(log_path),
            anchor_path.name: _file_sha256(anchor_path),
        },
    }
    summary["logical_sha256"] = canonical_sha256(summary)
    summary_path = output_dir / "r004_protocol_fingerprint.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-roster", type=Path, required=True)
    parser.add_argument("--token-index", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchors-per-session", type=int, default=16)
    args = parser.parse_args()
    summary = freeze_protocol(
        args.source_roster,
        args.token_index,
        args.assignment,
        args.output_dir,
        anchors_per_session=args.anchors_per_session,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
