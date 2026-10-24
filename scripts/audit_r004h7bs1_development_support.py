"""Independent artifact replay of one-shot H7BS1 development support qualification."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.support_adequacy import (
    evaluate_prefix_panel,
    summarize_prefix_qualification,
)
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs1_development_support import _development_groups


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    lock: dict[str, Any] = yaml.safe_load(args.lock.read_text(encoding="utf-8"))
    base: dict[str, Any] = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    prefix = int(lock["selected_prefix_index"])
    if (
        result.get("status") != "COMPLETED"
        or result.get("phase") != "one_shot_development_support_qualification"
        or result.get("selected_prefix_index") != prefix
        or result.get("field_teacher_outcome_accessed") is not False
        or result.get("calibration_test_navdream_or_target_accessed") is not False
    ):
        raise ValueError("result is not the frozen development support qualification")
    rows = result.get("anchor_ledgers")
    if not isinstance(rows, list) or len(rows) != 384:
        raise ValueError("development result lacks 384 anchor ledgers")
    stage = base["H7BS1"]
    grid_values = stage["candidate_panel"]["node_grid"]
    grid = RectangularGrid(
        x_min=float(grid_values["x_min_m"]),
        y_min=float(grid_values["y_min_m"]),
        columns=int(grid_values["columns"]),
        rows=int(grid_values["rows"]),
        resolution=float(grid_values["resolution_m"]),
    )
    vehicle = stage["candidate_panel"]["vehicle"]
    body = VehicleRectangle(
        rear_overhang=float(vehicle["rear_overhang_m"]),
        front_extent=float(vehicle["front_extent_m"]),
        half_width=float(vehicle["half_width_m"]),
    )
    replayed: list[dict[str, Any]] = []
    artifact_failures = 0
    status_failures = 0
    for raw in rows:
        row = cast(dict[str, Any], raw)
        artifact = (args.panel_root / str(row["panel_artifact"])).resolve()
        if not artifact.is_relative_to(args.panel_root.resolve()) or not artifact.is_file():
            artifact_failures += 1
            continue
        if sha256_file(artifact) != row["panel_artifact_sha256"]:
            artifact_failures += 1
            continue
        with np.load(artifact, allow_pickle=False) as archive:
            realized = np.asarray(archive["realized_local_poses"], dtype=np.float64)
            eligible = tuple(bool(value) for value in archive["challenger_eligible"])
        status = vars(
            evaluate_prefix_panel(
                realized,
                challenger_eligible=eligible,
                prefix_index=prefix,
                deterministic_execution=True,
                grid=grid,
                body=body,
            )
        )
        normalized = json.loads(json.dumps(status))
        if normalized != row["prefix_statuses"][str(prefix)]:
            status_failures += 1
        replayed.append(
            {
                "recording_group": row["recording_group"],
                "anchor_index": row["anchor_index"],
                "prefix_statuses": {str(prefix): normalized},
            }
        )
    groups = _development_groups(args.source_split_manifest)
    summary = summarize_prefix_qualification(
        replayed,
        group_names=groups,
        prefix_indices=(prefix,),
        anchors_per_group=16,
        minimum_valid_anchors_per_group=int(stage["minimum_valid_anchors_per_group"]),
    )
    summary_match = summary == result["summary"]
    expected_decision = (
        "DEVELOPMENT_SUPPORT_PASS"
        if summary["by_prefix"][str(prefix)]["coverage_pass"]
        else "STOP_CURRENT_FIXED_GRID_SHADOW_ROUTE"
    )
    passed = (
        not artifact_failures
        and not status_failures
        and summary_match
        and expected_decision == result["decision"]
    )
    output = {
        "experiment_id": lock["experiment_id"],
        "status": "PASS" if passed else "FAIL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_sha256": sha256_file(args.result),
        "panel_artifact_count": len(replayed),
        "artifact_failure_count": artifact_failures,
        "prefix_status_recomputation_failure_count": status_failures,
        "summary_reproduced": summary_match,
        "recomputed_decision": expected_decision,
        "recomputed_selected_prefix_index": prefix,
        "field_teacher_outcome_accessed": False,
        "calibration_test_navdream_or_target_accessed": False,
    }
    _write_json(args.output, output)
    if not passed:
        raise AssertionError("H7BS1 development support mechanical audit failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
