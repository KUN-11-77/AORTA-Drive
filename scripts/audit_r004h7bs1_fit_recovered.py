"""Mechanical audit for a materialized H7BS1 fit result."""

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
from run_r004h7bs1_fit_prefix import _groups


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage = cast(dict[str, Any], config["H7BS1"])
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if (
        result.get("status") != "COMPLETED_AFTER_REGISTERED_SERIALIZATION_RECOVERY"
        or result.get("phase") != "fit_prefix_selection"
        or result.get("all_execution_replays_match_artifacts_bit_exact") is not True
        or result.get("field_teacher_outcome_accessed") is not False
        or result.get("development_accessed") is not False
        or result.get("calibration_test_navdream_or_target_accessed") is not False
    ):
        raise ValueError("result is not the registered recovered fit-only qualification")
    stored_rows = result.get("anchor_ledgers")
    if not isinstance(stored_rows, list) or len(stored_rows) != 672:
        raise ValueError("fit result does not contain 672 anchor ledgers")
    if any(
        not bool(cast(Mapping[str, Any], row)["recovery_replay_matches_artifact_bit_exact"])
        for row in stored_rows
    ):
        raise ValueError("a recovered panel lacks bit-exact official execution replay")

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
    prefixes = tuple(int(value) for value in stage["prefix_indices"])
    replayed_rows: list[dict[str, Any]] = []
    artifact_failures = 0
    status_failures = 0
    for raw in stored_rows:
        row = cast(dict[str, Any], raw)
        artifact = (args.run_directory / str(row["panel_artifact"])).resolve()
        if not artifact.is_relative_to(args.run_directory.resolve()) or not artifact.is_file():
            artifact_failures += 1
            continue
        if sha256_file(artifact) != row["panel_artifact_sha256"]:
            artifact_failures += 1
            continue
        with np.load(artifact, allow_pickle=False) as archive:
            realized = np.asarray(archive["realized_local_poses"], dtype=np.float64)
            eligible = tuple(bool(value) for value in archive["challenger_eligible"])
        statuses = {
            str(prefix): vars(
                evaluate_prefix_panel(
                    realized,
                    challenger_eligible=eligible,
                    prefix_index=prefix,
                    deterministic_execution=True,
                    grid=grid,
                    body=body,
                )
            )
            for prefix in prefixes
        }
        normalized = json.loads(json.dumps(statuses))
        if normalized != row["prefix_statuses"]:
            status_failures += 1
        replayed_rows.append(
            {
                "recording_group": row["recording_group"],
                "anchor_index": row["anchor_index"],
                "prefix_statuses": normalized,
            }
        )
    groups = _groups(args.group_role_manifest)
    replayed = summarize_prefix_qualification(
        replayed_rows,
        group_names=groups,
        prefix_indices=prefixes,
        anchors_per_group=16,
        minimum_valid_anchors_per_group=int(stage["minimum_valid_anchors_per_group"]),
    )
    summary_match = replayed == result["summary"]
    passed = not artifact_failures and not status_failures and summary_match
    output = {
        "experiment_id": config["experiment_id"],
        "status": "PASS" if passed else "FAIL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_sha256": sha256_file(args.result),
        "config_sha256": sha256_file(args.config),
        "panel_artifact_count": len(replayed_rows),
        "artifact_failure_count": artifact_failures,
        "prefix_status_recomputation_failure_count": status_failures,
        "summary_reproduced": summary_match,
        "official_execution_replay_bound_by_materializer": True,
        "recomputed_decision": replayed["decision"],
        "recomputed_selected_prefix_index": replayed["selected_prefix_index"],
        "field_teacher_outcome_accessed": False,
        "development_accessed": False,
        "calibration_test_navdream_or_target_accessed": False,
    }
    _write_json(args.output, output)
    if not passed:
        raise AssertionError("recovered H7BS1 fit-prefix mechanical audit failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
