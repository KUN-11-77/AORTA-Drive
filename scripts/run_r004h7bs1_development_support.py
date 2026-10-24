"""Run the one-shot H7BS1 development support qualification at the frozen prefix."""

from __future__ import annotations

import argparse
import csv
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _development_groups(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        groups = tuple(
            row["recording_group"]
            for row in csv.DictReader(stream)
            if row["split"] == "development"
        )
    if len(groups) != 24 or len(set(groups)) != 24:
        raise ValueError("source manifest does not define 24 unique development groups")
    return groups


def _panel_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("record_count") != 384
        or payload.get("outcome_fields_included") is not False
    ):
        raise ValueError("development panel manifest is not the outcome-free 384-panel bundle")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 384:
        raise ValueError("development panel manifest is incomplete")
    return [cast(dict[str, Any], row) for row in records]


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock: dict[str, Any] = yaml.safe_load(args.lock.read_text(encoding="utf-8"))
    for key, path in {
        "fit_result": args.fit_result,
        "fit_audit": args.fit_audit,
        "development_panel_manifest": args.panel_manifest,
        "source_split_manifest": args.source_split_manifest,
        "base_config": args.base_config,
    }.items():
        if sha256_file(path) != lock["frozen_input_sha256"][key]:
            raise ValueError(f"development lock hash mismatch: {key}")
    repository = Path(__file__).resolve().parents[1]
    for key, path in {
        "support_adequacy_source": repository
        / "src/aorta_drive/evaluation/support_adequacy.py",
        "runner_source": Path(__file__).resolve(),
    }.items():
        if sha256_file(path) != lock["implementation_sha256"][key]:
            raise ValueError(f"development implementation hash mismatch: {key}")

    fit = json.loads(args.fit_result.read_text(encoding="utf-8"))
    audit = json.loads(args.fit_audit.read_text(encoding="utf-8"))
    prefix = int(lock["selected_prefix_index"])
    if (
        fit.get("decision") != "FIT_PREFIX_SELECTED"
        or fit.get("summary", {}).get("selected_prefix_index") != prefix
        or fit.get("development_accessed") is not False
        or audit.get("status") != "PASS"
        or audit.get("result_sha256") != lock["frozen_input_sha256"]["fit_result"]
        or audit.get("recomputed_selected_prefix_index") != prefix
    ):
        raise ValueError("fit result and audit do not authorize the frozen development prefix")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    base: dict[str, Any] = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
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
    groups = _development_groups(args.source_split_manifest)
    records = _panel_records(args.panel_manifest)
    expected = {(group, anchor) for group in groups for anchor in range(16)}
    observed = {(str(row["recording_group"]), int(row["anchor_index"])) for row in records}
    if observed != expected:
        raise ValueError("development panel manifest does not match the frozen 24x16 roster")

    rows: list[dict[str, Any]] = []
    for row in records:
        artifact = (args.panel_root / str(row["panel_artifact"])).resolve()
        if not artifact.is_relative_to(args.panel_root.resolve()) or not artifact.is_file():
            raise ValueError("development panel artifact is missing or escapes its root")
        if sha256_file(artifact) != row["panel_artifact_sha256"]:
            raise ValueError("development panel artifact hash mismatch")
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
        rows.append(
            {
                "recording_group": row["recording_group"],
                "anchor_index": row["anchor_index"],
                "scene_token": row["scene_token"],
                "panel_artifact": row["panel_artifact"],
                "panel_artifact_sha256": row["panel_artifact_sha256"],
                "prefix_statuses": {str(prefix): status},
            }
        )
    summary = summarize_prefix_qualification(
        rows,
        group_names=groups,
        prefix_indices=(prefix,),
        anchors_per_group=16,
        minimum_valid_anchors_per_group=int(stage["minimum_valid_anchors_per_group"]),
    )
    passed = summary["by_prefix"][str(prefix)]["coverage_pass"]
    decision = (
        "DEVELOPMENT_SUPPORT_PASS"
        if passed
        else "STOP_CURRENT_FIXED_GRID_SHADOW_ROUTE"
    )
    result = {
        "experiment_id": lock["experiment_id"],
        "status": "COMPLETED",
        "phase": "one_shot_development_support_qualification",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_prefix_index": prefix,
        "selected_prefix_seconds": prefix / 10.0,
        "anchor_ledgers": rows,
        "summary": summary,
        "decision": decision,
        "field_teacher_outcome_accessed": False,
        "calibration_test_navdream_or_target_accessed": False,
        "claim_boundary": (
            "One-shot development support qualification at a fit-selected prefix; no visual "
            "field, teacher, decision outcome, driving, safety, target, or novelty claim."
        ),
    }
    _write_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--fit-audit", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "anchor_ledgers"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
