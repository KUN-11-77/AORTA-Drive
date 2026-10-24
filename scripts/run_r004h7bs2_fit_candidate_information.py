"""Run the preregistered, outcome-blind R004H7-BS2 fit diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.candidate_information import (
    evaluate_native_challenger_information,
    evaluate_roi_panel,
    summarize_candidate_information,
)
from aorta_drive.vision.feature_cache import sha256_file


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _groups(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = tuple(row["recording_group"] for row in rows if row["role"] == "fit")
    if len(groups) != 42 or len(set(groups)) != 42:
        raise ValueError("role manifest does not define 42 unique fit groups")
    return groups


def _body(stage: Mapping[str, Any]) -> VehicleRectangle:
    vehicle = cast(Mapping[str, Any], stage["vehicle"])
    return VehicleRectangle(
        rear_overhang=float(vehicle["rear_overhang_m"]),
        front_extent=float(vehicle["front_extent_m"]),
        half_width=float(vehicle["half_width_m"]),
    )


def _grids(stage: Mapping[str, Any]) -> dict[str, RectangularGrid]:
    result: dict[str, RectangularGrid] = {}
    for row in cast(list[dict[str, Any]], stage["roi_candidates"]):
        result[str(row["id"])] = RectangularGrid(
            x_min=float(row["x_min_m"]),
            y_min=float(row["y_min_m"]),
            columns=int(row["columns"]),
            rows=int(row["rows"]),
            resolution=float(row["resolution_m"]),
        )
    return result


def _verify(
    config: Mapping[str, Any],
    *,
    fit_result: Path,
    group_role_manifest: Path,
    repository: Path,
) -> None:
    stage = cast(Mapping[str, Any], config["H7BS2"])
    inputs = cast(Mapping[str, str], stage["frozen_input_sha256"])
    for key, path in {
        "h7bs1_fit_result": fit_result,
        "group_role_manifest": group_role_manifest,
    }.items():
        if sha256_file(path) != inputs[key]:
            raise ValueError(f"frozen input hash mismatch: {key}")
    implementation = cast(Mapping[str, str], stage["implementation_sha256"])
    for key, path in {
        "candidate_information_source": repository
        / "src/aorta_drive/evaluation/candidate_information.py",
        "shadow_decision_source": repository
        / "src/aorta_drive/evaluation/shadow_decision.py",
        "runner_source": Path(__file__).resolve(),
    }.items():
        if sha256_file(path) != implementation[key]:
            raise ValueError(f"implementation hash mismatch: {key}")


def compute_rows(
    *,
    config: Mapping[str, Any],
    fit_payload: Mapping[str, Any],
    panel_run_root: Path,
    groups: tuple[str, ...],
) -> list[dict[str, Any]]:
    stage = cast(Mapping[str, Any], config["H7BS2"])
    ledgers = cast(list[dict[str, Any]], fit_payload["anchor_ledgers"])
    group_order = {group: index for index, group in enumerate(groups)}
    ledgers.sort(key=lambda row: (group_order[str(row["recording_group"])], int(row["anchor_index"])))
    expected = {(group, anchor) for group in groups for anchor in range(16)}
    observed = {(str(row["recording_group"]), int(row["anchor_index"])) for row in ledgers}
    if observed != expected or len(ledgers) != 672:
        raise ValueError("H7BS1 fit result does not cover the frozen 42x16 panel")
    body = _body(stage)
    grids = _grids(stage)
    prefix_index = int(stage["prefix_index"])
    thresholds = tuple(float(value) for value in stage["divergence_thresholds_m"])
    lattice = cast(Mapping[str, Any], stage["lattice"])
    rows: list[dict[str, Any]] = []
    for ledger in ledgers:
        artifact = (panel_run_root / str(ledger["panel_artifact"])).resolve()
        if not artifact.is_relative_to(panel_run_root.resolve()):
            raise ValueError("panel artifact escapes frozen run root")
        if sha256_file(artifact) != ledger["panel_artifact_sha256"]:
            raise ValueError("panel artifact hash mismatch")
        with np.load(artifact, allow_pickle=False) as payload:
            trajectories = np.asarray(payload["realized_local_poses"], dtype=np.float64)
            eligible = tuple(bool(value) for value in payload["challenger_eligible"])
        if trajectories.shape != (7, 41, 3) or len(eligible) != 6:
            raise ValueError("frozen panel artifact has unexpected shape")
        if eligible != tuple(bool(value) for value in ledger["challenger_eligible"]):
            raise ValueError("artifact eligibility differs from H7BS1 result")
        pair_information = [
            asdict(
                evaluate_native_challenger_information(
                    trajectories[0],
                    trajectories[index],
                    challenger_index=index,
                    prefix_index=prefix_index,
                    divergence_thresholds_m=thresholds,
                    body=body,
                    lattice_resolution_m=float(lattice["resolution_m"]),
                    lattice_origin_xy_m=(
                        float(lattice["center_origin_x_m"]),
                        float(lattice["center_origin_y_m"]),
                    ),
                )
            )
            for index in range(1, 7)
            if eligible[index - 1]
        ]
        roi_statuses = {
            roi_id: evaluate_roi_panel(
                trajectories,
                challenger_eligible=eligible,
                deterministic_execution=bool(
                    ledger["execution_deterministic_replay_bit_exact"]
                ),
                grid=grid,
                body=body,
            )
            for roi_id, grid in grids.items()
        }
        rows.append(
            {
                "recording_group": str(ledger["recording_group"]),
                "anchor_index": int(ledger["anchor_index"]),
                "scene_token": str(ledger["scene_token"]),
                "panel_artifact": str(ledger["panel_artifact"]),
                "panel_artifact_sha256": str(ledger["panel_artifact_sha256"]),
                "eligible_challenger_count": int(sum(eligible)),
                "pair_information": pair_information,
                "roi_statuses": roi_statuses,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    repository = Path(__file__).resolve().parents[1]
    _verify(
        config,
        fit_result=args.fit_result,
        group_role_manifest=args.group_role_manifest,
        repository=repository,
    )
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)
    fit_payload = cast(dict[str, Any], json.loads(args.fit_result.read_text(encoding="utf-8")))
    groups = _groups(args.group_role_manifest)
    rows = compute_rows(
        config=config,
        fit_payload=fit_payload,
        panel_run_root=args.panel_run_root,
        groups=groups,
    )
    stage = cast(Mapping[str, Any], config["H7BS2"])
    roi_rows = cast(list[dict[str, Any]], stage["roi_candidates"])
    summary = summarize_candidate_information(
        rows,
        group_names=groups,
        anchors_per_group=16,
        prefix_index=int(stage["prefix_index"]),
        primary_divergence_threshold_m=float(stage["primary_divergence_threshold_m"]),
        divergence_thresholds_m=tuple(
            float(value) for value in stage["divergence_thresholds_m"]
        ),
        roi_precedence=tuple(str(row["id"]) for row in roi_rows),
        roi_node_counts={
            str(row["id"]): int(row["columns"]) * int(row["rows"]) for row in roi_rows
        },
        minimum_valid_anchors_per_group=int(stage["minimum_valid_anchors_per_group"]),
        bootstrap_seed=int(stage["statistics"]["bootstrap_seed"]),
        bootstrap_replicates=int(stage["statistics"]["bootstrap_replicates"]),
    )
    elapsed = time.perf_counter() - started
    if elapsed > float(stage["resources"]["maximum_wall_minutes"]) * 60.0:
        raise RuntimeError("H7BS2 exceeded its frozen wall-time cap")
    result = {
        "experiment_id": config["experiment_id"],
        "code_commit": args.code_commit,
        "input_hashes": {
            "config": sha256_file(args.config),
            "h7bs1_fit_result": sha256_file(args.fit_result),
            "group_role_manifest": sha256_file(args.group_role_manifest),
        },
        "forbidden_payload_accessed": False,
        "anchor_ledgers": rows,
        "summary": summary,
        "runtime_seconds": elapsed,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if len(encoded.encode("utf-8")) > float(stage["resources"]["maximum_output_mib"]) * 2**20:
        raise RuntimeError("H7BS2 exceeded its frozen output-size cap")
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--panel-run-root", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    payload = run(parse_args())
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, allow_nan=False))
