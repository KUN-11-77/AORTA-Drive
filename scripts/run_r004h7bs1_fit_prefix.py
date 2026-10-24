"""Run the outcome-blind R004H7-BS1 fit-prefix support qualification."""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import pickle
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.certification.geometry import RectangularGrid
from aorta_drive.evaluation.h4 import (
    IntentEligibilityLimits,
    qualify_intent_match,
    sample_cached_native_trajectory,
    terminal_segment_average_speed,
)
from aorta_drive.evaluation.navsim_execution import execute_official_navsim_batch
from aorta_drive.evaluation.navsim_geometry import vehicle_geometry_contract
from aorta_drive.evaluation.support_adequacy import (
    evaluate_prefix_panel,
    summarize_prefix_qualification,
)
from aorta_drive.vision.feature_cache import sha256_file


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes its declared root")
    return path


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


def _grid(stage: Mapping[str, Any]) -> RectangularGrid:
    values = stage["candidate_panel"]["node_grid"]
    return RectangularGrid(
        x_min=float(values["x_min_m"]),
        y_min=float(values["y_min_m"]),
        columns=int(values["columns"]),
        rows=int(values["rows"]),
        resolution=float(values["resolution_m"]),
    )


def _intent_limits(stage: Mapping[str, Any]) -> IntentEligibilityLimits:
    values = stage["candidate_panel"]["intent_eligibility"]
    return IntentEligibilityLimits(
        minimum_native_terminal_displacement_m=float(
            values["minimum_native_terminal_displacement_m"]
        ),
        minimum_progress_ratio=float(values["progress_ratio"][0]),
        maximum_progress_ratio=float(values["progress_ratio"][1]),
        maximum_terminal_cross_track_m=float(values["maximum_terminal_cross_track_m"]),
        maximum_terminal_heading_difference_rad=float(
            values["maximum_terminal_heading_difference_rad"]
        ),
        maximum_terminal_speed_difference_mps=float(
            values["maximum_terminal_speed_difference_mps"]
        ),
    )


def _metric_records(path: Path, fit_groups: tuple[str, ...]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("records")
    if not isinstance(raw, list):
        raise ValueError("metric manifest lacks records")
    order = {group: index for index, group in enumerate(fit_groups)}
    records = [
        cast(dict[str, Any], row)
        for row in raw
        if isinstance(row, dict) and row.get("recording_group") in order
    ]
    records.sort(key=lambda row: (order[str(row["recording_group"])], int(row["anchor_index"])))
    expected = {(group, anchor) for group in fit_groups for anchor in range(16)}
    observed = {(str(row["recording_group"]), int(row["anchor_index"])) for row in records}
    if observed != expected or len(records) != 672:
        raise ValueError("metric manifest does not exactly cover 42 fit groups x 16 anchors")
    if any(row.get("split") != "train" for row in records):
        raise ValueError("fit-prefix qualification attempted to leave the train split")
    return records


def _vehicle_matches(stage: Mapping[str, Any], contract: Any) -> bool:
    expected = stage["candidate_panel"]["vehicle"]
    return bool(
        contract.width_m == float(expected["width_m"])
        and contract.length_m == float(expected["length_m"])
        and contract.rear_axle_to_center_m == float(expected["rear_axle_to_center_m"])
        and contract.body.rear_overhang == float(expected["rear_overhang_m"])
        and contract.body.front_extent == float(expected["front_extent_m"])
        and contract.body.half_width == float(expected["half_width_m"])
    )


def _verify_implementation(
    stage: Mapping[str, Any], *, repository: Path, navsim_root: Path
) -> None:
    frozen = cast(Mapping[str, Any], stage["implementation_sha256"])
    local = {
        "support_adequacy_source": repository
        / "src/aorta_drive/evaluation/support_adequacy.py",
        "navsim_execution_source": repository
        / "src/aorta_drive/evaluation/navsim_execution.py",
        "native_offset_source": repository / "src/aorta_drive/candidates/native_offset.py",
        "intent_filter_source": repository / "src/aorta_drive/evaluation/h4.py",
        "runner_source": Path(__file__).resolve(),
    }
    for key, path in local.items():
        if sha256_file(path) != frozen[key]:
            raise ValueError(f"H7BS1 implementation hash mismatch: {key}")
    for relative, expected in cast(Mapping[str, str], frozen["official_source"]).items():
        if sha256_file(navsim_root / relative) != expected:
            raise ValueError(f"H7BS1 official source hash mismatch: {relative}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage = cast(dict[str, Any], config["H7BS1"])
    _verify_implementation(
        stage,
        repository=Path(__file__).resolve().parents[1],
        navsim_root=args.navsim_root,
    )
    frozen = cast(dict[str, str], stage["frozen_input_sha256"])
    for key, path in {
        "group_role_manifest": args.group_role_manifest,
        "metric_manifest": args.metric_manifest,
        "h7b_config": args.h7b_config,
    }.items():
        if sha256_file(path) != frozen[key]:
            raise ValueError(f"frozen input hash mismatch: {key}")
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)
    panel_root = args.output_directory / "panels"
    panel_root.mkdir()

    groups = _groups(args.group_role_manifest)
    records = _metric_records(args.metric_manifest, groups)
    grid = _grid(stage)
    limits = _intent_limits(stage)
    prefix_indices = tuple(int(value) for value in stage["prefix_indices"])
    amplitudes = tuple(float(value) for value in stage["candidate_panel"]["challenger_amplitudes_m"])
    comparison_indices = np.arange(0, 41, 5, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for number, record in enumerate(records, start=1):
        cache_path = _inside(args.metric_root, str(record["relative_cache_path"]))
        cache_hash = sha256_file(cache_path)
        if cache_hash != record.get("cache_sha256"):
            raise ValueError("metric cache hash mismatch")
        with lzma.open(cache_path, "rb") as stream:
            cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM metric cache
        contract = vehicle_geometry_contract(cache.ego_state.car_footprint.vehicle_parameters)
        if not _vehicle_matches(stage, contract):
            raise ValueError("official vehicle geometry changed")
        native = sample_cached_native_trajectory(cache)
        desired = generate_native_offset_candidates(
            native.poses,
            amplitudes_m=amplitudes,
            minimum_reference_displacement_m=1.0e-6,
            enforce_desired_curve_dynamic_screen=False,
        )
        if len(desired.candidates) != 6:
            raise ValueError("frozen provider did not emit exactly six challengers")
        desired_array = np.stack([candidate.poses for candidate in desired.candidates])
        execution = execute_official_navsim_batch(cache, desired_array)
        half = execution.realized_local_poses[:, comparison_indices]
        native_half = half[0]
        eligibility = tuple(
            qualify_intent_match(
                half[index],
                native_half,
                challenger_terminal_speed_mps=terminal_segment_average_speed(
                    half[index], interval_s=0.5
                ),
                native_terminal_speed_mps=terminal_segment_average_speed(
                    native_half, interval_s=0.5
                ),
                limits=limits,
            )
            for index in range(1, half.shape[0])
        )
        eligible = tuple(row.eligible for row in eligibility)
        statuses = {
            str(prefix): asdict(
                evaluate_prefix_panel(
                    execution.realized_local_poses,
                    challenger_eligible=eligible,
                    prefix_index=prefix,
                    deterministic_execution=execution.deterministic_replay_bit_exact,
                    grid=grid,
                    body=contract.body,
                )
            )
            for prefix in prefix_indices
        }
        panel_path = panel_root / f"{record['scene_token']}.npz"
        np.savez_compressed(
            panel_path,
            realized_local_poses=execution.realized_local_poses,
            challenger_eligible=np.asarray(eligible, dtype=np.bool_),
        )
        rows.append(
            {
                "recording_group": str(record["recording_group"]),
                "anchor_index": int(record["anchor_index"]),
                "scene_token": str(record["scene_token"]),
                "log_name": str(record["log_name"]),
                "metric_cache_sha256": cache_hash,
                "execution_deterministic_replay_bit_exact": bool(
                    execution.deterministic_replay_bit_exact
                ),
                "intent_eligibility": [asdict(value) for value in eligibility],
                "prefix_statuses": statuses,
                "panel_artifact": panel_path.relative_to(args.output_directory).as_posix(),
                "panel_artifact_sha256": sha256_file(panel_path),
            }
        )
        elapsed = time.perf_counter() - started
        if elapsed > float(stage["resources"]["maximum_wall_minutes"]) * 60.0:
            raise RuntimeError("H7BS1 fit qualification exceeded its wall-time cap")
        output_mib = sum(
            path.stat().st_size
            for path in args.output_directory.rglob("*")
            if path.is_file()
        ) / 2**20
        if output_mib > float(stage["resources"]["maximum_output_mib"]):
            raise RuntimeError("H7BS1 fit qualification exceeded its output-size cap")
        if number % 16 == 0:
            progress = {
                "experiment_id": config["experiment_id"],
                "status": "IN_PROGRESS",
                "phase": "fit_prefix_selection",
                "completed_anchor_count": number,
                "total_anchor_count": 672,
                "field_teacher_outcome_accessed": False,
                "development_accessed": False,
            }
            _write_json(args.output_directory / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)

    summary = summarize_prefix_qualification(
        rows,
        group_names=groups,
        prefix_indices=prefix_indices,
        anchors_per_group=16,
        minimum_valid_anchors_per_group=int(stage["minimum_valid_anchors_per_group"]),
    )
    result = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "phase": "fit_prefix_selection",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "runner_sha256": sha256_file(Path(__file__)),
        "input_hashes": {key: sha256_file(path) for key, path in {
            "group_role_manifest": args.group_role_manifest,
            "metric_manifest": args.metric_manifest,
            "h7b_config": args.h7b_config,
        }.items()},
        "anchor_ledgers": rows,
        "summary": summary,
        "decision": summary["decision"],
        "elapsed_seconds": time.perf_counter() - started,
        "field_teacher_outcome_accessed": False,
        "development_accessed": False,
        "calibration_test_navdream_or_target_accessed": False,
        "claim_boundary": (
            "Fit-only deterministic support qualification; no visual field, teacher, outcome, "
            "decision relevance, driving, safety, NavDream, target, or novelty claim."
        ),
    }
    _write_json(args.output_directory / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--h7b-config", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--navsim-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "anchor_ledgers"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
