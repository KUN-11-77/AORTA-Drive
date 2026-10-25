"""Outcome-free H7BS1 qualification for the single R005 replacement anchor."""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.evaluation.h4 import (
    qualify_intent_match,
    sample_cached_native_trajectory,
    terminal_segment_average_speed,
)
from aorta_drive.evaluation.navsim_execution import execute_official_navsim_batch
from aorta_drive.evaluation.navsim_geometry import vehicle_geometry_contract
from aorta_drive.evaluation.support_adequacy import evaluate_prefix_panel
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs1_fit_prefix import (
    _grid,
    _inside,
    _intent_limits,
    _vehicle_matches,
    _write_json,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.h7bs1_config.read_text("utf-8")))
    stage = cast(dict[str, Any], config["H7BS1"])
    manifest = cast(dict[str, Any], json.loads(args.metric_manifest.read_text("utf-8")))
    records = [
        row
        for row in manifest["records"]
        if row["recording_group"] == "2021.06.07.19.29.59_veh-38"
        and int(row["anchor_index"]) == 4
    ]
    if len(records) != 1 or records[0]["scene_token"] != "e25e0f03413553f9":
        raise ValueError("replacement current cache record is not the frozen anchor")
    record = records[0]
    cache_path = _inside(args.metric_root, str(record["relative_cache_path"]))
    if sha256_file(cache_path) != record["cache_sha256"]:
        raise ValueError("replacement metric-cache hash mismatch")
    with lzma.open(cache_path, "rb") as stream:
        cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM cache

    contract = vehicle_geometry_contract(cache.ego_state.car_footprint.vehicle_parameters)
    if not _vehicle_matches(stage, contract):
        raise ValueError("official vehicle geometry changed")
    native = sample_cached_native_trajectory(cache)
    desired = generate_native_offset_candidates(
        native.poses,
        amplitudes_m=tuple(
            float(value) for value in stage["candidate_panel"]["challenger_amplitudes_m"]
        ),
        minimum_reference_displacement_m=1.0e-6,
        enforce_desired_curve_dynamic_screen=False,
    )
    if len(desired.candidates) != 6:
        raise ValueError("frozen provider did not emit six challengers")
    desired_array = np.stack([candidate.poses for candidate in desired.candidates])
    execution = execute_official_navsim_batch(cache, desired_array)
    comparison_indices = np.arange(0, 41, 5, dtype=np.int64)
    half = execution.realized_local_poses[:, comparison_indices]
    limits = _intent_limits(stage)
    eligibility = tuple(
        qualify_intent_match(
            half[index],
            half[0],
            challenger_terminal_speed_mps=terminal_segment_average_speed(
                half[index], interval_s=0.5
            ),
            native_terminal_speed_mps=terminal_segment_average_speed(
                half[0], interval_s=0.5
            ),
            limits=limits,
        )
        for index in range(1, half.shape[0])
    )
    eligible = tuple(value.eligible for value in eligibility)
    selected_prefix = 25
    status = evaluate_prefix_panel(
        execution.realized_local_poses,
        challenger_eligible=eligible,
        prefix_index=selected_prefix,
        deterministic_execution=execution.deterministic_replay_bit_exact,
        grid=_grid(stage),
        body=contract.body,
    )
    args.output_directory.mkdir(parents=True, exist_ok=False)
    panel_path = args.output_directory / "panels" / f"{record['scene_token']}.npz"
    panel_path.parent.mkdir()
    np.savez_compressed(
        panel_path,
        realized_local_poses=execution.realized_local_poses,
        challenger_eligible=np.asarray(eligible, dtype=np.bool_),
    )
    passed = bool(status.panel_valid)
    result = {
        "experiment_id": "R005E1A-R1-replacement-anchor-support-v1",
        "status": "PASS" if passed else "STOP",
        "decision": "REPLACEMENT_SUPPORT_PASS" if passed else "STOP_REPLACEMENT_SUPPORT_INVALID",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "h7bs1_config_sha256": sha256_file(args.h7bs1_config),
        "metric_manifest_sha256": sha256_file(args.metric_manifest),
        "recording_group": str(record["recording_group"]),
        "anchor_index": int(record["anchor_index"]),
        "scene_token": str(record["scene_token"]),
        "previous_token": "2c18d1604abe52b8",
        "metric_cache_sha256": str(record["cache_sha256"]),
        "execution_deterministic_replay_bit_exact": bool(
            execution.deterministic_replay_bit_exact
        ),
        "intent_eligibility": [asdict(value) for value in eligibility],
        "frozen_challenger_eligible": list(eligible),
        "selected_prefix_index": selected_prefix,
        "selected_prefix_seconds": 2.5,
        "selected_prefix_status": asdict(status),
        "panel_artifact": panel_path.relative_to(args.output_directory).as_posix(),
        "panel_artifact_sha256": sha256_file(panel_path),
        "scientific_outcomes_accessed": False,
        "development_calibration_test_navdream_target_accessed": False,
    }
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h7bs1-config", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, allow_nan=False))
