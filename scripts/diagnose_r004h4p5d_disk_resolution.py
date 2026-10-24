"""Diagnose the registered R004H4P5I disk-certificate resolution barrier."""

from __future__ import annotations

import argparse
import json
import lzma
import math
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.certification.geometry import (
    Pose2D,
    RectangularGrid,
    certify_disk_noninclusion_resolution_barrier,
)
from aorta_drive.evaluation.h4 import sample_cached_native_trajectory
from aorta_drive.evaluation.navsim_execution import execute_official_navsim_batch
from aorta_drive.evaluation.navsim_geometry import vehicle_geometry_contract
from diagnose_r004h4p2_candidate_supply import _metric_records, _sha256
from run_r004h4p5i_realized_geometry import _selected_sentinels


def _predecessor_rows(result: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = result.get("sentinel_ledger")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("predecessor result must contain eight sentinel rows")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("predecessor sentinel row is not a mapping")
        key = (str(row.get("log_name")), str(row.get("scene_token")))
        if key in indexed:
            raise ValueError("predecessor sentinel rows are not unique")
        indexed[key] = row
    return indexed


def _verify_sources(config: dict[str, Any], repository: Path) -> None:
    paths = {
        "geometry_source_sha256": repository / "src/aorta_drive/certification/geometry.py",
        "navsim_execution_source_sha256": repository
        / "src/aorta_drive/evaluation/navsim_execution.py",
        "native_offset_source_sha256": repository
        / "src/aorta_drive/candidates/native_offset.py",
    }
    for key, path in paths.items():
        if _sha256(path) != config["implementation"][key]:
            raise ValueError(f"local source hash mismatch: {key}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for name, path in (
        ("time_set", args.time_set),
        ("role_manifest", args.role_manifest),
        ("metric_manifest", args.metric_manifest),
    ):
        if _sha256(path) != config["inputs"][f"{name}_sha256"]:
            raise ValueError(f"{name} hash disagrees with registration")
    if _sha256(args.predecessor_result) != config["predecessor"]["raw_result_sha256"]:
        raise ValueError("predecessor result hash disagrees with registration")
    repository = Path(__file__).resolve().parents[1]
    _verify_sources(config, repository)

    predecessor = json.loads(args.predecessor_result.read_text(encoding="utf-8"))
    if predecessor.get("experiment_id") != config["predecessor"]["experiment"]:
        raise ValueError("predecessor experiment identity disagrees with registration")
    if predecessor.get("decision") != config["predecessor"]["decision"]:
        raise ValueError("predecessor decision disagrees with registration")
    predecessor_rows = _predecessor_rows(predecessor)
    anchors = _selected_sentinels(args.time_set, args.role_manifest)
    metric_records = _metric_records(
        json.loads(args.metric_manifest.read_text(encoding="utf-8"))
    )
    grid_config = config["geometry"]["grid"]
    grid = RectangularGrid(
        x_min=float(grid_config["x_min_m"]),
        y_min=float(grid_config["y_min_m"]),
        columns=int(grid_config["columns"]),
        rows=int(grid_config["rows"]),
        resolution=float(grid_config["resolution_m"]),
    )
    amplitudes = tuple(float(value) for value in config["provider"]["amplitudes_m"])

    rows: list[dict[str, Any]] = []
    implementation_failures: list[str] = []
    observed_zero_count = 0
    explained_zero_count = 0
    maximum_separation = 0.0
    minimum_barrier = math.inf
    for anchor in anchors:
        key = (anchor["log_name"], anchor["scene_token"])
        metric_record = metric_records.get(key)
        if metric_record is None:
            raise ValueError("metric manifest lacks a registered sentinel")
        cache_path = args.metric_root / str(metric_record["relative_cache_path"])
        if _sha256(cache_path) != metric_record["cache_sha256"]:
            raise ValueError("metric cache raw hash mismatch")
        with lzma.open(cache_path, "rb") as stream:
            cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM input
        native = sample_cached_native_trajectory(cache)
        desired_batch = generate_native_offset_candidates(
            native.poses,
            amplitudes_m=amplitudes,
            minimum_reference_displacement_m=1e-6,
            enforce_desired_curve_dynamic_screen=False,
        )
        desired = np.stack([candidate.poses for candidate in desired_batch.candidates])
        execution = execute_official_navsim_batch(cache, desired)
        contract = vehicle_geometry_contract(
            cache.ego_state.car_footprint.vehicle_parameters
        )
        prior_row = predecessor_rows.get(key)
        if prior_row is None:
            raise ValueError("predecessor result lacks a registered sentinel")
        prior_candidates = prior_row.get("candidate_ledger")
        if not isinstance(prior_candidates, list) or len(prior_candidates) != len(amplitudes):
            raise ValueError("predecessor candidate ledger is incomplete")

        native_poses = tuple(Pose2D(*pose) for pose in execution.realized_local_poses[0])
        candidate_rows: list[dict[str, Any]] = []
        for candidate_index, (candidate, prior_candidate) in enumerate(
            zip(desired_batch.candidates, prior_candidates, strict=True), start=1
        ):
            if float(prior_candidate.get("amplitude_m", math.nan)) != candidate.control.amplitude_m:
                raise ValueError("predecessor candidate amplitude disagrees with registration")
            challenger_poses = tuple(
                Pose2D(*pose) for pose in execution.realized_local_poses[candidate_index]
            )
            barrier = certify_disk_noninclusion_resolution_barrier(
                contract.body,
                challenger_poses,
                native_poses,
                grid,
                pose_error=execution.maximum_transform_error_m,
            )
            prior_geometry_valid = prior_candidate.get("geometry_valid") is True
            prior_bidirectional_zero = (
                prior_geometry_valid
                and float(
                    prior_candidate.get(
                        "challenger_minus_native_area_lower_m2", math.nan
                    )
                )
                == 0.0
                and float(
                    prior_candidate.get(
                        "native_minus_challenger_area_lower_m2", math.nan
                    )
                )
                == 0.0
            )
            explained = prior_bidirectional_zero and barrier.guarantees_zero_witnesses
            observed_zero_count += int(prior_bidirectional_zero)
            explained_zero_count += int(explained)
            maximum_separation = max(
                maximum_separation, barrier.maximum_paired_separation_upper_m
            )
            minimum_barrier = min(
                minimum_barrier, barrier.no_witness_threshold_lower_m
            )
            candidate_rows.append(
                {
                    "amplitude_m": candidate.control.amplitude_m,
                    "predecessor_geometry_valid": prior_geometry_valid,
                    "predecessor_bidirectional_zero": prior_bidirectional_zero,
                    "maximum_paired_separation_upper_m": (
                        barrier.maximum_paired_separation_upper_m
                    ),
                    "no_witness_threshold_lower_m": (
                        barrier.no_witness_threshold_lower_m
                    ),
                    "barrier_guarantees_zero_witnesses": (
                        barrier.guarantees_zero_witnesses
                    ),
                    "barrier_explains_predecessor_zero": explained,
                }
            )
        if not execution.deterministic_replay_bit_exact:
            implementation_failures.append(f"{key}: official replay is not bit exact")
        rows.append(
            {
                "recording_group": anchor["recording_group"],
                "anchor_index": int(anchor["anchor_index"]),
                "log_name": anchor["log_name"],
                "scene_token": anchor["scene_token"],
                "execution_deterministic_replay_bit_exact": (
                    execution.deterministic_replay_bit_exact
                ),
                "candidate_ledger": candidate_rows,
            }
        )
        print(json.dumps({"sentinels_completed": len(rows)}), flush=True)

    expected_zero = int(
        config["predecessor"]["observed_bidirectional_zero_candidate_count"]
    )
    ledger_complete = len(rows) == 8 and all(
        len(row["candidate_ledger"]) == 6 for row in rows
    )
    predecessor_bound = observed_zero_count == expected_zero
    implementation_passed = not implementation_failures and ledger_complete and predecessor_bound
    all_explained = observed_zero_count > 0 and explained_zero_count == observed_zero_count
    if not implementation_passed:
        decision = "STOP_DISK_RESOLUTION_DIAGNOSIS_IMPLEMENTATION"
    elif all_explained:
        decision = "RETIRE_DISK_NONINCLUSION_AND_QUALIFY_ORIENTED_RECTANGLE"
    else:
        decision = "STOP_UNEXPLAINED_DISK_ZERO"
    return {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": decision,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": _sha256(args.config),
        "elapsed_seconds": time.perf_counter() - started,
        "data_scope": {
            "role": "fit",
            "recording_group_count": 8,
            "anchor_count": 8,
            "candidate_count": 48,
            "label_payload_read": False,
            "prevalence_or_performance_claim": False,
            "development_calibration_test_or_target_accessed": False,
        },
        "implementation_failures": implementation_failures,
        "diagnostic_counts": {
            "predecessor_bidirectional_zero_candidates": observed_zero_count,
            "zero_candidates_explained_by_barrier": explained_zero_count,
        },
        "maximum_paired_separation_upper_m": maximum_separation,
        "minimum_no_witness_threshold_lower_m": minimum_barrier,
        "sentinel_ledger": rows,
        "claims": {
            "disk_certificate_resolution_barrier_verified": (
                implementation_passed and all_explained
            ),
            "physical_opportunity_absent": False,
            "prevalence_estimated": False,
            "method_performance_evaluated": False,
            "coverage_established": False,
            "gate_b_passed": False,
            "development_access_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--predecessor-result", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "sentinel_ledger"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
