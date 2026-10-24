"""Run the registered one-anchor R004H4 continuous-opportunity preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import pickle
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_kinematic_candidates
from aorta_drive.certification.geometry import (
    Pose2D,
    RectangularGrid,
    VehicleRectangle,
    certify_disk_noninclusion_from_covers,
    certify_native_inner_disk_nodes,
    certify_swept_disk_cover,
    certify_swept_inner_disk_cells,
)
from aorta_drive.certification.opportunity import (
    continuous_teacher_improvement_lower_bound,
)
from aorta_drive.evaluation.h4 import (
    IntentEligibilityLimits,
    qualify_intent_match,
    sample_cached_native_trajectory,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _selected_anchor(
    time_set: Path, role_manifest: Path
) -> tuple[dict[str, str], dict[str, str]]:
    role_rows = _load_rows(role_manifest)
    fit_row = next((row for row in role_rows if row["role"] == "fit"), None)
    if fit_row is None:
        raise ValueError("role manifest contains no fit group")
    matches = [
        row
        for row in _load_rows(time_set)
        if row["recording_group"] == fit_row["recording_group"]
    ]
    if len(matches) != 16 or {int(row["anchor_index"]) for row in matches} != set(range(16)):
        raise ValueError("selected fit group does not contain exact anchor indices 0..15")
    return fit_row, min(matches, key=lambda row: int(row["anchor_index"]))


def _manifest_record(
    manifest: dict[str, Any], log_name: str, scene_token: str, key_name: str
) -> dict[str, Any]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("manifest lacks a records list")
    matches: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("manifest contains a non-object record")
        if key_name == "cache_id":
            if item.get("cache_id") == f"{log_name}/{scene_token}":
                matches.append(item)
        elif item.get("log_name") == log_name and item.get("scene_token") == scene_token:
            matches.append(item)
    if len(matches) != 1:
        raise ValueError("manifest does not contain exactly one selected anchor record")
    return matches[0]


def _load_teacher(
    label_path: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    with np.load(label_path, allow_pickle=False) as archive:
        required = {
            "label_m",
            "transform_error_m",
            "distance_error_m",
            "total_error_m",
        }
        if set(archive.files) != required:
            raise ValueError("teacher artifact arrays do not match the frozen contract")
        label = np.asarray(archive["label_m"])
        error = np.asarray(archive["total_error_m"])
    if label.dtype != np.float64 or error.dtype != np.float64:
        raise ValueError("teacher arrays must be float64")
    if label.shape != (64, 64) or error.shape != (64, 64):
        raise ValueError("teacher arrays must have shape [64, 64]")
    if not np.all(np.isfinite(label)) or not np.all(np.isfinite(error)):
        raise ValueError("teacher arrays must be finite")
    if np.any(error < 0.0):
        raise ValueError("teacher errors must be non-negative")
    return label.reshape(-1), error.reshape(-1)


def _raw_speed(log_path: Path, scene_token: str) -> float:
    with log_path.open("rb") as stream:
        frames = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM input
    matches = [frame for frame in frames if str(frame["token"]) == scene_token]
    if len(matches) != 1:
        raise ValueError("raw log does not contain exactly one selected scene token")
    dynamic = np.asarray(matches[0]["ego_dynamic_state"], dtype=np.float64)
    if dynamic.ndim != 1 or dynamic.size < 2 or not np.all(np.isfinite(dynamic[:2])):
        raise ValueError("raw ego dynamic state is invalid")
    return float(np.linalg.norm(dynamic[:2]))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected = config["inputs"]
    for name, path in (
        ("time_set", args.time_set),
        ("role_manifest", args.role_manifest),
        ("metric_manifest", args.metric_manifest),
        ("label_manifest", args.label_manifest),
    ):
        expected_hash = expected[f"{name}_sha256"]
        if _sha256(path) != expected_hash:
            raise ValueError(f"{name} hash disagrees with registration")

    fit_role, anchor = _selected_anchor(args.time_set, args.role_manifest)
    if anchor["split"] != "train" or fit_role["role"] != "fit":
        raise ValueError("preflight selected a prohibited role or split")
    log_name = anchor["log_name"]
    scene_token = anchor["scene_token"]

    metric_manifest = json.loads(args.metric_manifest.read_text(encoding="utf-8"))
    metric_record = _manifest_record(metric_manifest, log_name, scene_token, "scene_key")
    metric_path = args.metric_root / str(metric_record["relative_cache_path"])
    if _sha256(metric_path) != metric_record["cache_sha256"]:
        raise ValueError("selected metric cache raw hash mismatch")
    with lzma.open(metric_path, "rb") as stream:
        cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM input

    label_manifest = json.loads(args.label_manifest.read_text(encoding="utf-8"))
    label_record = _manifest_record(label_manifest, log_name, scene_token, "cache_id")
    label_path = args.label_root / "labels" / str(label_record["artifact_name"])
    if _sha256(label_path) != label_record["artifact_sha256"]:
        raise ValueError("selected teacher artifact hash mismatch")
    labels, errors = _load_teacher(label_path)

    native = sample_cached_native_trajectory(cache)
    native_array = native.poses
    native_terminal_speed = native.terminal_speed_mps
    native_pose_error = native.maximum_transform_error_m
    native_hash = native.canonical_sample_sha256
    initial_speed = _raw_speed(args.log_root / f"{log_name}.pkl", scene_token)
    first = generate_kinematic_candidates(initial_speed)
    replay = generate_kinematic_candidates(initial_speed)
    if len(first.candidates) != len(replay.candidates) or any(
        left.control != right.control or not np.array_equal(left.poses, right.poses)
        for left, right in zip(first.candidates, replay.candidates, strict=True)
    ):
        raise AssertionError("candidate provider failed deterministic replay")

    grid_config = config["geometry"]["grid"]
    grid = RectangularGrid(
        x_min=float(grid_config["x_min_m"]),
        y_min=float(grid_config["y_min_m"]),
        columns=int(grid_config["columns"]),
        rows=int(grid_config["rows"]),
        resolution=float(grid_config["resolution_m"]),
    )
    vehicle_config = config["geometry"]["vehicle"]
    body = VehicleRectangle(
        rear_overhang=float(vehicle_config["rear_overhang_m"]),
        front_extent=float(vehicle_config["front_extent_m"]),
        half_width=float(vehicle_config["half_width_m"]),
    )
    intent = config["intent_eligibility"]
    limits = IntentEligibilityLimits(
        minimum_native_terminal_displacement_m=float(
            intent["minimum_native_terminal_displacement_m"]
        ),
        minimum_progress_ratio=float(intent["progress_ratio"][0]),
        maximum_progress_ratio=float(intent["progress_ratio"][1]),
        maximum_terminal_cross_track_m=float(intent["maximum_terminal_cross_track_m"]),
        maximum_terminal_heading_difference_rad=float(
            intent["maximum_terminal_heading_difference_rad"]
        ),
        maximum_terminal_speed_difference_mps=float(
            intent["maximum_terminal_speed_difference_mps"]
        ),
    )
    native_poses = tuple(Pose2D(*row) for row in native_array)
    geometry_failures: Counter[str] = Counter()
    candidate_rows: list[dict[str, Any]] = []
    intent_count = 0
    geometry_count = 0
    meaningful_count = 0
    opportunity_count = 0
    native_outer = None
    native_inner_nodes = None
    native_inner_cells = None
    try:
        native_outer = certify_swept_disk_cover(
            body, native_poses, grid, pose_error=native_pose_error
        )
        native_inner_nodes = certify_native_inner_disk_nodes(
            body, native_poses, grid, pose_error=native_pose_error
        )
        native_inner_cells = certify_swept_inner_disk_cells(
            body, native_poses, grid, pose_error=native_pose_error
        )
    except ValueError as error:
        geometry_failures[f"native:{error}"] += 1

    area_j0_min = float(
        config["opportunity"]["minimum_challenger_minus_native_area_lower_m2"]
    )
    area_0j_min = float(
        config["opportunity"]["minimum_native_minus_challenger_area_lower_m2"]
    )
    improvement_min = float(
        config["opportunity"]["minimum_teacher_improvement_lower_m"]
    )
    for candidate in first.candidates:
        candidate_array = np.vstack((np.zeros((1, 3)), candidate.poses))
        eligibility = qualify_intent_match(
            candidate_array,
            native_array,
            challenger_terminal_speed_mps=candidate.terminal_speed_mps,
            native_terminal_speed_mps=native_terminal_speed,
            limits=limits,
        )
        row: dict[str, Any] = {
            "control": candidate.control.provenance,
            "intent_eligible": eligibility.eligible,
            "intent_reason": eligibility.reason,
            "progress_ratio": eligibility.progress_ratio,
            "terminal_cross_track_m": eligibility.terminal_cross_track_m,
            "terminal_heading_difference_rad": (
                eligibility.terminal_heading_difference_rad
            ),
            "terminal_speed_difference_mps": eligibility.terminal_speed_difference_mps,
        }
        if not eligibility.eligible:
            candidate_rows.append(row)
            continue
        intent_count += 1
        if native_outer is None or native_inner_nodes is None or native_inner_cells is None:
            row["geometry_valid"] = False
            row["geometry_reason"] = "native_geometry_invalid"
            candidate_rows.append(row)
            continue
        challenger_poses = tuple(Pose2D(*pose) for pose in candidate_array)
        try:
            challenger_outer = certify_swept_disk_cover(body, challenger_poses, grid)
            challenger_inner = certify_swept_inner_disk_cells(body, challenger_poses, grid)
            noninclusion = certify_disk_noninclusion_from_covers(
                challenger_outer,
                challenger_inner,
                native_outer,
                native_inner_cells,
                grid,
            )
            teacher = continuous_teacher_improvement_lower_bound(
                labels,
                errors,
                challenger_outer.node_indices,
                native_inner_nodes.node_indices,
                challenger_cover_radius_m=challenger_outer.h,
            )
        except ValueError as error:
            geometry_failures[str(error)] += 1
            row["geometry_valid"] = False
            row["geometry_reason"] = str(error)
            candidate_rows.append(row)
            continue
        geometry_count += 1
        meaningful = (
            noninclusion.challenger_minus_native_area_lower_m2 >= area_j0_min
            and noninclusion.native_minus_challenger_area_lower_m2 >= area_0j_min
        )
        opportunity = meaningful and teacher.improvement_lower_m >= improvement_min
        meaningful_count += int(meaningful)
        opportunity_count += int(opportunity)
        row.update(
            {
                "geometry_valid": True,
                "challenger_minus_native_area_lower_m2": (
                    noninclusion.challenger_minus_native_area_lower_m2
                ),
                "native_minus_challenger_area_lower_m2": (
                    noninclusion.native_minus_challenger_area_lower_m2
                ),
                "teacher_improvement_lower_m": teacher.improvement_lower_m,
                "meaningful_noninclusion": meaningful,
                "teacher_opportunity": opportunity,
            }
        )
        candidate_rows.append(row)

    ledger_complete = len(candidate_rows) == len(first.candidates)
    stage1_pass = ledger_complete and intent_count > 0 and geometry_count > 0
    if not ledger_complete:
        decision = "FAIL_INCOMPLETE_LEDGER"
        next_step = "repair_before_any_further_data_access"
    elif intent_count == 0:
        decision = "STOP_NO_INTENT_ELIGIBLE_CANDIDATE"
        next_step = "registered_expanded_fit_only_candidate_supply_diagnosis"
    elif geometry_count == 0:
        decision = "STOP_NO_CONTINUOUS_GEOMETRY_VALID_CANDIDATE"
        next_step = "registered_expanded_fit_only_geometry_diagnosis"
    else:
        decision = "PASS"
        next_step = "expanded_fit_only_preflight"
    result: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": decision,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": _sha256(args.config),
        "elapsed_seconds": time.perf_counter() - started,
        "data_scope": {
            "role": "fit",
            "split": "train",
            "recording_group": anchor["recording_group"],
            "anchor_index": int(anchor["anchor_index"]),
            "scene_token": scene_token,
            "calibration_test_target_or_development_accessed": False,
        },
        "native": {
            "identity": "cached_PDM_Closed_planner_trajectory",
            "human_trajectory_used": False,
            "sample_count": int(native_array.shape[0]),
            "canonical_sample_sha256": native_hash,
            "maximum_inverse_transform_error_m": native_pose_error,
            "terminal_displacement_m": float(np.linalg.norm(native_array[-1, :2])),
            "terminal_speed_mps": native_terminal_speed,
        },
        "initial_speed_mps": initial_speed,
        "candidate_counts": {
            "raw": len(first.candidates),
            "intent_eligible": intent_count,
            "continuous_geometry_valid": geometry_count,
            "meaningful_bidirectional_noninclusion": meaningful_count,
            "teacher_opportunity": opportunity_count,
        },
        "geometry_failures": dict(sorted(geometry_failures.items())),
        "candidate_ledger": candidate_rows,
        "claims": {
            "native_interface_closed": ledger_complete,
            "continuous_noninclusion_interface_closed": geometry_count > 0,
            "opportunity_prevalence_estimated": False,
            "r004h4_registered": False,
            "development_access_authorized": False,
            "coverage_established": False,
            "gate_b_passed": False,
        },
        "next_step": next_step,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()
    result = run(args)
    summary = {key: value for key, value in result.items() if key != "candidate_ledger"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
