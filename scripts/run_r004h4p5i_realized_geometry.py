"""Run the registered eight-block realized-geometry interface sentinel."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import pickle
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.certification.geometry import (
    Pose2D,
    RectangularGrid,
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
    terminal_segment_average_speed,
)
from aorta_drive.evaluation.navsim_execution import execute_official_navsim_batch
from aorta_drive.evaluation.navsim_geometry import (
    audit_official_corners,
    rear_axle_states_to_corners,
    vehicle_geometry_contract,
)
from diagnose_r004h4p2_candidate_supply import (
    _metric_records,
    _selected_anchors,
    _sha256,
)


def _manifest_record(
    manifest: dict[str, Any], log_name: str, scene_token: str
) -> dict[str, Any]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("label manifest lacks a records list")
    cache_id = f"{log_name}/{scene_token}"
    matches = [item for item in records if isinstance(item, dict) and item.get("cache_id") == cache_id]
    if len(matches) != 1:
        raise ValueError("label manifest does not contain exactly one selected cache")
    return matches[0]


def _load_teacher(
    path: Path, record: dict[str, Any]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if _sha256(path) != record.get("artifact_sha256"):
        raise ValueError("teacher artifact raw hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        required = {"label_m", "transform_error_m", "distance_error_m", "total_error_m"}
        if set(archive.files) != required:
            raise ValueError("teacher artifact arrays do not match the frozen contract")
        arrays = {name: np.asarray(archive[name]) for name in required}
    if any(array.dtype != np.float64 or array.shape != (64, 64) for array in arrays.values()):
        raise ValueError("teacher arrays must be float64 [64,64]")
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("teacher arrays must be finite")
    total = arrays["total_error_m"]
    if (
        np.any(total < 0.0)
        or np.any(total < arrays["transform_error_m"])
        or np.any(total < arrays["distance_error_m"])
    ):
        raise ValueError("teacher error ledger has an optimistic direction")
    if int(record.get("node_count", -1)) != 4096:
        raise ValueError("label record node count is not 4096")
    if float(record.get("max_total_error_m", math.nan)) != float(np.max(total)):
        raise ValueError("label record maximum error disagrees with artifact")
    return arrays["label_m"].reshape(-1), total.reshape(-1)


def _selected_sentinels(time_set: Path, role_manifest: Path) -> list[dict[str, str]]:
    anchors = _selected_anchors(time_set, role_manifest)
    sentinels = [anchor for anchor in anchors if int(anchor["anchor_index"]) == 0]
    if len(sentinels) != 8 or len({row["recording_group"] for row in sentinels}) != 8:
        raise ValueError("sentinel design requires anchor zero from eight unique fit groups")
    return sentinels


def _limits(config: dict[str, Any]) -> IntentEligibilityLimits:
    intent = config["intent_eligibility"]
    return IntentEligibilityLimits(
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


def _verify_sources(config: dict[str, Any], repository: Path, navsim_root: Path) -> None:
    local = {
        "navsim_geometry_source_sha256": repository
        / "src/aorta_drive/evaluation/navsim_geometry.py",
        "navsim_execution_source_sha256": repository
        / "src/aorta_drive/evaluation/navsim_execution.py",
        "geometry_source_sha256": repository / "src/aorta_drive/certification/geometry.py",
        "opportunity_source_sha256": repository
        / "src/aorta_drive/certification/opportunity.py",
        "native_offset_source_sha256": repository
        / "src/aorta_drive/candidates/native_offset.py",
    }
    for key, path in local.items():
        if _sha256(path) != config["implementation"][key]:
            raise ValueError(f"local source hash mismatch: {key}")
    for relative, expected in config["implementation"]["official_source_sha256"].items():
        if _sha256(navsim_root / relative) != expected:
            raise ValueError(f"official source hash mismatch: {relative}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for name, path in (
        ("time_set", args.time_set),
        ("role_manifest", args.role_manifest),
        ("metric_manifest", args.metric_manifest),
        ("label_manifest", args.label_manifest),
    ):
        if _sha256(path) != config["inputs"][f"{name}_sha256"]:
            raise ValueError(f"{name} hash disagrees with registration")
    repository = Path(__file__).resolve().parents[1]
    _verify_sources(config, repository, args.navsim_root)

    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (  # type: ignore[import-not-found]
        state_array_to_coords_array,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (  # type: ignore[import-not-found]
        StateIndex,
    )

    anchors = _selected_sentinels(args.time_set, args.role_manifest)
    metric_records = _metric_records(
        json.loads(args.metric_manifest.read_text(encoding="utf-8"))
    )
    label_manifest = json.loads(args.label_manifest.read_text(encoding="utf-8"))
    expected_layers = sorted(config["map_teacher"]["selected_layers"])
    actual_layers = sorted(label_manifest.get("protocol", {}).get("selected_layers", []))
    if actual_layers != expected_layers:
        raise ValueError("label manifest selected layers disagree with registration")

    grid_config = config["geometry"]["grid"]
    grid = RectangularGrid(
        x_min=float(grid_config["x_min_m"]),
        y_min=float(grid_config["y_min_m"]),
        columns=int(grid_config["columns"]),
        rows=int(grid_config["rows"]),
        resolution=float(grid_config["resolution_m"]),
    )
    amplitudes = tuple(float(value) for value in config["desired_reference"]["amplitudes_m"])
    limits = _limits(config)
    comparison_indices = np.arange(0, 41, 5, dtype=np.int64)
    minimum_j0 = float(
        config["geometry"]["meaningful_noninclusion"][
            "minimum_challenger_minus_native_area_lower_m2"
        ]
    )
    minimum_0j = float(
        config["geometry"]["meaningful_noninclusion"][
            "minimum_native_minus_challenger_area_lower_m2"
        ]
    )
    minimum_improvement = float(
        config["map_teacher"]["minimum_teacher_improvement_lower_m"]
    )

    rows: list[dict[str, Any]] = []
    interface_failures: list[str] = []
    geometry_failures: Counter[str] = Counter()
    anchors_with_supply = 0
    anchors_with_geometry = 0
    anchors_with_meaningful = 0
    anchors_with_teacher_bound = 0
    anchors_with_opportunity = 0
    for anchor in anchors:
        key = (anchor["log_name"], anchor["scene_token"])
        metric_record = metric_records.get(key)
        if metric_record is None:
            raise ValueError("metric manifest lacks a registered sentinel")
        cache_path = args.metric_root / str(metric_record["relative_cache_path"])
        actual_cache_hash = _sha256(cache_path)
        if actual_cache_hash != metric_record["cache_sha256"]:
            raise ValueError("metric cache raw hash mismatch")
        label_record = _manifest_record(label_manifest, *key)
        cache_bound = label_record.get("cache_sha256") == actual_cache_hash
        row: dict[str, Any] = {
            "recording_group": anchor["recording_group"],
            "anchor_index": int(anchor["anchor_index"]),
            "log_name": anchor["log_name"],
            "scene_token": anchor["scene_token"],
            "metric_cache_label_cache_bound": cache_bound,
        }
        if not cache_bound:
            interface_failures.append(f"{key}: metric-cache/label-cache hash mismatch")
            rows.append(row)
            continue
        with lzma.open(cache_path, "rb") as stream:
            cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM input
        artifact = args.label_root / "labels" / str(label_record["artifact_name"])
        labels, errors = _load_teacher(artifact, label_record)
        parameters = cache.ego_state.car_footprint.vehicle_parameters
        contract = vehicle_geometry_contract(parameters)
        vehicle_expected = (
            contract.width_m == float(config["vehicle"]["expected_width_m"])
            and contract.length_m == float(config["vehicle"]["expected_length_m"])
            and contract.rear_axle_to_center_m
            == float(config["vehicle"]["expected_rear_axle_to_center_m"])
            and contract.body.rear_overhang
            == float(config["vehicle"]["derived_rear_overhang_m"])
            and contract.body.front_extent
            == float(config["vehicle"]["derived_front_extent_m"])
            and contract.body.half_width == float(config["vehicle"]["derived_half_width_m"])
        )
        if not vehicle_expected:
            interface_failures.append(f"{key}: official vehicle geometry mismatch")

        native = sample_cached_native_trajectory(cache)
        desired_batch = generate_native_offset_candidates(
            native.poses,
            amplitudes_m=amplitudes,
            minimum_reference_displacement_m=1e-6,
            enforce_desired_curve_dynamic_screen=False,
        )
        desired = np.stack([candidate.poses for candidate in desired_batch.candidates])
        execution = execute_official_navsim_batch(cache, desired)
        analytic_corners = rear_axle_states_to_corners(
            execution.realized_states,
            contract=contract,
            x_index=int(StateIndex.X),
            y_index=int(StateIndex.Y),
            heading_index=int(StateIndex.HEADING),
        )
        official_coords = state_array_to_coords_array(execution.realized_states, parameters)
        corner_audit = audit_official_corners(
            analytic_corners,
            official_coords[..., :4, :],
            ulp_multiplier=int(
                config["vehicle"]["official_corner_compatibility_ulp_multiplier"]
            ),
        )
        if not corner_audit.verified:
            interface_failures.append(f"{key}: official corner convention mismatch")

        realized_half = execution.realized_local_poses[:, comparison_indices]
        realized_native_half = realized_half[0]
        realized_full = execution.realized_local_poses
        eligible_indices: list[int] = []
        candidate_rows: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(desired_batch.candidates, start=1):
            half = realized_half[candidate_index]
            eligibility = qualify_intent_match(
                half,
                realized_native_half,
                challenger_terminal_speed_mps=terminal_segment_average_speed(
                    half, interval_s=0.5
                ),
                native_terminal_speed_mps=terminal_segment_average_speed(
                    realized_native_half, interval_s=0.5
                ),
                limits=limits,
            )
            candidate_rows.append(
                {
                    "amplitude_m": candidate.control.amplitude_m,
                    "intent_eligible": eligibility.eligible,
                    "intent_reason": eligibility.reason,
                    "geometry_valid": False,
                    "meaningful_noninclusion": False,
                    "teacher_bound_computed": False,
                    "teacher_opportunity": False,
                }
            )
            if eligibility.eligible:
                eligible_indices.append(candidate_index)
        has_supply = bool(eligible_indices)
        anchors_with_supply += int(has_supply)

        native_geometry: tuple[Any, Any, Any] | None = None
        if has_supply:
            native_poses = tuple(Pose2D(*pose) for pose in realized_full[0])
            try:
                native_geometry = (
                    certify_swept_disk_cover(
                        contract.body,
                        native_poses,
                        grid,
                        pose_error=execution.maximum_transform_error_m,
                    ),
                    certify_native_inner_disk_nodes(
                        contract.body,
                        native_poses,
                        grid,
                        pose_error=execution.maximum_transform_error_m,
                    ),
                    certify_swept_inner_disk_cells(
                        contract.body,
                        native_poses,
                        grid,
                        pose_error=execution.maximum_transform_error_m,
                    ),
                )
            except ValueError as error:
                geometry_failures[f"native:{error}"] += 1

        anchor_geometry = False
        anchor_meaningful = False
        anchor_teacher = False
        anchor_opportunity = False
        if native_geometry is not None:
            native_outer, native_inner_nodes, native_inner_cells = native_geometry
            for candidate_index in eligible_indices:
                candidate_row = candidate_rows[candidate_index - 1]
                candidate_poses = tuple(Pose2D(*pose) for pose in realized_full[candidate_index])
                try:
                    challenger_outer = certify_swept_disk_cover(
                        contract.body,
                        candidate_poses,
                        grid,
                        pose_error=execution.maximum_transform_error_m,
                    )
                    challenger_inner = certify_swept_inner_disk_cells(
                        contract.body,
                        candidate_poses,
                        grid,
                        pose_error=execution.maximum_transform_error_m,
                    )
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
                    candidate_row["geometry_reason"] = str(error)
                    continue
                meaningful = (
                    noninclusion.challenger_minus_native_area_lower_m2 >= minimum_j0
                    and noninclusion.native_minus_challenger_area_lower_m2 >= minimum_0j
                )
                opportunity = meaningful and teacher.improvement_lower_m >= minimum_improvement
                candidate_row.update(
                    {
                        "geometry_valid": True,
                        "meaningful_noninclusion": meaningful,
                        "teacher_bound_computed": True,
                        "teacher_opportunity": opportunity,
                        "challenger_minus_native_area_lower_m2": (
                            noninclusion.challenger_minus_native_area_lower_m2
                        ),
                        "native_minus_challenger_area_lower_m2": (
                            noninclusion.native_minus_challenger_area_lower_m2
                        ),
                        "teacher_improvement_lower_m": teacher.improvement_lower_m,
                    }
                )
                anchor_geometry = True
                anchor_meaningful |= meaningful
                anchor_teacher = True
                anchor_opportunity |= opportunity
        anchors_with_geometry += int(anchor_geometry)
        anchors_with_meaningful += int(anchor_meaningful)
        anchors_with_teacher_bound += int(anchor_teacher)
        anchors_with_opportunity += int(anchor_opportunity)
        row.update(
            {
                "official_vehicle_parameters_exact": vehicle_expected,
                "corner_compatibility_verified": corner_audit.verified,
                "maximum_corner_coordinate_disagreement_m": (
                    corner_audit.maximum_coordinate_disagreement_m
                ),
                "corner_tolerance_m": corner_audit.tolerance_m,
                "execution_deterministic_replay_bit_exact": (
                    execution.deterministic_replay_bit_exact
                ),
                "maximum_realized_transform_error_m": execution.maximum_transform_error_m,
                "registered_candidate_count": len(amplitudes),
                "emitted_candidate_count": len(desired_batch.candidates),
                "realized_intent_eligible_count": len(eligible_indices),
                "has_realized_intent_supply": has_supply,
                "has_geometry_valid_candidate": anchor_geometry,
                "has_meaningful_noninclusion": anchor_meaningful,
                "has_teacher_bound": anchor_teacher,
                "has_teacher_opportunity": anchor_opportunity,
                "candidate_ledger": candidate_rows,
            }
        )
        rows.append(row)
        print(json.dumps({"sentinels_completed": len(rows)}), flush=True)

    ledger_complete = len(rows) == 8 and all(
        len(row.get("candidate_ledger", [])) == 6 for row in rows
    )
    implementation_passed = (
        not interface_failures
        and ledger_complete
        and all(row.get("metric_cache_label_cache_bound") is True for row in rows)
        and all(row.get("official_vehicle_parameters_exact") is True for row in rows)
        and all(row.get("corner_compatibility_verified") is True for row in rows)
        and all(
            row.get("execution_deterministic_replay_bit_exact") is True for row in rows
        )
    )
    pipeline_exercised = (
        anchors_with_supply > 0
        and anchors_with_geometry > 0
        and anchors_with_teacher_bound > 0
    )
    if not implementation_passed:
        decision = "STOP_REALIZED_GEOMETRY_IMPLEMENTATION"
    elif not pipeline_exercised:
        decision = "STOP_REALIZED_GEOMETRY_PIPELINE_UNEXERCISED"
    else:
        decision = "PASS_TO_REGISTER_128_ANCHOR_GEOMETRY_PREFLIGHT"
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
            "split": "train",
            "recording_group_count": 8,
            "anchor_count": 8,
            "sentinel_anchor_index": 0,
            "prevalence_or_significance_claim": False,
            "development_calibration_test_or_target_accessed": False,
        },
        "interface_failures": interface_failures,
        "geometry_failures": dict(sorted(geometry_failures.items())),
        "pipeline_counts": {
            "anchors_with_realized_intent_supply": anchors_with_supply,
            "anchors_with_geometry_valid_candidate": anchors_with_geometry,
            "anchors_with_meaningful_noninclusion": anchors_with_meaningful,
            "anchors_with_teacher_bound": anchors_with_teacher_bound,
            "anchors_with_teacher_opportunity": anchors_with_opportunity,
        },
        "sentinel_ledger": rows,
        "claims": {
            "realized_geometry_implementation_closed": implementation_passed,
            "full_pipeline_exercised": pipeline_exercised,
            "prevalence_estimated": False,
            "method_performance_evaluated": False,
            "development_access_authorized": False,
            "coverage_established": False,
            "gate_b_passed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--navsim-root", type=Path, required=True)
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
