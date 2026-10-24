"""Run the bounded eight-sentinel orientation-aware geometry qualification."""

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
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.certification.geometry import (
    Pose2D,
    RectangularGrid,
    certify_disk_noninclusion_from_covers,
    certify_swept_disk_cover,
    certify_swept_inner_disk_cells,
)
from aorta_drive.certification.opportunity import (
    continuous_teacher_improvement_lower_bound,
)
from aorta_drive.certification.oriented_geometry import (
    certify_oriented_bidirectional_noninclusion,
    certify_oriented_inner_cells,
    certify_oriented_swept_outer_cover,
)
from aorta_drive.evaluation.h4 import (
    qualify_intent_match,
    sample_cached_native_trajectory,
    terminal_segment_average_speed,
)
from aorta_drive.evaluation.navsim_execution import execute_official_navsim_batch
from aorta_drive.evaluation.navsim_geometry import vehicle_geometry_contract
from aorta_drive.evaluation.polygon_reference import polygon_sweep_reference
from diagnose_r004h4p2_candidate_supply import _metric_records, _sha256
from run_r004h4p5i_realized_geometry import (
    _limits,
    _load_teacher,
    _manifest_record,
    _selected_sentinels,
    _verify_sources,
)


def _phase_grid(
    resolution: float, phase_x: float, phase_y: float
) -> RectangularGrid:
    base_cells = round(64.0 / resolution)
    if not math.isclose(base_cells * resolution, 64.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("geometry resolution must exactly partition the 64 m support")
    return RectangularGrid(
        x_min=-32.0 - phase_x * resolution,
        y_min=-32.0 - phase_y * resolution,
        columns=base_cells + int(phase_x > 0.0),
        rows=base_cells + int(phase_y > 0.0),
        resolution=resolution,
    )


def _candidate_geometry(
    *,
    body: Any,
    challenger: tuple[Pose2D, ...],
    native: tuple[Pose2D, ...],
    pose_error: float,
    terminal_config: dict[str, Any],
    labels: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
) -> dict[str, Any]:
    support = RectangularGrid(-32.0, -32.0, 64, 64, 1.0)
    levels = [
        polygon_sweep_reference(body, challenger, native, int(subdivisions))
        for subdivisions in terminal_config["polygon_reference"][
            "temporal_subdivisions_per_official_interval"
        ]
    ]
    polygon_ledger = [
        {
            "subdivisions": level.subdivisions_per_interval,
            "challenger_minus_native_area_m2": (
                level.challenger_minus_native_area_m2
            ),
            "native_minus_challenger_area_m2": (
                level.native_minus_challenger_area_m2
            ),
        }
        for level in levels
    ]
    if len(levels) >= 2:
        polygon_change = {
            "challenger_minus_native_m2": abs(
                levels[-1].challenger_minus_native_area_m2
                - levels[-2].challenger_minus_native_area_m2
            ),
            "native_minus_challenger_m2": abs(
                levels[-1].native_minus_challenger_area_m2
                - levels[-2].native_minus_challenger_area_m2
            ),
        }
    else:
        polygon_change = {
            "challenger_minus_native_m2": math.nan,
            "native_minus_challenger_m2": math.nan,
        }

    disk_challenger_outer = certify_swept_disk_cover(
        body, challenger, support, pose_error=pose_error
    )
    disk_native_outer = certify_swept_disk_cover(
        body, native, support, pose_error=pose_error
    )
    disk_challenger_inner = certify_swept_inner_disk_cells(
        body, challenger, support, pose_error=pose_error
    )
    disk_native_inner = certify_swept_inner_disk_cells(
        body, native, support, pose_error=pose_error
    )
    disk = certify_disk_noninclusion_from_covers(
        disk_challenger_outer,
        disk_challenger_inner,
        disk_native_outer,
        disk_native_inner,
        support,
    )

    phases = terminal_config["formal_certificates"]["diagnostic_grid_phase_fractions"]
    resolutions = terminal_config["formal_certificates"]["geometry_resolutions_m"]
    oriented: list[dict[str, Any]] = []
    for resolution_raw in resolutions:
        resolution = float(resolution_raw)
        for phase_raw in phases:
            phase_x, phase_y = float(phase_raw[0]), float(phase_raw[1])
            grid = _phase_grid(resolution, phase_x, phase_y)
            started = time.perf_counter()
            certificate = certify_oriented_bidirectional_noninclusion(
                body,
                challenger,
                native,
                grid,
                pose_error=pose_error,
                certification_support=support,
            )
            oriented.append(
                {
                    "resolution_m": resolution,
                    "phase_fraction": [phase_x, phase_y],
                    "challenger_minus_native_area_lower_m2": (
                        certificate.challenger_minus_native_area_lower_m2
                    ),
                    "native_minus_challenger_area_lower_m2": (
                        certificate.native_minus_challenger_area_lower_m2
                    ),
                    "challenger_witness_count": len(
                        certificate.challenger_witness_cells
                    ),
                    "native_witness_count": len(certificate.native_witness_cells),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

    challenger_outer = certify_oriented_swept_outer_cover(
        body,
        challenger,
        support,
        pose_error=pose_error,
        certification_support=support,
    )
    native_inner = certify_oriented_inner_cells(
        body, native, support, pose_error=pose_error
    )
    if not native_inner.cell_indices:
        raise ValueError("oriented native inner-node set is empty")
    teacher = continuous_teacher_improvement_lower_bound(
        labels,
        errors,
        challenger_outer.node_indices,
        native_inner.cell_indices,
        challenger_cover_radius_m=challenger_outer.h,
    )
    canonical = [
        item
        for item in oriented
        if item["phase_fraction"] == [0.0, 0.0]
    ]
    for first, second in zip(canonical, canonical[1:], strict=False):
        if (
            float(second["challenger_minus_native_area_lower_m2"])
            + 1e-12
            < float(first["challenger_minus_native_area_lower_m2"])
            or float(second["native_minus_challenger_area_lower_m2"])
            + 1e-12
            < float(first["native_minus_challenger_area_lower_m2"])
        ):
            raise RuntimeError("canonical nested refinement decreased a formal lower bound")
    area_threshold = float(
        terminal_config["frozen_scope"]["meaningful_area_threshold_each_direction_m2"]
    )
    nontrivial = any(
        float(item["resolution_m"]) in {0.25, 0.125}
        and float(item["challenger_minus_native_area_lower_m2"]) >= area_threshold
        and float(item["native_minus_challenger_area_lower_m2"]) >= area_threshold
        for item in canonical
    )
    teacher_threshold = float(
        terminal_config["frozen_scope"]["teacher_improvement_threshold_m"]
    )
    return {
        "polygon_reference": polygon_ledger,
        "polygon_last_level_change_m2": polygon_change,
        "disk_control": {
            "challenger_minus_native_area_lower_m2": (
                disk.challenger_minus_native_area_lower_m2
            ),
            "native_minus_challenger_area_lower_m2": (
                disk.native_minus_challenger_area_lower_m2
            ),
        },
        "oriented_certificates": oriented,
        "canonical_nested_refinement_nondecreasing": True,
        "formal_nontrivial_bidirectional": nontrivial,
        "teacher_improvement_lower_m": teacher.improvement_lower_m,
        "teacher_opportunity": (
            nontrivial and teacher.improvement_lower_m >= teacher_threshold
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    predecessor_config: dict[str, Any] = yaml.safe_load(
        args.predecessor_config.read_text(encoding="utf-8")
    )
    if _sha256(args.predecessor_config) != config["predecessors"]["interface"][
        "config_sha256"
    ]:
        raise ValueError("predecessor config hash mismatch")
    if _sha256(args.predecessor_result) != config["predecessors"]["interface"][
        "raw_result_sha256"
    ]:
        raise ValueError("predecessor result hash mismatch")
    for name, path in (
        ("time_set", args.time_set),
        ("role_manifest", args.role_manifest),
        ("metric_manifest", args.metric_manifest),
        ("label_manifest", args.label_manifest),
    ):
        if _sha256(path) != config["frozen_scope"][f"{name}_sha256"]:
            raise ValueError(f"{name} hash disagrees with terminal registration")
    repository = Path(__file__).resolve().parents[1]
    _verify_sources(predecessor_config, repository, args.navsim_root)

    anchors = _selected_sentinels(args.time_set, args.role_manifest)
    metric_records = _metric_records(
        json.loads(args.metric_manifest.read_text(encoding="utf-8"))
    )
    label_manifest = json.loads(args.label_manifest.read_text(encoding="utf-8"))
    amplitudes = tuple(
        float(value) for value in config["frozen_scope"]["amplitudes_m"]
    )
    limits = _limits(predecessor_config)
    comparison_indices = np.arange(0, 41, 5, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    geometry_failure_count = 0
    for anchor in anchors:
        key = (anchor["log_name"], anchor["scene_token"])
        metric_record = metric_records[key]
        cache_path = args.metric_root / str(metric_record["relative_cache_path"])
        if _sha256(cache_path) != metric_record["cache_sha256"]:
            raise ValueError("metric cache raw hash mismatch")
        label_record = _manifest_record(label_manifest, *key)
        if label_record.get("cache_sha256") != metric_record["cache_sha256"]:
            raise ValueError("metric/label cache binding mismatch")
        with lzma.open(cache_path, "rb") as stream:
            cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM input
        labels, errors = _load_teacher(
            args.label_root / "labels" / str(label_record["artifact_name"]),
            label_record,
        )
        body = vehicle_geometry_contract(
            cache.ego_state.car_footprint.vehicle_parameters
        ).body
        native = sample_cached_native_trajectory(cache)
        candidate_batch = generate_native_offset_candidates(
            native.poses,
            amplitudes_m=amplitudes,
            minimum_reference_displacement_m=1e-6,
            enforce_desired_curve_dynamic_screen=False,
        )
        desired = np.stack([candidate.poses for candidate in candidate_batch.candidates])
        execution = execute_official_navsim_batch(cache, desired)
        half = execution.realized_local_poses[:, comparison_indices]
        native_half = half[0]
        native_poses = tuple(
            Pose2D(*pose) for pose in execution.realized_local_poses[0]
        )
        candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidate_batch.candidates, start=1):
            candidate_half = half[index]
            eligibility = qualify_intent_match(
                candidate_half,
                native_half,
                challenger_terminal_speed_mps=terminal_segment_average_speed(
                    candidate_half, interval_s=0.5
                ),
                native_terminal_speed_mps=terminal_segment_average_speed(
                    native_half, interval_s=0.5
                ),
                limits=limits,
            )
            candidate_row: dict[str, Any] = {
                "amplitude_m": candidate.control.amplitude_m,
                "intent_eligible": eligibility.eligible,
                "intent_reason": eligibility.reason,
            }
            if eligibility.eligible:
                challenger_poses = tuple(
                    Pose2D(*pose) for pose in execution.realized_local_poses[index]
                )
                try:
                    candidate_row.update(
                        _candidate_geometry(
                            body=body,
                            challenger=challenger_poses,
                            native=native_poses,
                            pose_error=execution.maximum_transform_error_m,
                            terminal_config=config,
                            labels=labels,
                            errors=errors,
                        )
                    )
                    candidate_row["geometry_valid"] = True
                except ValueError as error:
                    geometry_failure_count += 1
                    candidate_row.update(
                        {"geometry_valid": False, "geometry_reason": str(error)}
                    )
            candidates.append(candidate_row)
        rows.append(
            {
                "recording_group": anchor["recording_group"],
                "scene_token": anchor["scene_token"],
                "maximum_transform_error_m": execution.maximum_transform_error_m,
                "candidates": candidates,
            }
        )
        print(json.dumps({"sentinels_completed": len(rows)}), flush=True)

    valid = [
        candidate
        for row in rows
        for candidate in row["candidates"]
        if candidate.get("geometry_valid") is True
    ]
    nontrivial = [
        candidate
        for candidate in valid
        if candidate.get("formal_nontrivial_bidirectional") is True
    ]
    teacher_positive = [
        candidate
        for candidate in nontrivial
        if candidate.get("teacher_opportunity") is True
    ]
    if not valid:
        decision = "STOP_GEOMETRY_PIPELINE_INVALID"
    elif not nontrivial:
        decision = "STOP_GRID_CERTIFICATE_VACUOUS_EVALUATE_POLYGON_CERTIFICATE"
    elif not teacher_positive:
        decision = "PASS_GEOMETRY_STOP_AND_REVIEW_PROVIDER_OR_TASK_MARGIN"
    else:
        decision = "PASS_GEOMETRY_AND_TEACHER_TO_128_FIT_REGISTRATION"
    elapsed = time.perf_counter() - started
    if elapsed > float(config["resources"]["maximum_wall_minutes"]) * 60.0:
        raise RuntimeError("terminal geometry run exceeded its registered wall cap")
    return {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": decision,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": _sha256(args.config),
        "script_sha256": _sha256(Path(__file__)),
        "oriented_geometry_source_sha256": _sha256(
            Path(__file__).resolve().parents[1]
            / "src/aorta_drive/certification/oriented_geometry.py"
        ),
        "polygon_reference_source_sha256": _sha256(
            Path(__file__).resolve().parents[1]
            / "src/aorta_drive/evaluation/polygon_reference.py"
        ),
        "elapsed_seconds": elapsed,
        "sentinel_count": len(rows),
        "geometry_valid_candidate_count": len(valid),
        "formal_nontrivial_candidate_count": len(nontrivial),
        "teacher_opportunity_candidate_count": len(teacher_positive),
        "geometry_failure_count": geometry_failure_count,
        "development_calibration_test_or_target_accessed": False,
        "sentinel_ledger": rows,
        "claim_boundary": (
            "Eight fit-sentinel geometry qualification only. Polygon values are diagnostic, "
            "not exact truth. No prevalence, coverage, OOD, driving, safety, or Gate B claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predecessor-config", type=Path, required=True)
    parser.add_argument("--predecessor-result", type=Path, required=True)
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

