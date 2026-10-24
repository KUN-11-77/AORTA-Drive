"""Qualify fork-aware set-min operators on the frozen fit teacher panel."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.operator_anatomy import (
    evaluate_spatial_fork,
    evaluate_temporal_fork,
)
from aorta_drive.evaluation.shadow_decision import (
    build_shadow_panel,
    rasterize_discrete_footprint_nodes,
)
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4a_fit_projective_observability import _write_json


FloatArray = npt.NDArray[np.float64]


def _bootstrap_mean_ci(values: FloatArray, seed: int, resamples: int) -> list[float]:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be a nonempty finite vector")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draw = rng.integers(0, values.size, size=values.size)
        estimates[index] = float(np.mean(values[draw]))
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _fixed_suffix_contrast(
    reference_by_time: tuple[npt.NDArray[np.int64], ...],
    comparison_by_time: tuple[npt.NDArray[np.int64], ...],
    labels: FloatArray,
    start_index: int,
) -> float:
    if start_index < 0 or start_index >= len(reference_by_time):
        raise ValueError("fixed suffix start index is outside the trajectory")
    reference = np.unique(np.concatenate(reference_by_time[start_index:]))
    comparison = np.unique(np.concatenate(comparison_by_time[start_index:]))
    return float(np.min(labels[comparison]) - np.min(labels[reference]))


def _summarize(
    rows: list[dict[str, Any]],
    *,
    group_names: tuple[str, ...],
    expected_common_min_pairs: int,
    minimum_valid_fraction: float,
    minimum_recovery_rate: float,
    minimum_affected_groups: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    pairs = [pair for row in rows for pair in row["pair_factors"]]
    common = [
        pair
        for pair in pairs
        if pair["parent_category"] == "common_minimum_domination"
    ]
    if len(common) != expected_common_min_pairs:
        raise ValueError("B2 common-minimum pair identity count changed")
    if any(float(pair["global_contrast_m"]) != 0.0 for pair in common):
        raise ValueError("a frozen B2 common-minimum pair is no longer exact zero")

    group_rows: list[dict[str, Any]] = []
    for group in group_names:
        group_pairs = [
            pair
            for row in rows
            if row["recording_group"] == group
            for pair in row["pair_factors"]
            if pair["parent_category"] == "common_minimum_domination"
        ]
        if not group_pairs:
            raise ValueError(f"group has no common-minimum pair: {group}")
        temporal_valid = sum(bool(pair["temporal_fork"]["valid"]) for pair in group_pairs)
        temporal_recovered = sum(
            bool(pair["temporal_fork"]["valid"])
            and bool(pair["temporal_fork"]["nonzero_branch_contrast"])
            for pair in group_pairs
        )
        spatial_valid = sum(bool(pair["spatial_fork"]["valid"]) for pair in group_pairs)
        spatial_recovered = sum(
            bool(pair["spatial_fork"]["valid"])
            and bool(pair["spatial_fork"]["nonzero_branch_contrast"])
            for pair in group_pairs
        )
        fixed_recovered = sum(
            float(pair["fixed_suffix_contrast_m"]) != 0.0 for pair in group_pairs
        )
        group_rows.append(
            {
                "recording_group": group,
                "common_minimum_pair_count": len(group_pairs),
                "temporal_valid_pair_count": temporal_valid,
                "temporal_recovered_nonzero_pair_count": temporal_recovered,
                "temporal_valid_fraction": temporal_valid / len(group_pairs),
                "temporal_recovery_rate": temporal_recovered / len(group_pairs),
                "spatial_valid_pair_count": spatial_valid,
                "spatial_recovered_nonzero_pair_count": spatial_recovered,
                "spatial_valid_fraction": spatial_valid / len(group_pairs),
                "spatial_recovery_rate": spatial_recovered / len(group_pairs),
                "fixed_suffix_recovered_nonzero_pair_count": fixed_recovered,
                "fixed_suffix_recovery_rate": fixed_recovered / len(group_pairs),
            }
        )

    temporal_rates = np.asarray(
        [float(row["temporal_recovery_rate"]) for row in group_rows],
        dtype=np.float64,
    )
    spatial_rates = np.asarray(
        [float(row["spatial_recovery_rate"]) for row in group_rows],
        dtype=np.float64,
    )
    temporal_valid_count = sum(bool(pair["temporal_fork"]["valid"]) for pair in common)
    spatial_valid_count = sum(bool(pair["spatial_fork"]["valid"]) for pair in common)
    temporal_recovered_count = sum(
        bool(pair["temporal_fork"]["valid"])
        and bool(pair["temporal_fork"]["nonzero_branch_contrast"])
        for pair in common
    )
    spatial_recovered_count = sum(
        bool(pair["spatial_fork"]["valid"])
        and bool(pair["spatial_fork"]["nonzero_branch_contrast"])
        for pair in common
    )
    temporal_ci = _bootstrap_mean_ci(
        temporal_rates, bootstrap_seed, bootstrap_resamples
    )
    spatial_ci = _bootstrap_mean_ci(
        spatial_rates, bootstrap_seed + 1, bootstrap_resamples
    )
    temporal_affected_groups = sum(
        int(row["temporal_recovered_nonzero_pair_count"]) > 0 for row in group_rows
    )
    spatial_affected_groups = sum(
        int(row["spatial_recovered_nonzero_pair_count"]) > 0 for row in group_rows
    )
    temporal_valid_fraction = temporal_valid_count / len(common)
    spatial_valid_fraction = spatial_valid_count / len(common)
    temporal_pass = (
        temporal_valid_fraction >= minimum_valid_fraction
        and temporal_ci[0] > minimum_recovery_rate
        and temporal_affected_groups >= minimum_affected_groups
    )
    spatial_pass = (
        spatial_valid_fraction >= minimum_valid_fraction
        and spatial_ci[0] > minimum_recovery_rate
        and spatial_affected_groups >= minimum_affected_groups
    )
    if temporal_pass:
        decision = (
            "TEMPORAL_FORK_OPERATOR_NONDEGENERATE_ON_FIT__DECISION_RELEVANCE_HOLD"
        )
    elif spatial_pass:
        decision = "SPATIAL_ONLY_SIGNAL_REQUIRES_THEORY_REVIEW__NO_OPERATOR_SELECTED"
    else:
        decision = "STOP_ROAD_MARGIN_AS_GENERAL_RERANKER"

    temporal_valid_pairs = [
        pair for pair in common if pair["temporal_fork"]["valid"]
    ]
    spatial_valid_pairs = [pair for pair in common if pair["spatial_fork"]["valid"]]
    return {
        "anchor_count": len(rows),
        "recording_group_count": len(group_names),
        "pair_count": len(pairs),
        "common_minimum_pair_count": len(common),
        "temporal_valid_pair_count": temporal_valid_count,
        "temporal_valid_fraction": temporal_valid_fraction,
        "temporal_recovered_nonzero_pair_count": temporal_recovered_count,
        "temporal_pair_pooled_recovery_rate": temporal_recovered_count / len(common),
        "temporal_group_equal_recovery_rate": float(np.mean(temporal_rates)),
        "temporal_group_bootstrap_ci95": temporal_ci,
        "temporal_affected_group_count": temporal_affected_groups,
        "spatial_valid_pair_count": spatial_valid_count,
        "spatial_valid_fraction": spatial_valid_fraction,
        "spatial_recovered_nonzero_pair_count": spatial_recovered_count,
        "spatial_pair_pooled_recovery_rate": spatial_recovered_count / len(common),
        "spatial_group_equal_recovery_rate": float(np.mean(spatial_rates)),
        "spatial_group_bootstrap_ci95": spatial_ci,
        "spatial_affected_group_count": spatial_affected_groups,
        "fixed_suffix_recovered_nonzero_pair_count": sum(
            float(pair["fixed_suffix_contrast_m"]) != 0.0 for pair in common
        ),
        "temporal_shared_teacher_positive_pair_count": sum(
            bool(pair["temporal_fork"]["shared_strictly_safe"])
            for pair in temporal_valid_pairs
        ),
        "temporal_teacher_road_rescue_opportunity_count": sum(
            bool(pair["temporal_fork"]["road_rescue_opportunity"])
            for pair in temporal_valid_pairs
        ),
        "temporal_teacher_road_induction_risk_count": sum(
            bool(pair["temporal_fork"]["road_induction_risk"])
            for pair in temporal_valid_pairs
        ),
        "spatial_shared_teacher_positive_pair_count": sum(
            bool(pair["spatial_fork"]["shared_strictly_safe"])
            for pair in spatial_valid_pairs
        ),
        "temporal_fork_index_histogram": {
            str(index): sum(
                pair["temporal_fork"]["fork_index"] == index
                for pair in temporal_valid_pairs
            )
            for index in sorted(
                {
                    int(pair["temporal_fork"]["fork_index"])
                    for pair in temporal_valid_pairs
                }
            )
        },
        "minimum_valid_fraction": minimum_valid_fraction,
        "minimum_group_equal_recovery_rate_strictly_greater_than": minimum_recovery_rate,
        "minimum_affected_groups": minimum_affected_groups,
        "temporal_primary_pass": temporal_pass,
        "spatial_sensitivity_pass": spatial_pass,
        "group_summaries": group_rows,
        "decision": decision,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(Mapping[str, Any], config["H7BS4B3"])
    repository = Path(__file__).resolve().parents[1]
    frozen = cast(Mapping[str, str], stage["frozen_input_sha256"])
    for name, path in {
        "h7bs4b2_result": args.b2_result,
        "h7bs1_fit_result": args.fit_result,
        "fit_forward96_teacher_manifest": args.teacher_manifest,
    }.items():
        if sha256_file(path) != frozen[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")
    implementation = cast(Mapping[str, str], stage["implementation_sha256"])
    for name, path in {
        "operator_anatomy_source": repository
        / "src/aorta_drive/evaluation/operator_anatomy.py",
        "shadow_decision_source": repository
        / "src/aorta_drive/evaluation/shadow_decision.py",
        "runner_source": Path(__file__).resolve(),
    }.items():
        if sha256_file(path) != implementation[name]:
            raise ValueError(f"implementation hash mismatch: {name}")
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)

    b2 = cast(dict[str, Any], json.loads(args.b2_result.read_text(encoding="utf-8")))
    fit = cast(dict[str, Any], json.loads(args.fit_result.read_text(encoding="utf-8")))
    teacher = cast(
        dict[str, Any], json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    )
    if b2.get("forbidden_payload_accessed") is not False:
        raise ValueError("B2 violated its forbidden-payload boundary")
    if b2.get("summary", {}).get("decision") != (
        "COMMON_MIN_DOMINANCE_ELIGIBLE_FOR_FORK_OPERATOR_DESIGN"
    ):
        raise ValueError("B2 parent decision is not the frozen fork-design branch")
    b2_rows = cast(list[dict[str, Any]], b2["anchor_ledgers"])
    fit_rows = cast(list[dict[str, Any]], fit["anchor_ledgers"])
    fit_by_key = {
        (str(row["recording_group"]), int(row["anchor_index"])): row
        for row in fit_rows
    }
    expected_keys = {
        (str(row["recording_group"]), int(row["anchor_index"])) for row in b2_rows
    }
    if len(b2_rows) != 672 or set(fit_by_key) != expected_keys:
        raise ValueError("parent anchor identities do not align")
    if teacher.get("status") != "completed" or teacher.get("anchor_count") != 672:
        raise ValueError("fit forward96 teacher is not qualified")
    teacher_by_id = {
        str(record["cache_id"]): record
        for record in cast(list[dict[str, Any]], teacher["records"])
    }
    if len(teacher_by_id) != 672:
        raise ValueError("teacher record identities are incomplete")

    roi = cast(Mapping[str, Any], stage["roi"])
    grid = RectangularGrid(
        x_min=float(roi["x_min_m"]),
        y_min=float(roi["y_min_m"]),
        columns=int(roi["columns"]),
        rows=int(roi["rows"]),
        resolution=float(roi["resolution_m"]),
    )
    vehicle = cast(Mapping[str, Any], stage["vehicle"])
    body = VehicleRectangle(
        rear_overhang=float(vehicle["rear_overhang_m"]),
        front_extent=float(vehicle["front_extent_m"]),
        half_width=float(vehicle["half_width_m"]),
    )
    fixed_suffix_start = int(stage["fixed_suffix_control"]["start_index"])

    rows: list[dict[str, Any]] = []
    for b2_row in b2_rows:
        group = str(b2_row["recording_group"])
        anchor = int(b2_row["anchor_index"])
        fit_row = fit_by_key[(group, anchor)]
        panel_path = (args.panel_run_root / str(fit_row["panel_artifact"])).resolve()
        if not panel_path.is_relative_to(args.panel_run_root.resolve()):
            raise ValueError("panel artifact escapes frozen root")
        if sha256_file(panel_path) != b2_row["panel_artifact_sha256"]:
            raise ValueError("panel artifact hash mismatch")
        with np.load(panel_path, allow_pickle=False) as payload:
            trajectories = np.asarray(payload["realized_local_poses"], dtype=np.float64)
            eligible = tuple(bool(value) for value in payload["challenger_eligible"])
        panel = build_shadow_panel(
            trajectories,
            challenger_eligible=eligible,
            identifiers=("native",)
            + tuple(f"challenger_{index}" for index in range(1, trajectories.shape[0])),
            deterministic_execution=True,
            grid=grid,
            body=body,
        )
        candidates = tuple(candidate for candidate in panel.candidates if candidate.valid)
        if not candidates or candidates[0].registered_index != 0:
            raise ValueError("valid native candidate is missing")

        cache_id = f"{b2_row['log_name']}/{b2_row['scene_token']}"
        teacher_record = teacher_by_id[cache_id]
        label_path = args.teacher_root / "labels" / str(teacher_record["artifact_name"])
        if sha256_file(label_path) != teacher_record["artifact_sha256"]:
            raise ValueError("teacher artifact hash mismatch")
        with np.load(label_path, allow_pickle=False) as payload:
            labels = np.asarray(payload["label_m"], dtype=np.float64).reshape(-1)

        parent_pairs = {
            int(pair["comparison_registered_index"]): pair
            for pair in b2_row["pair_anatomy"]
        }
        per_time: dict[int, tuple[npt.NDArray[np.int64], ...]] = {}
        for candidate in candidates:
            poses = trajectories[candidate.registered_index]
            per_time[candidate.registered_index] = tuple(
                rasterize_discrete_footprint_nodes(
                    poses[index : index + 1], grid=grid, body=body
                )
                for index in range(poses.shape[0])
            )
        native = candidates[0]
        pair_rows: list[dict[str, Any]] = []
        for challenger in candidates[1:]:
            parent = parent_pairs.get(challenger.registered_index)
            if parent is None:
                raise ValueError("B2 pair identity is missing")
            reference_by_time = per_time[native.registered_index]
            comparison_by_time = per_time[challenger.registered_index]
            spatial = evaluate_spatial_fork(
                native.node_indices, challenger.node_indices, labels
            )
            temporal = evaluate_temporal_fork(
                reference_by_time, comparison_by_time, labels
            )
            fixed_suffix = _fixed_suffix_contrast(
                reference_by_time,
                comparison_by_time,
                labels,
                fixed_suffix_start,
            )
            pair_rows.append(
                {
                    "reference_registered_index": native.registered_index,
                    "comparison_registered_index": challenger.registered_index,
                    "parent_category": str(parent["set_min"]["category"]),
                    "global_contrast_m": float(parent["set_min"]["contrast"]),
                    "spatial_fork": asdict(spatial),
                    "temporal_fork": asdict(temporal),
                    "fixed_suffix_contrast_m": fixed_suffix,
                }
            )
        if set(parent_pairs) != {
            int(pair["comparison_registered_index"]) for pair in pair_rows
        }:
            raise ValueError("B2 pair identities do not align")
        rows.append(
            {
                "recording_group": group,
                "anchor_index": anchor,
                "log_name": str(b2_row["log_name"]),
                "scene_token": str(b2_row["scene_token"]),
                "panel_artifact_sha256": b2_row["panel_artifact_sha256"],
                "teacher_artifact_sha256": teacher_record["artifact_sha256"],
                "trajectory_sample_count": int(trajectories.shape[1]),
                "pair_factors": pair_rows,
            }
        )

    expected = cast(Mapping[str, int], stage["expected_scope_counts"])
    protocol = cast(Mapping[str, Any], stage["decision_protocol"])
    group_names = tuple(sorted({str(row["recording_group"]) for row in rows}))
    summary = _summarize(
        rows,
        group_names=group_names,
        expected_common_min_pairs=int(expected["common_minimum_pair_count"]),
        minimum_valid_fraction=float(protocol["minimum_valid_fraction"]),
        minimum_recovery_rate=float(protocol["minimum_recovery_rate"]),
        minimum_affected_groups=int(protocol["minimum_affected_groups"]),
        bootstrap_seed=int(protocol["bootstrap_seed"]),
        bootstrap_resamples=int(protocol["bootstrap_resamples"]),
    )
    for key in ("anchor_count", "recording_group_count", "pair_count"):
        if int(summary[key]) != int(expected[key]):
            raise ValueError(f"scope count mismatch: {key}")
    result = {
        "experiment_id": config["experiment_id"],
        "code_commit": args.code_commit,
        "forbidden_payload_accessed": False,
        "parent_b2_decision_preserved": True,
        "primary_factorization": "temporal_first_exact_discrete_footprint_difference",
        "spatial_factorization_role": "preregistered_sensitivity_only",
        "absolute_teacher_zero_role": "oracle_context_not_quotient_identifiable",
        "anchor_ledgers": rows,
        "summary": summary,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--b2-result", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--panel-run-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False, allow_nan=False))
