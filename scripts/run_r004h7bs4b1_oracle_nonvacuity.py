"""Evaluate whether support-conditioned joint uncertainty is non-vacuous on fit."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.difference_constraints import (
    robust_set_min_lower_bound_directed,
)
from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.certification.lipschitz_oracle import (
    OracleLipschitzProblem,
    build_oracle_lipschitz_problem,
    independent_candidate_contrast_interval,
)
from aorta_drive.evaluation.shadow_decision import build_shadow_panel
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4a_fit_projective_observability import _projective_evidence, _write_json


def _interval(
    problem: OracleLipschitzProblem,
    challenger: tuple[int, ...],
    native: tuple[int, ...],
    *,
    variant: str,
) -> tuple[float, float]:
    if variant == "joint_lipschitz":
        constraints = problem.joint_constraints
    elif variant == "shared_node_box":
        constraints = problem.box_constraints
    elif variant == "candidate_independent":
        return independent_candidate_contrast_interval(
            problem.marginal_lower,
            problem.marginal_upper,
            challenger,
            native,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")
    lower = robust_set_min_lower_bound_directed(
        problem.node_values_with_reference,
        constraints,
        challenger,
        native,
    )
    upper = -robust_set_min_lower_bound_directed(
        problem.node_values_with_reference,
        constraints,
        native,
        challenger,
    )
    return lower, upper


def _interval_payload(lower: float, upper: float) -> dict[str, Any]:
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("invalid robust contrast interval")
    return {
        "lower_m": lower,
        "upper_m": upper,
        "width_m": upper - lower,
        "sign_identified": bool(lower > 0.0 or upper < 0.0),
        "positive_identified": bool(lower > 0.0),
        "negative_identified": bool(upper < 0.0),
    }


def _bootstrap_mean_ci(
    values: npt.NDArray[np.float64], seed: int, resamples: int
) -> list[float]:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be a non-empty finite vector")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, values.size, size=values.size)
        estimates[index] = float(np.mean(values[selected]))
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _summarize(
    rows: list[dict[str, Any]],
    *,
    group_names: tuple[str, ...],
    anchors_per_group: int,
    minimum_anchors_per_group: int,
    strict_improvement_lower_bound: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    expected = {
        (group, anchor)
        for group in group_names
        for anchor in range(anchors_per_group)
    }
    observed = {
        (str(row["recording_group"]), int(row["anchor_index"])) for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("anchor rows do not exhaust the frozen panel")

    pair_count = 0
    containment_failures = 0
    joint_box_dominance_failures = 0
    box_independent_dominance_failures = 0
    teacher_inclusion_failures = 0
    harmful_interventions = {
        "joint_lipschitz": 0,
        "shared_node_box": 0,
        "candidate_independent": 0,
    }
    group_rows: list[dict[str, Any]] = []
    for group in group_names:
        anchors = [row for row in rows if row["recording_group"] == group]
        pairs = [pair for row in anchors for pair in row["pair_intervals"]]
        pair_count += len(pairs)
        teacher_inclusion_failures += sum(not row["teacher_included"] for row in anchors)
        containment_failures += sum(not pair["all_variants_contain_teacher"] for pair in pairs)
        joint_box_dominance_failures += sum(
            not pair["joint_subset_of_shared_box_numerically"] for pair in pairs
        )
        box_independent_dominance_failures += sum(
            not pair["shared_box_subset_of_candidate_independent_numerically"]
            for pair in pairs
        )
        identifiable: dict[str, int] = {}
        fractions: dict[str, float] = {}
        widths: dict[str, float] = {}
        anchors_with_any: dict[str, int] = {}
        for variant in ("joint_lipschitz", "shared_node_box", "candidate_independent"):
            identifiable[variant] = sum(
                bool(pair[variant]["sign_identified"]) for pair in pairs
            )
            fractions[variant] = identifiable[variant] / len(pairs) if pairs else 0.0
            widths[variant] = float(
                np.mean([float(pair[variant]["width_m"]) for pair in pairs])
            )
            anchors_with_any[variant] = sum(
                any(bool(pair[variant]["sign_identified"]) for pair in row["pair_intervals"])
                for row in anchors
            )
            harmful_interventions[variant] += sum(
                bool(row["selection"][variant]["harmful_intervention"])
                for row in anchors
            )
        group_rows.append(
            {
                "recording_group": group,
                "pair_count": len(pairs),
                "identifiable_pair_count": identifiable,
                "identifiable_pair_fraction": fractions,
                "mean_interval_width_m": widths,
                "anchors_with_any_identifiable_pair": anchors_with_any,
                "mean_teacher_regret_m": {
                    variant: float(
                        np.mean(
                            [
                                float(row["selection"][variant]["teacher_regret_m"])
                                for row in anchors
                            ]
                        )
                    )
                    for variant in (
                        "joint_lipschitz",
                        "shared_node_box",
                        "candidate_independent",
                    )
                },
                "intervention_count": {
                    variant: sum(
                        bool(row["selection"][variant]["intervened"])
                        for row in anchors
                    )
                    for variant in (
                        "joint_lipschitz",
                        "shared_node_box",
                        "candidate_independent",
                    )
                },
                "joint_minus_shared_box_identifiable_fraction": (
                    fractions["joint_lipschitz"] - fractions["shared_node_box"]
                ),
            }
        )

    differences = np.asarray(
        [row["joint_minus_shared_box_identifiable_fraction"] for row in group_rows],
        dtype=np.float64,
    )
    difference_ci = _bootstrap_mean_ci(differences, bootstrap_seed, bootstrap_resamples)
    passing_groups = sum(
        int(row["anchors_with_any_identifiable_pair"]["joint_lipschitz"])
        >= minimum_anchors_per_group
        for row in group_rows
    )
    all_teacher_included = teacher_inclusion_failures == 0
    arithmetic_valid = (
        containment_failures == 0
        and joint_box_dominance_failures == 0
        and box_independent_dominance_failures == 0
    )
    nonvacuous = passing_groups == len(group_names)
    graph_strictly_useful = difference_ci[0] > strict_improvement_lower_bound
    no_harm = all(value == 0 for value in harmful_interventions.values())
    if not all_teacher_included or not arithmetic_valid or not no_harm:
        decision = "INVALID_ORACLE_SET_OR_NUMERICS"
    elif not nonvacuous:
        decision = "ORACLE_SUPPORT_SET_NEAR_VACUOUS_REDESIGN_INTERFACE"
    elif not graph_strictly_useful:
        decision = "SHARED_NODE_BOX_SUFFICIENT_GRAPH_CORRELATION_NOT_JUSTIFIED"
    else:
        decision = "JOINT_LIPSCHITZ_ORACLE_NONVACUITY_SUPPORTED"
    return {
        "anchor_count": len(rows),
        "recording_group_count": len(group_names),
        "pair_count": pair_count,
        "teacher_inclusion_failure_count": teacher_inclusion_failures,
        "contrast_containment_failure_count": containment_failures,
        "joint_box_dominance_failure_count": joint_box_dominance_failures,
        "box_independent_dominance_failure_count": box_independent_dominance_failures,
        "harmful_intervention_count": harmful_interventions,
        "groups_meeting_joint_minimum_identifiable_anchors": passing_groups,
        "minimum_identifiable_anchors_per_group": minimum_anchors_per_group,
        "mean_joint_minus_shared_box_identifiable_fraction": float(np.mean(differences)),
        "joint_minus_shared_box_group_bootstrap_ci95": difference_ci,
        "group_summaries": group_rows,
        "decision": decision,
        "strict_improvement_lower_bound": strict_improvement_lower_bound,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(Mapping[str, Any], config["H7BS4B1"])
    repository = Path(__file__).resolve().parents[1]
    frozen_inputs = cast(Mapping[str, str], stage["frozen_input_sha256"])
    for name, path in {
        "h7bs4a_result": args.h7bs4a_result,
        "h7bs4b0b_result": args.h7bs4b0b_result,
        "h7bs1_fit_result": args.fit_result,
        "fit_forward96_teacher_manifest": args.teacher_manifest,
    }.items():
        if sha256_file(path) != frozen_inputs[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")
    implementation = cast(Mapping[str, str], stage["implementation_sha256"])
    for name, path in {
        "difference_constraint_source": repository
        / "src/aorta_drive/certification/difference_constraints.py",
        "lipschitz_oracle_source": repository
        / "src/aorta_drive/certification/lipschitz_oracle.py",
        "shadow_decision_source": repository
        / "src/aorta_drive/evaluation/shadow_decision.py",
        "projection_runner_source": repository
        / "scripts/run_r004h7bs4a_fit_projective_observability.py",
        "runner_source": Path(__file__).resolve(),
    }.items():
        if sha256_file(path) != implementation[name]:
            raise ValueError(f"implementation hash mismatch: {name}")
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)

    source = cast(dict[str, Any], json.loads(args.h7bs4a_result.read_text(encoding="utf-8")))
    predecessor = cast(
        dict[str, Any], json.loads(args.h7bs4b0b_result.read_text(encoding="utf-8"))
    )
    if source.get("forbidden_payload_accessed") is not False:
        raise ValueError("H7BS4A source violated its forbidden-payload boundary")
    if (
        predecessor.get("summary", {}).get("scientific_decision")
        != "TEACHER_CRITICAL_PROJECTIVE_BLINDNESS_PRESENT"
    ):
        raise ValueError("B0B predecessor did not authorize B1")
    source_rows = cast(list[dict[str, Any]], source["anchor_ledgers"])
    fit = cast(dict[str, Any], json.loads(args.fit_result.read_text(encoding="utf-8")))
    fit_rows = cast(list[dict[str, Any]], fit["anchor_ledgers"])
    fit_by_key = {
        (str(row["recording_group"]), int(row["anchor_index"])): row for row in fit_rows
    }
    if len(source_rows) != 672 or set(fit_by_key) != {
        (str(row["recording_group"]), int(row["anchor_index"])) for row in source_rows
    }:
        raise ValueError("H7BS4A and fit panel identities do not align")
    teacher = cast(
        dict[str, Any], json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    )
    teacher_by_id = {
        str(record["cache_id"]): record
        for record in cast(list[dict[str, Any]], teacher["records"])
    }
    if teacher.get("status") != "completed" or len(teacher_by_id) != 672:
        raise ValueError("fit forward96 teacher is not qualified")

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
    projection = cast(Mapping[str, Any], stage["projection"])
    raw_shape = tuple(int(value) for value in projection["raw_shape"])
    resized_shape = tuple(int(value) for value in projection["resized_shape"])
    if len(raw_shape) != 2 or len(resized_shape) != 2:
        raise ValueError("projection shapes must be two-dimensional")

    source_log_hashes = cast(dict[str, str], source["source_log_sha256"])
    log_cache: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        group = str(source_row["recording_group"])
        anchor = int(source_row["anchor_index"])
        log_name = str(source_row["log_name"])
        scene_token = str(source_row["scene_token"])
        if log_name not in log_cache:
            log_path = args.log_root / f"{log_name}.pkl"
            if sha256_file(log_path) != source_log_hashes[log_name]:
                raise ValueError("source log hash mismatch")
            with log_path.open("rb") as stream:
                log_cache[log_name] = pickle.load(stream)  # noqa: S301
        frames = log_cache[log_name]
        token_to_index = {str(frame["token"]): index for index, frame in enumerate(frames)}
        current_index = token_to_index[scene_token]
        has_evidence, _ = _projective_evidence(
            grid=grid,
            frames=frames,
            current_index=current_index,
            resized_shape=(resized_shape[0], resized_shape[1]),
            raw_shape=(raw_shape[0], raw_shape[1]),
            patch_size=int(projection["patch_size"]),
            maximum_normalized_radius=float(projection["maximum_normalized_radius"]),
        )

        fit_row = fit_by_key[(group, anchor)]
        panel_path = (args.panel_run_root / str(fit_row["panel_artifact"])).resolve()
        if not panel_path.is_relative_to(args.panel_run_root.resolve()):
            raise ValueError("panel artifact escapes frozen root")
        if sha256_file(panel_path) != source_row["panel_artifact_sha256"]:
            raise ValueError("panel artifact hash mismatch")
        with np.load(panel_path, allow_pickle=False) as panel_payload:
            trajectories = np.asarray(panel_payload["realized_local_poses"], dtype=np.float64)
            eligible = tuple(bool(value) for value in panel_payload["challenger_eligible"])
        panel = build_shadow_panel(
            trajectories,
            challenger_eligible=eligible,
            identifiers=("native",) + tuple(f"challenger_{index}" for index in range(1, 7)),
            deterministic_execution=True,
            grid=grid,
            body=body,
        )
        candidates = tuple(candidate for candidate in panel.candidates if candidate.valid)
        if not candidates or candidates[0].registered_index != 0:
            raise ValueError("valid native candidate is missing")

        cache_id = f"{log_name}/{scene_token}"
        teacher_record = teacher_by_id[cache_id]
        label_path = args.teacher_root / "labels" / str(teacher_record["artifact_name"])
        if sha256_file(label_path) != teacher_record["artifact_sha256"]:
            raise ValueError("teacher artifact hash mismatch")
        with np.load(label_path, allow_pickle=False) as label_payload:
            labels = np.asarray(label_payload["label_m"], dtype=np.float64).reshape(-1)
            errors = np.asarray(label_payload["total_error_m"], dtype=np.float64).reshape(-1)
        problem = build_oracle_lipschitz_problem(
            labels,
            errors,
            has_evidence,
            rows=grid.rows,
            columns=grid.columns,
            resolution=grid.resolution,
        )

        native_nodes = tuple(int(value) for value in candidates[0].node_indices)
        pair_rows: list[dict[str, Any]] = []
        for challenger in candidates[1:]:
            challenger_nodes = tuple(int(value) for value in challenger.node_indices)
            teacher_contrast = float(
                np.min(labels[np.asarray(challenger_nodes, dtype=np.int64)])
                - np.min(labels[np.asarray(native_nodes, dtype=np.int64)])
            )
            intervals = {
                variant: _interval_payload(
                    *_interval(
                        problem,
                        challenger_nodes,
                        native_nodes,
                        variant=variant,
                    )
                )
                for variant in (
                    "joint_lipschitz",
                    "shared_node_box",
                    "candidate_independent",
                )
            }
            tolerance = float(stage["numerics"]["comparison_tolerance_m"])
            all_contain = all(
                float(payload["lower_m"]) <= teacher_contrast + tolerance
                and float(payload["upper_m"]) >= teacher_contrast - tolerance
                for payload in intervals.values()
            )
            joint = intervals["joint_lipschitz"]
            box = intervals["shared_node_box"]
            independent = intervals["candidate_independent"]
            pair_rows.append(
                {
                    "reference_registered_index": 0,
                    "comparison_registered_index": challenger.registered_index,
                    "teacher_contrast_m": teacher_contrast,
                    **intervals,
                    "all_variants_contain_teacher": all_contain,
                    "joint_subset_of_shared_box_numerically": bool(
                        float(joint["lower_m"]) + tolerance >= float(box["lower_m"])
                        and float(joint["upper_m"]) <= float(box["upper_m"]) + tolerance
                    ),
                    "shared_box_subset_of_candidate_independent_numerically": bool(
                        float(box["lower_m"]) + tolerance
                        >= float(independent["lower_m"])
                        and float(box["upper_m"])
                        <= float(independent["upper_m"]) + tolerance
                    ),
                }
            )
        teacher_best_margin = max(
            [0.0, *[float(pair["teacher_contrast_m"]) for pair in pair_rows]]
        )
        selection: dict[str, Any] = {}
        for variant in ("joint_lipschitz", "shared_node_box", "candidate_independent"):
            positive = [
                pair for pair in pair_rows if float(pair[variant]["lower_m"]) > 0.0
            ]
            if positive:
                chosen = max(
                    positive,
                    key=lambda pair: (
                        float(pair[variant]["lower_m"]),
                        -int(pair["comparison_registered_index"]),
                    ),
                )
                selected_index = int(chosen["comparison_registered_index"])
                selected_teacher_margin = float(chosen["teacher_contrast_m"])
            else:
                selected_index = 0
                selected_teacher_margin = 0.0
            selection[variant] = {
                "selected_registered_index": selected_index,
                "intervened": selected_index != 0,
                "selected_teacher_margin_m": selected_teacher_margin,
                "teacher_regret_m": teacher_best_margin - selected_teacher_margin,
                "harmful_intervention": bool(
                    selected_index != 0 and selected_teacher_margin < 0.0
                ),
            }
        rows.append(
            {
                "recording_group": group,
                "anchor_index": anchor,
                "log_name": log_name,
                "scene_token": scene_token,
                "teacher_artifact_sha256": teacher_record["artifact_sha256"],
                "observed_node_count": problem.observed_node_count,
                "maximum_numerical_widening": problem.maximum_numerical_widening,
                "teacher_included": problem.teacher_included,
                "pair_intervals": pair_rows,
                "teacher_best_margin_m": teacher_best_margin,
                "selection": selection,
            }
        )

    groups = tuple(sorted({str(row["recording_group"]) for row in source_rows}))
    protocol = cast(Mapping[str, Any], stage["decision_protocol"])
    summary = _summarize(
        rows,
        group_names=groups,
        anchors_per_group=int(protocol["anchors_per_group"]),
        minimum_anchors_per_group=int(protocol["minimum_identifiable_anchors_per_group"]),
        strict_improvement_lower_bound=float(
            protocol["strict_group_bootstrap_improvement_lower_bound"]
        ),
        bootstrap_seed=int(protocol["bootstrap_seed"]),
        bootstrap_resamples=int(protocol["bootstrap_resamples"]),
    )
    expected = cast(Mapping[str, int], stage["expected_scope_counts"])
    for key in ("anchor_count", "recording_group_count", "pair_count"):
        if int(summary[key]) != int(expected[key]):
            raise ValueError(f"frozen scope count mismatch: {key}")
    elapsed = time.perf_counter() - started
    if elapsed > float(stage["resources"]["maximum_wall_minutes"]) * 60.0:
        raise RuntimeError("H7BS4B1 exceeded frozen wall-time cap")
    result = {
        "experiment_id": config["experiment_id"],
        "code_commit": args.code_commit,
        "forbidden_payload_accessed": False,
        "teacher_oracle_not_deployable": True,
        "parent_h7bs4a_decision_preserved": True,
        "anchor_ledgers": rows,
        "summary": summary,
        "runtime_seconds": elapsed,
    }
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--h7bs4a-result", type=Path, required=True)
    parser.add_argument("--h7bs4b0b-result", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--panel-run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False, allow_nan=False))
