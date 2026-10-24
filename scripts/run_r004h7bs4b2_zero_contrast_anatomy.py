"""Attribute exact-zero teacher contrasts before redesigning the task operator."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.operator_anatomy import (
    analyze_set_min_pair,
    analyze_trajectory_pair,
)
from aorta_drive.evaluation.shadow_decision import build_shadow_panel
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4a_fit_projective_observability import _write_json


ZERO_CATEGORIES = (
    "discrete_support_duplicate",
    "common_minimum_domination",
    "distinct_minimum_teacher_tie",
)


def _summarize(
    rows: list[dict[str, Any]],
    *,
    dominance_fraction: float,
    minimum_affected_groups: int,
) -> dict[str, Any]:
    pairs = [pair for row in rows for pair in row["pair_anatomy"]]
    zero_pairs = [pair for pair in pairs if float(pair["set_min"]["contrast"]) == 0.0]
    category_counts = {
        category: sum(pair["set_min"]["category"] == category for pair in zero_pairs)
        for category in ZERO_CATEGORIES
    }
    if sum(category_counts.values()) != len(zero_pairs):
        raise ValueError("exact-zero category partition is not exhaustive")
    groups_by_category = {
        category: sorted(
            {
                str(row["recording_group"])
                for row in rows
                if any(
                    pair["set_min"]["category"] == category
                    for pair in row["pair_anatomy"]
                )
            }
        )
        for category in ZERO_CATEGORIES
    }
    category_fractions = {
        category: count / len(zero_pairs) if zero_pairs else 0.0
        for category, count in category_counts.items()
    }
    qualifying = [
        category
        for category in ZERO_CATEGORIES
        if category_fractions[category] > dominance_fraction
        and len(groups_by_category[category]) >= minimum_affected_groups
    ]
    if len(qualifying) > 1:
        raise ValueError("mutually exclusive categories cannot both dominate")
    decision_by_category = {
        "discrete_support_duplicate": (
            "DISCRETE_QUERY_ALIASING_ELIGIBLE_FOR_CONTINUOUS_QUERY_STUDY"
        ),
        "common_minimum_domination": (
            "COMMON_MIN_DOMINANCE_ELIGIBLE_FOR_FORK_OPERATOR_DESIGN"
        ),
        "distinct_minimum_teacher_tie": (
            "DISTINCT_MIN_TIES_REQUIRE_TEACHER_TASK_AUDIT"
        ),
    }
    decision = (
        decision_by_category[qualifying[0]]
        if qualifying
        else "MIXED_ZERO_CONTRAST_CAUSES_NO_SINGLE_REDESIGN_AUTHORIZED"
    )
    duplicate_pairs = [
        pair
        for pair in zero_pairs
        if pair["set_min"]["category"] == "discrete_support_duplicate"
    ]
    return {
        "anchor_count": len(rows),
        "recording_group_count": len({row["recording_group"] for row in rows}),
        "pair_count": len(pairs),
        "exact_zero_contrast_pair_count": len(zero_pairs),
        "absolute_contrast_below_0_01_m_pair_count": sum(
            abs(float(pair["set_min"]["contrast"])) < 0.01 for pair in pairs
        ),
        "absolute_contrast_below_0_10_m_pair_count": sum(
            abs(float(pair["set_min"]["contrast"])) < 0.10 for pair in pairs
        ),
        "exact_zero_category_counts": category_counts,
        "exact_zero_category_fractions": category_fractions,
        "affected_group_count_by_category": {
            category: len(groups) for category, groups in groups_by_category.items()
        },
        "affected_groups_by_category": groups_by_category,
        "duplicate_support_but_continuous_poses_differ_count": sum(
            not bool(pair["trajectory"]["poses_exactly_equal"])
            for pair in duplicate_pairs
        ),
        "duplicate_support_with_max_center_separation_gt_half_grid_count": sum(
            float(pair["trajectory"]["maximum_center_separation_m"]) > 0.5
            for pair in duplicate_pairs
        ),
        "dominance_fraction_strictly_greater_than": dominance_fraction,
        "minimum_affected_groups": minimum_affected_groups,
        "decision": decision,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(Mapping[str, Any], config["H7BS4B2"])
    repository = Path(__file__).resolve().parents[1]
    frozen = cast(Mapping[str, str], stage["frozen_input_sha256"])
    for name, path in {
        "h7bs4a_result": args.h7bs4a_result,
        "h7bs4b1_result": args.b1_result,
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

    source = cast(dict[str, Any], json.loads(args.h7bs4a_result.read_text(encoding="utf-8")))
    b1 = cast(dict[str, Any], json.loads(args.b1_result.read_text(encoding="utf-8")))
    fit = cast(dict[str, Any], json.loads(args.fit_result.read_text(encoding="utf-8")))
    teacher = cast(
        dict[str, Any], json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    )
    if source.get("forbidden_payload_accessed") is not False:
        raise ValueError("H7BS4A source violated its forbidden-payload boundary")
    if b1.get("summary", {}).get("decision") != (
        "ORACLE_SUPPORT_SET_NEAR_VACUOUS_REDESIGN_INTERFACE"
    ):
        raise ValueError("B1 parent decision is not the frozen redesign branch")
    source_rows = cast(list[dict[str, Any]], source["anchor_ledgers"])
    b1_rows = cast(list[dict[str, Any]], b1["anchor_ledgers"])
    b1_by_key = {
        (str(row["recording_group"]), int(row["anchor_index"])): row for row in b1_rows
    }
    fit_rows = cast(list[dict[str, Any]], fit["anchor_ledgers"])
    fit_by_key = {
        (str(row["recording_group"]), int(row["anchor_index"])): row for row in fit_rows
    }
    expected_keys = {
        (str(row["recording_group"]), int(row["anchor_index"])) for row in source_rows
    }
    if len(source_rows) != 672 or set(fit_by_key) != expected_keys or set(b1_by_key) != expected_keys:
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

    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        group = str(source_row["recording_group"])
        anchor = int(source_row["anchor_index"])
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
            identifiers=("native",) + tuple(
                f"challenger_{index}" for index in range(1, trajectories.shape[0])
            ),
            deterministic_execution=True,
            grid=grid,
            body=body,
        )
        candidates = tuple(candidate for candidate in panel.candidates if candidate.valid)
        if not candidates or candidates[0].registered_index != 0:
            raise ValueError("valid native candidate is missing")

        cache_id = f"{source_row['log_name']}/{source_row['scene_token']}"
        teacher_record = teacher_by_id[cache_id]
        label_path = args.teacher_root / "labels" / str(teacher_record["artifact_name"])
        if sha256_file(label_path) != teacher_record["artifact_sha256"]:
            raise ValueError("teacher artifact hash mismatch")
        with np.load(label_path, allow_pickle=False) as label_payload:
            labels = np.asarray(label_payload["label_m"], dtype=np.float64).reshape(-1)

        parent_pairs = {
            int(pair["comparison_registered_index"]): pair
            for pair in b1_by_key[(group, anchor)]["pair_intervals"]
        }
        native = candidates[0]
        pair_rows: list[dict[str, Any]] = []
        for challenger in candidates[1:]:
            set_min = analyze_set_min_pair(
                native.node_indices,
                challenger.node_indices,
                labels,
            )
            parent = parent_pairs.get(challenger.registered_index)
            if parent is None or float(parent["teacher_contrast_m"]) != set_min.contrast:
                raise ValueError("B1 teacher contrast replay mismatch")
            trajectory = analyze_trajectory_pair(
                trajectories[native.registered_index],
                trajectories[challenger.registered_index],
                grid_resolution_m=grid.resolution,
            )
            pair_rows.append(
                {
                    "reference_registered_index": native.registered_index,
                    "comparison_registered_index": challenger.registered_index,
                    "set_min": asdict(set_min),
                    "trajectory": asdict(trajectory),
                }
            )
        if set(parent_pairs) != {
            int(pair["comparison_registered_index"]) for pair in pair_rows
        }:
            raise ValueError("B1 pair identities do not align")
        rows.append(
            {
                "recording_group": group,
                "anchor_index": anchor,
                "log_name": str(source_row["log_name"]),
                "scene_token": str(source_row["scene_token"]),
                "panel_artifact_sha256": source_row["panel_artifact_sha256"],
                "teacher_artifact_sha256": teacher_record["artifact_sha256"],
                "pair_anatomy": pair_rows,
            }
        )

    expected = cast(Mapping[str, int], stage["expected_scope_counts"])
    summary = _summarize(
        rows,
        dominance_fraction=float(stage["decision_protocol"]["dominance_fraction"]),
        minimum_affected_groups=int(
            stage["decision_protocol"]["minimum_affected_groups"]
        ),
    )
    for key, observed in {
        "anchor_count": summary["anchor_count"],
        "recording_group_count": summary["recording_group_count"],
        "pair_count": summary["pair_count"],
        "exact_zero_contrast_pair_count": summary["exact_zero_contrast_pair_count"],
    }.items():
        if int(observed) != int(expected[key]):
            raise ValueError(f"scope count mismatch: {key}")
    result = {
        "experiment_id": config["experiment_id"],
        "code_commit": args.code_commit,
        "forbidden_payload_accessed": False,
        "parent_b1_decision_preserved": True,
        "anchor_ledgers": rows,
        "summary": summary,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--h7bs4a-result", type=Path, required=True)
    parser.add_argument("--b1-result", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--panel-run-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False, allow_nan=False))
