"""Diagnose whether unsupported query nodes can control the exact teacher set-min."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import (
    RectangularGrid,
    VehicleRectangle,
)
from aorta_drive.evaluation.shadow_decision import build_shadow_panel
from aorta_drive.evaluation.teacher_criticality import (
    CriticalSupport,
    critical_support,
    outward_node_intervals,
)
from aorta_drive.evaluation.visual_observability import pairwise_contrast_observability
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4a_fit_projective_observability import _projective_evidence, _write_json


def _critical_payload(value: CriticalSupport) -> dict[str, Any]:
    return {
        "possible_minimizer_count": len(value.possible_minimizer_indices),
        "supported_possible_minimizer_count": len(value.supported_possible_minimizer_indices),
        "unsupported_possible_minimizer_count": len(
            value.unsupported_possible_minimizer_indices
        ),
        "all_possible_minimizers_supported": value.all_possible_minimizers_supported,
        "possible_minimizer_indices": list(value.possible_minimizer_indices),
        "unsupported_possible_minimizer_indices": list(
            value.unsupported_possible_minimizer_indices
        ),
    }


def _intersection_count(left: set[int], right: set[int]) -> int:
    return len(left & right)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [candidate for row in rows for candidate in row["candidate_criticality"]]
    pairs = [pair for row in rows for pair in row["pair_criticality"]]
    distinct = [pair for pair in pairs if pair["geometrically_distinct"]]
    blind_difference = [
        pair for pair in distinct if pair["symmetric_difference_supported_node_count"] == 0
    ]
    critical_blind_difference = [
        pair for pair in blind_difference if pair["blind_difference_is_teacher_critical"]
    ]
    failed_groups = sorted(
        {
            str(row["recording_group"])
            for row in rows
            if any(
                not candidate["all_possible_minimizers_supported"]
                for candidate in row["candidate_criticality"]
            )
        }
    )
    failed_candidates = [
        candidate
        for candidate in candidates
        if not candidate["all_possible_minimizers_supported"]
    ]
    if failed_candidates:
        decision = "TEACHER_CRITICAL_PROJECTIVE_BLINDNESS_PRESENT"
    else:
        decision = "STRICT_ALL_NODE_GATE_OVERCONSERVATIVE_ON_FIT_TEACHER"
    if critical_blind_difference:
        d0_decision = "D0_BLIND_DIFFERENCE_CAN_CONTROL_SET_MIN"
    else:
        d0_decision = "D0_BLIND_DIFFERENCE_NOT_TEACHER_CRITICAL_ON_FIT"
    return {
        "anchor_count": len(rows),
        "candidate_count": len(candidates),
        "candidate_with_unsupported_possible_minimizer_count": len(failed_candidates),
        "recording_group_with_critical_blindness_count": len(failed_groups),
        "recording_groups_with_critical_blindness": failed_groups,
        "possible_minimizer_node_count": sum(
            int(candidate["possible_minimizer_count"]) for candidate in candidates
        ),
        "unsupported_possible_minimizer_node_count": sum(
            int(candidate["unsupported_possible_minimizer_count"])
            for candidate in candidates
        ),
        "native_challenger_pair_count": len(pairs),
        "geometrically_distinct_pair_count": len(distinct),
        "zero_supported_symmetric_difference_pair_count": len(blind_difference),
        "teacher_critical_zero_supported_symmetric_difference_pair_count": len(
            critical_blind_difference
        ),
        "noncritical_zero_supported_symmetric_difference_pair_count": len(blind_difference)
        - len(critical_blind_difference),
        "scientific_decision": decision,
        "d0_specific_decision": d0_decision,
        "parent_h7bs4a_decision": "STOP_OR_REDESIGN_VISUAL_SUPPORT_INTERFACE_UNCHANGED",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(Mapping[str, Any], config["H7BS4B0B"])
    repository = Path(__file__).resolve().parents[1]
    frozen_inputs = cast(Mapping[str, str], stage["frozen_input_sha256"])
    for name, path in {
        "h7bs4a_result": args.h7bs4a_result,
        "h7bs1_fit_result": args.fit_result,
        "fit_forward96_teacher_manifest": args.teacher_manifest,
    }.items():
        if sha256_file(path) != frozen_inputs[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")
    implementation = cast(Mapping[str, str], stage["implementation_sha256"])
    for name, path in {
        "criticality_source": repository
        / "src/aorta_drive/evaluation/teacher_criticality.py",
        "observability_source": repository
        / "src/aorta_drive/evaluation/visual_observability.py",
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
    if teacher.get("status") != "completed" or teacher.get("anchor_count") != 672:
        raise ValueError("fit forward96 teacher is not qualified")
    teacher_by_id = {
        str(record["cache_id"]): record
        for record in cast(list[dict[str, Any]], teacher["records"])
    }
    if len(teacher_by_id) != 672:
        raise ValueError("fit forward96 teacher record identities are incomplete")

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
        lower, upper = outward_node_intervals(labels, errors)

        candidate_rows: list[dict[str, Any]] = []
        critical_by_index: dict[int, CriticalSupport] = {}
        node_by_index: dict[int, set[int]] = {}
        for candidate in candidates:
            critical = critical_support(candidate.node_indices, lower, upper, has_evidence)
            critical_by_index[candidate.registered_index] = critical
            node_by_index[candidate.registered_index] = set(
                int(value) for value in candidate.node_indices
            )
            candidate_rows.append(
                {
                    "registered_index": candidate.registered_index,
                    "query_node_count": len(candidate.node_indices),
                    "teacher_set_min_m": float(np.min(labels[candidate.node_indices])),
                    "teacher_set_min_interval_m": [
                        float(np.min(lower[candidate.node_indices])),
                        float(np.min(upper[candidate.node_indices])),
                    ],
                    **_critical_payload(critical),
                }
            )

        native = candidates[0]
        native_nodes = node_by_index[0]
        native_critical = critical_by_index[0]
        native_pm = set(native_critical.possible_minimizer_indices)
        native_unsupported = set(native_critical.unsupported_possible_minimizer_indices)
        pair_rows: list[dict[str, Any]] = []
        for challenger in candidates[1:]:
            challenger_nodes = node_by_index[challenger.registered_index]
            challenger_critical = critical_by_index[challenger.registered_index]
            challenger_pm = set(challenger_critical.possible_minimizer_indices)
            challenger_unsupported = set(
                challenger_critical.unsupported_possible_minimizer_indices
            )
            common = native_nodes & challenger_nodes
            native_only = native_nodes - challenger_nodes
            challenger_only = challenger_nodes - native_nodes
            attribution = pairwise_contrast_observability(
                native.node_indices, challenger.node_indices, has_evidence
            )
            native_only_critical = _intersection_count(native_unsupported, native_only)
            challenger_only_critical = _intersection_count(
                challenger_unsupported, challenger_only
            )
            pair_rows.append(
                {
                    "reference_registered_index": 0,
                    "comparison_registered_index": challenger.registered_index,
                    "geometrically_distinct": bool(attribution["geometrically_distinct"]),
                    "symmetric_difference_node_count": int(
                        attribution["symmetric_difference"]["node_count"]
                    ),
                    "symmetric_difference_supported_node_count": int(
                        attribution["symmetric_difference"]["supported_node_count"]
                    ),
                    "reference_only_unsupported_possible_minimizer_count": (
                        native_only_critical
                    ),
                    "comparison_only_unsupported_possible_minimizer_count": (
                        challenger_only_critical
                    ),
                    "common_unsupported_possible_minimizer_count": len(
                        (native_unsupported | challenger_unsupported) & common
                    ),
                    "reference_possible_minimizer_in_difference_count": len(
                        native_pm & native_only
                    ),
                    "comparison_possible_minimizer_in_difference_count": len(
                        challenger_pm & challenger_only
                    ),
                    "blind_difference_is_teacher_critical": bool(
                        native_only_critical or challenger_only_critical
                    ),
                    "teacher_contrast_m": float(
                        np.min(labels[challenger.node_indices])
                        - np.min(labels[native.node_indices])
                    ),
                }
            )
        rows.append(
            {
                "recording_group": group,
                "anchor_index": anchor,
                "log_name": log_name,
                "scene_token": scene_token,
                "teacher_artifact_sha256": teacher_record["artifact_sha256"],
                "candidate_criticality": candidate_rows,
                "pair_criticality": pair_rows,
            }
        )

    summary = _summarize(rows)
    elapsed = time.perf_counter() - started
    if elapsed > float(stage["resources"]["maximum_wall_minutes"]) * 60.0:
        raise RuntimeError("H7BS4B0B exceeded frozen wall-time cap")
    result = {
        "experiment_id": config["experiment_id"],
        "code_commit": args.code_commit,
        "forbidden_payload_accessed": False,
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
