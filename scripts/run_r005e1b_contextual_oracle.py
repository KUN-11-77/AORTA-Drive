"""Materialize the frozen native-history contextual candidate oracle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import lzma
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.evaluation.contextual_oracle import (
    contextual_utility,
    summarize_contextual_oracle,
)
from aorta_drive.evaluation.contextual_target_qualification import filtered_one_stage
from aorta_drive.evaluation.h4 import sample_cached_native_trajectory
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4b4_preference_oracle import (
    _groups,
    _inside,
    _metric_records,
    _score_candidate,
    _scoring_objects,
    _write_json,
)
from run_r004h7bs4b4r1_eligible_preference_oracle import _protocol


def _array_sha256(value: np.ndarray[Any, np.dtype[np.float64]]) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _cache_time_s(cache: Any) -> float:
    time_point = cache.ego_state.time_point
    if hasattr(time_point, "time_s"):
        return float(time_point.time_s)
    return float(time_point.time_us) / 1_000_000.0


def _two_frame_comfort(
    current_states: np.ndarray[Any, np.dtype[np.float64]],
    previous_states: np.ndarray[Any, np.dtype[np.float64]],
    *,
    observation_interval_s: float,
    simulation_interval_s: float,
) -> float:
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (  # type: ignore[import-not-found]
        ego_is_two_frame_extended_comfort,
    )

    if not 0.0 < observation_interval_s < 0.55:
        raise ValueError("observation interval is outside the official contract")
    overlap_start = round(observation_interval_s / simulation_interval_s)
    if overlap_start <= 0 or overlap_start >= len(current_states):
        raise ValueError("invalid trajectory overlap")
    current_overlap = current_states[:-overlap_start]
    previous_overlap = previous_states[overlap_start:]
    if current_overlap.shape != previous_overlap.shape:
        raise ValueError("current and previous simulated-state overlaps differ")
    time_points = np.arange(current_overlap.shape[0]) * simulation_interval_s
    value = ego_is_two_frame_extended_comfort(
        current_overlap[None, :],
        previous_overlap[None, :],
        time_points,
    )[0]
    return float(np.asarray(value, dtype=np.float64))


def _load_cache(root: Path, record: Mapping[str, Any]) -> Any:
    path = _inside(root, str(record["relative_cache_path"]))
    if sha256_file(path) != record["cache_sha256"]:
        raise ValueError("metric cache hash mismatch")
    with lzma.open(path, "rb") as stream:
        return pickle.load(stream)  # noqa: S301 - frozen official NAVSIM cache


def _process_anchor(
    task: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        str,
        str,
    ],
) -> dict[str, Any]:
    (
        current_record,
        previous_record,
        b4_row,
        context,
        b4_stage,
        current_root_text,
        previous_root_text,
    ) = task
    current_cache = _load_cache(Path(current_root_text), current_record)
    previous_cache = _load_cache(Path(previous_root_text), previous_record)
    current_time = _cache_time_s(current_cache)
    previous_time = _cache_time_s(previous_cache)
    observed_gap = current_time - previous_time
    registered_gap = float(context["gap_s"])
    if abs(observed_gap - registered_gap) > 1.0e-6:
        raise ValueError("metric-cache timestamps disagree with the frozen context roster")

    current_native = sample_cached_native_trajectory(current_cache)
    previous_native = sample_cached_native_trajectory(previous_cache)
    generated = generate_native_offset_candidates(
        current_native.poses,
        amplitudes_m=tuple(
            float(value) for value in b4_stage["candidate_panel"]["amplitudes_m"]
        ),
        minimum_reference_displacement_m=1.0e-6,
        enforce_desired_curve_dynamic_screen=False,
    )
    if len(generated.candidates) != 6:
        raise ValueError("frozen provider did not generate six challengers")
    challenger_by_index = {
        index + 1: np.asarray(candidate.poses, dtype=np.float64)
        for index, candidate in enumerate(generated.candidates)
    }
    legal_indices = [int(value) for value in b4_row["legal_registered_indices"]]
    poses_by_index = {0: np.asarray(current_native.poses[1:], dtype=np.float64)}
    poses_by_index.update(challenger_by_index)
    candidate_poses = [poses_by_index[index] for index in legal_indices]

    previous_objects = _scoring_objects(b4_stage)
    previous_outcome, previous_states = _score_candidate(
        previous_cache,
        np.asarray(previous_native.poses[1:], dtype=np.float64),
        proposal_sampling=previous_objects[0],
        simulator=previous_objects[1],
        scorer=previous_objects[2],
        traffic_policy=previous_objects[3],
    )
    current_objects = _scoring_objects(b4_stage)
    expected_candidates = cast(list[dict[str, Any]], b4_row["candidates"])
    candidates: list[dict[str, Any]] = []
    for expected, registered_index, poses in zip(
        expected_candidates, legal_indices, candidate_poses
    ):
        outcome, current_states = _score_candidate(
            current_cache,
            poses,
            proposal_sampling=current_objects[0],
            simulator=current_objects[1],
            scorer=current_objects[2],
            traffic_policy=current_objects[3],
        )
        if int(expected["registered_index"]) != registered_index:
            raise ValueError("candidate order differs from B4R1")
        expected_outcome = {
            key: expected[key]
            for key in ("one_frame_pdm_score", "hard_metrics", "weighted_metrics")
        }
        if outcome != expected_outcome:
            raise ValueError("current candidate score does not exactly replay B4R1")
        comfort = _two_frame_comfort(
            current_states,
            previous_states,
            observation_interval_s=observed_gap,
            simulation_interval_s=0.1,
        )
        candidates.append(
            {
                "registered_index": registered_index,
                **outcome,
                "filtered_one_frame_score": filtered_one_stage(outcome),
                "two_frame_extended_comfort": comfort,
                "contextual_utility": contextual_utility(outcome, comfort),
                "current_simulated_states_sha256": _array_sha256(current_states),
            }
        )

    return {
        "recording_group": str(current_record["recording_group"]),
        "anchor_index": int(current_record["anchor_index"]),
        "log_name": str(current_record["log_name"]),
        "scene_token": str(current_record["scene_token"]),
        "previous_token": str(previous_record["scene_token"]),
        "observation_interval_s": observed_gap,
        "current_metric_cache_sha256": str(current_record["cache_sha256"]),
        "previous_metric_cache_sha256": str(previous_record["cache_sha256"]),
        "previous_native_one_frame_outcome": previous_outcome,
        "previous_native_simulated_states_sha256": _array_sha256(previous_states),
        "frozen_challenger_eligible": list(b4_row["frozen_challenger_eligible"]),
        "legal_registered_indices": legal_indices,
        "panel_pose_reconstruction_bit_exact": bool(
            b4_row["panel_pose_reconstruction_bit_exact"]
        ),
        "membership_artifact_exact": bool(b4_row["membership_artifact_exact"]),
        "ineligible_candidate_score_count": 0,
        "sentinel_replay_exact": True,
        "label_generation_valid": True,
        "b4r1_candidate_scores_exact": True,
        "candidates": candidates,
    }


def _indexed(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    result = {
        (str(row["recording_group"]), int(row["anchor_index"])): row for row in rows
    }
    if len(result) != len(rows):
        raise ValueError("duplicate anchor keys")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(dict[str, Any], config["R005E1B"])
    b4_config = cast(
        dict[str, Any], yaml.safe_load(args.b4_config.read_text(encoding="utf-8"))
    )
    b4_stage = cast(dict[str, Any], b4_config["H7BS4B4R1"])
    frozen = cast(Mapping[str, str], stage["frozen_input_sha256"])
    inputs = {
        "e1a_result": args.e1a_result,
        "b4r1_config": args.b4_config,
        "b4r1_primary_result": args.b4_result,
        "temporal_context_roster": args.context_roster,
        "current_metric_manifest": args.current_metric_manifest,
        "previous_metric_manifest": args.previous_metric_manifest,
        "group_role_manifest": args.group_role_manifest,
    }
    for name, path in inputs.items():
        if sha256_file(path) != frozen[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)

    groups = _groups(args.group_role_manifest)
    current_records = _metric_records(args.current_metric_manifest, groups)
    previous_records = _metric_records(args.previous_metric_manifest, groups)
    b4_result = cast(dict[str, Any], json.loads(args.b4_result.read_text(encoding="utf-8")))
    context_roster = cast(
        dict[str, Any], json.loads(args.context_roster.read_text(encoding="utf-8"))
    )
    current_by_key = _indexed(current_records)
    previous_by_key = _indexed(previous_records)
    b4_by_key = _indexed(cast(list[dict[str, Any]], b4_result["anchor_ledgers"]))
    context_by_key = _indexed(cast(list[dict[str, Any]], context_roster["anchor_ledger"]))
    if not (
        set(current_by_key)
        == set(previous_by_key)
        == set(b4_by_key)
        == set(context_by_key)
    ):
        raise ValueError("current, previous, B4R1, and context anchor keys differ")
    for key in current_by_key:
        context = context_by_key[key]
        if current_by_key[key]["scene_token"] != context["current_token"]:
            raise ValueError("current token join mismatch")
        if previous_by_key[key]["scene_token"] != context["previous_token"]:
            raise ValueError("previous token join mismatch")

    tasks = [
        (
            current_by_key[(group, anchor)],
            previous_by_key[(group, anchor)],
            b4_by_key[(group, anchor)],
            context_by_key[(group, anchor)],
            copy.deepcopy(b4_stage),
            str(args.current_metric_root),
            str(args.previous_metric_root),
        )
        for group in groups
        for anchor in range(16)
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(stage["resources"]["workers"])) as executor:
        for completed, row in enumerate(executor.map(_process_anchor, tasks), start=1):
            rows.append(row)
            if completed % 16 == 0:
                _write_json(
                    args.output_directory / "progress.json",
                    {
                        "experiment_id": config["experiment_id"],
                        "status": "IN_PROGRESS",
                        "completed_anchor_count": completed,
                        "total_anchor_count": 672,
                        "development_calibration_test_navdream_target_accessed": False,
                        "scientific_metrics_emitted": False,
                    },
                )

    summary = summarize_contextual_oracle(
        rows,
        group_names=groups,
        protocol=_protocol(b4_stage),
        diagnostic_margins=tuple(
            float(value)
            for value in b4_stage["decision_protocol"]["diagnostic_absolute_margins"]
        ),
    )
    result = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": args.code_commit,
        "config_sha256": sha256_file(args.config),
        "frozen_input_sha256": dict(frozen),
        "target_estimand": "native_history_teacher_forced_contextual_row_utility",
        "q_pi_ref_materialized": False,
        "q_pi_ref_reason": "official_two_stage_mapping_coverage_is_zero_on_source_fit",
        "development_calibration_test_navdream_target_accessed": False,
        "full_official_pdms_claimed": False,
        "anchor_ledgers": rows,
        "summary": summary,
        "decision": summary["decision"],
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Source-fit native-history contextual oracle only; not RGB recoverability, "
            "architecture superiority, pseudo-closed-loop policy value, driving gain, "
            "NavDream robustness, or safety evidence."
        ),
    }
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--e1a-result", type=Path, required=True)
    parser.add_argument("--b4-config", type=Path, required=True)
    parser.add_argument("--b4-result", type=Path, required=True)
    parser.add_argument("--context-roster", type=Path, required=True)
    parser.add_argument("--current-metric-root", type=Path, required=True)
    parser.add_argument("--current-metric-manifest", type=Path, required=True)
    parser.add_argument("--previous-metric-root", type=Path, required=True)
    parser.add_argument("--previous-metric-manifest", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "anchor_ledgers"},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
