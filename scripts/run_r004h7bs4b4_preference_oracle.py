"""Build the source-only NAVSIM candidate-preference oracle for H7BS4B4."""

from __future__ import annotations

import argparse
import copy
import csv
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
from aorta_drive.evaluation.h4 import sample_cached_native_trajectory
from aorta_drive.evaluation.navsim_execution import execute_official_navsim_batch
from aorta_drive.evaluation.preference_oracle import (
    PreferenceOracleProtocol,
    summarize_preference_oracle,
)
from aorta_drive.vision.feature_cache import sha256_file


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes its declared root")
    return path


def _groups(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = tuple(row["recording_group"] for row in rows if row["role"] == "fit")
    if len(groups) != 42 or len(set(groups)) != 42:
        raise ValueError("role manifest does not define 42 unique fit groups")
    return groups


def _metric_records(
    path: Path, fit_groups: tuple[str, ...]
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("records")
    if not isinstance(raw, list):
        raise ValueError("metric manifest lacks records")
    order = {group: index for index, group in enumerate(fit_groups)}
    records = [
        cast(dict[str, Any], row)
        for row in raw
        if isinstance(row, dict) and row.get("recording_group") in order
    ]
    records.sort(
        key=lambda row: (order[str(row["recording_group"])], int(row["anchor_index"]))
    )
    expected = {(group, anchor) for group in fit_groups for anchor in range(16)}
    observed = {
        (str(row["recording_group"]), int(row["anchor_index"])) for row in records
    }
    if observed != expected or len(records) != 672:
        raise ValueError("metric manifest does not exactly cover 42 fit groups x 16 anchors")
    if any(row.get("split") != "train" for row in records):
        raise ValueError("preference oracle attempted to leave the train split")
    return records


def _scoring_objects(stage: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (  # type: ignore[import-untyped]
        TrajectorySampling,
    )

    from navsim.planning.simulation.observation.navsim_idm_agents import (  # type: ignore[import-not-found]
        NavsimIDMAgents,
    )
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (  # type: ignore[import-not-found]
        PDMScorer,
        PDMScorerConfig,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (  # type: ignore[import-not-found]
        PDMSimulator,
    )
    from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import (  # type: ignore[import-not-found]
        NavsimIDMTrafficAgents,
    )

    proposal = TrajectorySampling(num_poses=40, interval_length=0.1)
    scorer_values = cast(Mapping[str, Any], stage["one_frame_scorer"])
    scorer_config = PDMScorerConfig(
        progress_weight=float(scorer_values["progress_weight"]),
        ttc_weight=float(scorer_values["ttc_weight"]),
        lane_keeping_weight=float(scorer_values["lane_keeping_weight"]),
        history_comfort_weight=float(scorer_values["history_comfort_weight"]),
        two_frame_extended_comfort_weight=float(
            scorer_values["two_frame_extended_comfort_weight"]
        ),
        driving_direction_horizon=float(
            scorer_values["driving_direction_horizon"]
        ),
        driving_direction_compliance_threshold=float(
            scorer_values["driving_direction_compliance_threshold"]
        ),
        driving_direction_violation_threshold=float(
            scorer_values["driving_direction_violation_threshold"]
        ),
        stopped_speed_threshold=float(scorer_values["stopped_speed_threshold"]),
        future_collision_horizon_window=float(
            scorer_values["future_collision_horizon_window"]
        ),
        progress_distance_threshold=float(
            scorer_values["progress_distance_threshold"]
        ),
        lane_keeping_deviation_limit=float(
            scorer_values["lane_keeping_deviation_limit"]
        ),
        lane_keeping_horizon_window=float(
            scorer_values["lane_keeping_horizon_window"]
        ),
        human_penalty_filter=bool(scorer_values["human_penalty_filter"]),
    )
    idm_values = cast(Mapping[str, Any], stage["reactive_traffic_policy"])
    idm = NavsimIDMAgents(
        target_velocity=float(idm_values["target_velocity"]),
        min_gap_to_lead_agent=float(idm_values["min_gap_to_lead_agent"]),
        headway_time=float(idm_values["headway_time"]),
        accel_max=float(idm_values["accel_max"]),
        decel_max=float(idm_values["decel_max"]),
        open_loop_detections_types=list(idm_values["open_loop_detections_types"]),
        minimum_path_length=float(idm_values["minimum_path_length"]),
        planned_trajectory_samples=None,
        planned_trajectory_sample_interval=None,
        radius=float(idm_values["radius"]),
        add_open_loop_parked_vehicles=bool(
            idm_values["add_open_loop_parked_vehicles"]
        ),
        idm_snap_threshold=float(idm_values["idm_snap_threshold"]),
    )
    return (
        proposal,
        PDMSimulator(proposal),
        PDMScorer(proposal, scorer_config),
        NavsimIDMTrafficAgents(proposal, idm),
    )


def _score_candidate(
    cache: Any,
    poses: np.ndarray[Any, np.dtype[np.float64]],
    *,
    proposal_sampling: Any,
    simulator: Any,
    scorer: Any,
    traffic_policy: Any,
) -> tuple[dict[str, Any], np.ndarray[Any, np.dtype[np.float64]]]:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    from navsim.common.dataclasses import Trajectory  # type: ignore[import-not-found]
    from navsim.evaluate.pdm_score import pdm_score  # type: ignore[import-not-found]

    local_cache = copy.copy(cache)
    local_cache.observation = copy.deepcopy(cache.observation)
    desired_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
    frame, simulated = pdm_score(
        metric_cache=local_cache,
        model_trajectory=Trajectory(
            poses=np.asarray(poses, dtype=np.float64),
            trajectory_sampling=desired_sampling,
        ),
        future_sampling=proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_policy,
    )
    if len(frame) != 1:
        raise ValueError("official scorer did not return exactly one candidate row")
    row = frame.iloc[0]
    hard_names = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "traffic_light_compliance",
    )
    weighted_names = (
        "ego_progress",
        "time_to_collision_within_bound",
        "lane_keeping",
        "history_comfort",
    )
    hard_metrics = {name: float(row[name]) for name in hard_names}
    weighted_metrics = {name: float(row[name]) for name in weighted_names}
    outcome: dict[str, Any] = {
        "one_frame_pdm_score": float(row["pdm_score"]),
        "hard_metrics": hard_metrics,
        "weighted_metrics": weighted_metrics,
    }
    flattened = np.asarray(
        [
            outcome["one_frame_pdm_score"],
            *hard_metrics.values(),
            *weighted_metrics.values(),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(flattened)):
        raise ValueError("official scorer returned a nonfinite outcome")
    return outcome, np.asarray(simulated, dtype=np.float64)


def _process_anchor(task: tuple[dict[str, Any], dict[str, Any], str, str]) -> dict[str, Any]:
    record, stage, metric_root_text, panel_root_text = task
    metric_root = Path(metric_root_text)
    panel_root = Path(panel_root_text)
    cache_path = _inside(metric_root, str(record["relative_cache_path"]))
    if sha256_file(cache_path) != record.get("cache_sha256"):
        raise ValueError("metric cache hash mismatch")
    with lzma.open(cache_path, "rb") as stream:
        cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM cache

    native = sample_cached_native_trajectory(cache)
    amplitudes = tuple(float(value) for value in stage["candidate_panel"]["amplitudes_m"])
    generated = generate_native_offset_candidates(
        native.poses,
        amplitudes_m=amplitudes,
        minimum_reference_displacement_m=1.0e-6,
        enforce_desired_curve_dynamic_screen=False,
    )
    if len(generated.candidates) != 6:
        raise ValueError("frozen provider did not generate six challengers")
    challenger_poses = np.stack([candidate.poses for candidate in generated.candidates])
    execution = execute_official_navsim_batch(cache, challenger_poses)

    panel_path = panel_root / "panels" / f"{record['scene_token']}.npz"
    expected_panel_sha = stage["panel_sha256_by_token"].get(str(record["scene_token"]))
    if expected_panel_sha is None or sha256_file(panel_path) != expected_panel_sha:
        raise ValueError("panel artifact hash mismatch")
    with np.load(panel_path, allow_pickle=False) as payload:
        frozen_realized = np.asarray(payload["realized_local_poses"], dtype=np.float64)
        frozen_eligible = np.asarray(payload["challenger_eligible"], dtype=np.bool_)
    reconstruction_exact = bool(
        np.array_equal(execution.realized_local_poses, frozen_realized)
        and frozen_eligible.shape == (6,)
        and np.all(frozen_eligible)
    )

    candidate_poses = [np.asarray(native.poses, dtype=np.float64), *challenger_poses]
    proposal, simulator, scorer, policy = _scoring_objects(stage)
    candidates: list[dict[str, Any]] = []
    primary_states: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for registered_index, poses in enumerate(candidate_poses):
        outcome, simulated = _score_candidate(
            cache,
            poses,
            proposal_sampling=proposal,
            simulator=simulator,
            scorer=scorer,
            traffic_policy=policy,
        )
        candidates.append({"registered_index": registered_index, **outcome})
        primary_states.append(simulated)

    sentinel = int(record["anchor_index"]) == 0
    replay_exact = True
    if sentinel:
        replay_proposal, replay_simulator, replay_scorer, replay_policy = _scoring_objects(
            stage
        )
        replay_candidates: list[dict[str, Any]] = []
        replay_states: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        for registered_index, poses in enumerate(candidate_poses):
            outcome, simulated = _score_candidate(
                cache,
                poses,
                proposal_sampling=replay_proposal,
                simulator=replay_simulator,
                scorer=replay_scorer,
                traffic_policy=replay_policy,
            )
            replay_candidates.append({"registered_index": registered_index, **outcome})
            replay_states.append(simulated)
        replay_exact = bool(
            candidates == replay_candidates
            and all(
                np.array_equal(primary, replay)
                for primary, replay in zip(primary_states, replay_states)
            )
        )

    return {
        "recording_group": str(record["recording_group"]),
        "anchor_index": int(record["anchor_index"]),
        "log_name": str(record["log_name"]),
        "scene_token": str(record["scene_token"]),
        "metric_cache_sha256": str(record["cache_sha256"]),
        "panel_artifact_sha256": str(expected_panel_sha),
        "panel_reconstruction_bit_exact": reconstruction_exact,
        "sentinel_replay_required": sentinel,
        "sentinel_replay_exact": replay_exact,
        "label_generation_valid": True,
        "candidates": candidates,
    }


def _verify_sources(
    stage: Mapping[str, Any], *, repository: Path, navsim_root: Path
) -> None:
    frozen = cast(Mapping[str, Any], stage["implementation_sha256"])
    local = {
        "preference_oracle_source": repository
        / "src/aorta_drive/evaluation/preference_oracle.py",
        "navsim_execution_source": repository
        / "src/aorta_drive/evaluation/navsim_execution.py",
        "native_offset_source": repository / "src/aorta_drive/candidates/native_offset.py",
        "h4_source": repository / "src/aorta_drive/evaluation/h4.py",
        "runner_source": Path(__file__).resolve(),
    }
    for name, path in local.items():
        if sha256_file(path) != frozen[name]:
            raise ValueError(f"local implementation hash mismatch: {name}")
    for relative, expected in cast(Mapping[str, str], frozen["official_source"]).items():
        if sha256_file(navsim_root / relative) != expected:
            raise ValueError(f"official implementation hash mismatch: {relative}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(dict[str, Any], config["H7BS4B4"])
    repository = Path(__file__).resolve().parents[1]
    _verify_sources(stage, repository=repository, navsim_root=args.navsim_root)
    frozen = cast(Mapping[str, str], stage["frozen_input_sha256"])
    for name, path in {
        "group_role_manifest": args.group_role_manifest,
        "metric_manifest": args.metric_manifest,
        "h7bs1_fit_result": args.fit_result,
    }.items():
        if sha256_file(path) != frozen[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)

    groups = _groups(args.group_role_manifest)
    records = _metric_records(args.metric_manifest, groups)
    fit = cast(dict[str, Any], json.loads(args.fit_result.read_text(encoding="utf-8")))
    if fit.get("decision") != "FIT_PREFIX_SELECTED" or len(fit.get("anchor_ledgers", [])) != 672:
        raise ValueError("frozen H7BS1 fit result is not qualified")
    panel_hashes = {
        str(row["scene_token"]): str(row["panel_artifact_sha256"])
        for row in cast(list[dict[str, Any]], fit["anchor_ledgers"])
    }
    if len(panel_hashes) != 672:
        raise ValueError("frozen panel hash ledger is incomplete")
    worker_stage = copy.deepcopy(stage)
    worker_stage["panel_sha256_by_token"] = panel_hashes
    tasks = [
        (record, worker_stage, str(args.metric_root), str(args.panel_root))
        for record in records
    ]
    workers = int(stage["resources"]["workers"])
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed, row in enumerate(executor.map(_process_anchor, tasks), start=1):
            rows.append(row)
            if completed % 16 == 0:
                progress = {
                    "experiment_id": config["experiment_id"],
                    "status": "IN_PROGRESS",
                    "completed_anchor_count": completed,
                    "total_anchor_count": 672,
                    "development_calibration_test_navdream_target_accessed": False,
                    "scientific_metrics_emitted": False,
                }
                _write_json(args.output_directory / "progress.json", progress)
                print(json.dumps(progress, sort_keys=True), flush=True)
            elapsed = time.perf_counter() - started
            if elapsed > float(stage["resources"]["maximum_wall_minutes"]) * 60.0:
                raise RuntimeError("preference oracle exceeded its wall-time cap")

    rows.sort(key=lambda row: (groups.index(row["recording_group"]), row["anchor_index"]))
    protocol_values = cast(Mapping[str, Any], stage["decision_protocol"])
    protocol = PreferenceOracleProtocol(
        numerical_tie_tolerance=float(protocol_values["numerical_tie_tolerance"]),
        minimum_groups_with_nonzero_pair=int(
            protocol_values["minimum_groups_with_nonzero_pair"]
        ),
        minimum_groups_with_strict_native_improvement=int(
            protocol_values["minimum_groups_with_strict_native_improvement"]
        ),
        bootstrap_seed=int(protocol_values["bootstrap_seed"]),
        bootstrap_resamples=int(protocol_values["bootstrap_resamples"]),
    )
    summary = summarize_preference_oracle(
        rows,
        group_names=groups,
        protocol=protocol,
        diagnostic_margins=tuple(
            float(value) for value in protocol_values["diagnostic_absolute_margins"]
        ),
    )
    result = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": args.code_commit,
        "config_sha256": sha256_file(args.config),
        "label_estimand": (
            "candidate-specific one-frame NAVSIM PDM score, each candidate independently "
            "paired with the same native trajectory"
        ),
        "full_official_pdms_claimed": False,
        "development_calibration_test_navdream_target_accessed": False,
        "anchor_ledgers": rows,
        "summary": summary,
        "decision": summary["decision"],
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Fit-only label legality and oracle signal qualification. This is not learned "
            "performance, held-out generalization, NavDream robustness, full official PDMS, "
            "driving safety, architecture necessity, or top-venue evidence."
        ),
    }
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--navsim-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "anchor_ledgers"},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
