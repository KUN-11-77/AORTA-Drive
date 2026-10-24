"""Run the candidate-domain-safe source preference oracle for H7BS4B4R1."""

from __future__ import annotations

import argparse
import copy
import json
import lzma
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.evaluation.eligible_preference_oracle import (
    EligiblePreferenceOracleProtocol,
    summarize_eligible_preference_oracle,
)
from aorta_drive.evaluation.h4 import sample_cached_native_trajectory
from aorta_drive.evaluation.navsim_execution import execute_official_navsim_batch
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4b4_preference_oracle import (
    _groups,
    _inside,
    _metric_records,
    _score_candidate,
    _scoring_objects,
    _write_json,
)


def _require_progress_mode(required: str) -> str:
    """Fail closed before scoring if NAVSIM would use a different estimand."""

    observed = os.environ.get("PROGRESS_MODE", "eval")
    if observed != required:
        raise RuntimeError(
            f"PROGRESS_MODE must be {required!r}, observed {observed!r}"
        )
    return observed


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
    if frozen_eligible.shape != (6,):
        raise ValueError("frozen membership artifact must contain six challengers")
    pose_exact = bool(np.array_equal(execution.realized_local_poses, frozen_realized))

    registered_indices = [0, *[index + 1 for index, flag in enumerate(frozen_eligible) if flag]]
    candidate_poses = [
        np.asarray(native.poses[1:], dtype=np.float64),
        *[
            challenger_poses[index]
            for index, flag in enumerate(frozen_eligible)
            if flag
        ],
    ]
    if len(candidate_poses) != len(registered_indices):
        raise AssertionError("candidate-domain construction mismatch")

    proposal, simulator, scorer, policy = _scoring_objects(stage)
    candidates: list[dict[str, Any]] = []
    primary_states: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for registered_index, poses in zip(registered_indices, candidate_poses):
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
        replay_proposal, replay_simulator, replay_scorer, replay_policy = _scoring_objects(stage)
        replay_candidates: list[dict[str, Any]] = []
        replay_states: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        for registered_index, poses in zip(registered_indices, candidate_poses):
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

    scored_indices = {int(candidate["registered_index"]) for candidate in candidates}
    ineligible_indices = {
        index + 1 for index, flag in enumerate(frozen_eligible) if not flag
    }
    return {
        "recording_group": str(record["recording_group"]),
        "anchor_index": int(record["anchor_index"]),
        "log_name": str(record["log_name"]),
        "scene_token": str(record["scene_token"]),
        "metric_cache_sha256": str(record["cache_sha256"]),
        "panel_artifact_sha256": str(expected_panel_sha),
        "panel_pose_reconstruction_bit_exact": pose_exact,
        "membership_artifact_exact": True,
        "frozen_challenger_eligible": frozen_eligible.tolist(),
        "legal_registered_indices": registered_indices,
        "ineligible_candidate_score_count": len(scored_indices & ineligible_indices),
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
        "eligible_preference_oracle_source": repository
        / "src/aorta_drive/evaluation/eligible_preference_oracle.py",
        "navsim_execution_source": repository
        / "src/aorta_drive/evaluation/navsim_execution.py",
        "native_offset_source": repository / "src/aorta_drive/candidates/native_offset.py",
        "h4_source": repository / "src/aorta_drive/evaluation/h4.py",
        "b4_runner_dependency": repository
        / "scripts/run_r004h7bs4b4_preference_oracle.py",
        "runner_source": Path(__file__).resolve(),
    }
    for name, path in local.items():
        if sha256_file(path) != frozen[name]:
            raise ValueError(f"local implementation hash mismatch: {name}")
    for relative, expected in cast(Mapping[str, str], frozen["official_source"]).items():
        if sha256_file(navsim_root / relative) != expected:
            raise ValueError(f"official implementation hash mismatch: {relative}")


def _protocol(stage: Mapping[str, Any]) -> EligiblePreferenceOracleProtocol:
    values = cast(Mapping[str, Any], stage["decision_protocol"])
    domain = cast(Mapping[str, Any], stage["candidate_domain"])
    return EligiblePreferenceOracleProtocol(
        numerical_tie_tolerance=float(values["numerical_tie_tolerance"]),
        minimum_groups_with_nonzero_pair=int(values["minimum_groups_with_nonzero_pair"]),
        minimum_groups_with_strict_native_improvement=int(
            values["minimum_groups_with_strict_native_improvement"]
        ),
        bootstrap_seed=int(values["bootstrap_seed"]),
        bootstrap_resamples=int(values["bootstrap_resamples"]),
        expected_eligible_challenger_count=int(domain["expected_eligible_challengers"]),
        expected_zero_eligible_anchor_count=int(domain["expected_zero_eligible_anchors"]),
        expected_nonzero_eligible_anchor_count=int(
            domain["expected_nonzero_eligible_anchors"]
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(dict[str, Any], config["H7BS4B4R1"])
    scorer_contract = cast(Mapping[str, Any], stage["one_frame_scorer"])
    progress_mode = _require_progress_mode(
        str(scorer_contract["required_progress_mode"])
    )
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
    rows: list[dict[str, Any]] = []
    workers = int(stage["resources"]["workers"])
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
            maximum_seconds = (
                float(stage["resources"]["maximum_wall_minutes"]) * 60.0
            )
            if time.perf_counter() - started > maximum_seconds:
                raise RuntimeError("eligible preference oracle exceeded its wall-time cap")

    rows.sort(key=lambda row: (groups.index(row["recording_group"]), row["anchor_index"]))
    protocol_values = cast(Mapping[str, Any], stage["decision_protocol"])
    summary = summarize_eligible_preference_oracle(
        rows,
        group_names=groups,
        protocol=_protocol(stage),
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
        "frozen_input_sha256": dict(frozen),
        "implementation_sha256": copy.deepcopy(stage["implementation_sha256"]),
        "label_estimand": str(stage["estimand"]["result_label"]),
        "progress_mode": progress_mode,
        "zero_eligible_anchor_policy": "retain_anchor_with_native_only_and_zero_gain",
        "full_official_pdms_claimed": False,
        "development_calibration_test_navdream_target_accessed": False,
        "anchor_ledgers": rows,
        "summary": summary,
        "decision": summary["decision"],
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Fit-only legal-domain label and oracle qualification. This is not learned "
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
