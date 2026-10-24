"""Recover H7BS1 fit result from complete artifacts after JSON serialization failure.

This script does not generate a second scientific panel.  It replays the
official deterministic execution, requires bit-exact equality with every
already-written artifact, recomputes the frozen prefix rule, and writes the
missing finite JSON result.
"""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.evaluation.h4 import (
    qualify_intent_match,
    sample_cached_native_trajectory,
    terminal_segment_average_speed,
)
from aorta_drive.evaluation.navsim_execution import execute_official_navsim_batch
from aorta_drive.evaluation.navsim_geometry import vehicle_geometry_contract
from aorta_drive.evaluation.support_adequacy import (
    evaluate_prefix_panel,
    summarize_prefix_qualification,
)
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs1_fit_prefix import (
    _grid,
    _groups,
    _inside,
    _intent_limits,
    _metric_records,
    _vehicle_matches,
    _verify_implementation,
    _write_json,
)


def _artifact_hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        normalized = relative.lstrip("* ").replace("\\", "/")
        if normalized in result:
            raise ValueError("panel manifest contains a duplicate path")
        result[normalized] = digest
    if len(result) != 672:
        raise ValueError("panel manifest does not contain exactly 672 artifacts")
    return result


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if sha256_file(args.failed_run_log) != args.expected_failed_run_log_sha256:
        raise ValueError("failed-run log hash does not match the registered failure")
    if sha256_file(args.panel_manifest) != args.expected_panel_manifest_sha256:
        raise ValueError("panel manifest hash does not match the registered complete bundle")
    if "ValueError: Out of range float values are not JSON compliant: nan" not in (
        args.failed_run_log.read_text(encoding="utf-8")
    ):
        raise ValueError("failed-run log does not contain the registered serialization failure")

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage = config["H7BS1"]
    _verify_implementation(
        stage,
        repository=Path(__file__).resolve().parents[1],
        navsim_root=args.navsim_root,
    )
    frozen = stage["frozen_input_sha256"]
    for key, path in {
        "group_role_manifest": args.group_role_manifest,
        "metric_manifest": args.metric_manifest,
        "h7b_config": args.h7b_config,
    }.items():
        if sha256_file(path) != frozen[key]:
            raise ValueError(f"frozen input hash mismatch: {key}")

    artifact_hashes = _artifact_hashes(args.panel_manifest)
    groups = _groups(args.group_role_manifest)
    records = _metric_records(args.metric_manifest, groups)
    grid = _grid(stage)
    limits = _intent_limits(stage)
    prefixes = tuple(int(value) for value in stage["prefix_indices"])
    amplitudes = tuple(
        float(value) for value in stage["candidate_panel"]["challenger_amplitudes_m"]
    )
    comparison_indices = np.arange(0, 41, 5, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for number, record in enumerate(records, start=1):
        relative = f"panels/{record['scene_token']}.npz"
        artifact = (args.run_directory / relative).resolve()
        if not artifact.is_relative_to(args.run_directory.resolve()):
            raise ValueError("panel artifact escapes the failed run directory")
        expected_hash = artifact_hashes.get(relative)
        if expected_hash is None or sha256_file(artifact) != expected_hash:
            raise ValueError("panel artifact does not match the frozen artifact manifest")
        with np.load(artifact, allow_pickle=False) as archive:
            stored_realized = np.asarray(archive["realized_local_poses"], dtype=np.float64)
            stored_eligible = tuple(bool(value) for value in archive["challenger_eligible"])

        cache_path = _inside(args.metric_root, str(record["relative_cache_path"]))
        cache_hash = sha256_file(cache_path)
        if cache_hash != record.get("cache_sha256"):
            raise ValueError("metric cache hash mismatch")
        with lzma.open(cache_path, "rb") as stream:
            cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM metric cache
        contract = vehicle_geometry_contract(cache.ego_state.car_footprint.vehicle_parameters)
        if not _vehicle_matches(stage, contract):
            raise ValueError("official vehicle geometry changed")
        native = sample_cached_native_trajectory(cache)
        desired = generate_native_offset_candidates(
            native.poses,
            amplitudes_m=amplitudes,
            minimum_reference_displacement_m=1.0e-6,
            enforce_desired_curve_dynamic_screen=False,
        )
        if len(desired.candidates) != 6:
            raise ValueError("frozen provider did not emit exactly six challengers")
        desired_array = np.stack([candidate.poses for candidate in desired.candidates])
        replay = execute_official_navsim_batch(cache, desired_array)
        if not replay.deterministic_replay_bit_exact:
            raise ValueError("official execution replay is not bit exact")
        if not np.array_equal(replay.realized_local_poses, stored_realized):
            raise ValueError("official execution does not reproduce the stored fit artifact")
        half = replay.realized_local_poses[:, comparison_indices]
        native_half = half[0]
        eligibility = tuple(
            qualify_intent_match(
                half[index],
                native_half,
                challenger_terminal_speed_mps=terminal_segment_average_speed(
                    half[index], interval_s=0.5
                ),
                native_terminal_speed_mps=terminal_segment_average_speed(
                    native_half, interval_s=0.5
                ),
                limits=limits,
            )
            for index in range(1, half.shape[0])
        )
        replay_eligible = tuple(row.eligible for row in eligibility)
        if replay_eligible != stored_eligible:
            raise ValueError("intent eligibility does not reproduce the stored fit artifact")
        statuses = {
            str(prefix): asdict(
                evaluate_prefix_panel(
                    stored_realized,
                    challenger_eligible=stored_eligible,
                    prefix_index=prefix,
                    deterministic_execution=True,
                    grid=grid,
                    body=contract.body,
                )
            )
            for prefix in prefixes
        }
        rows.append(
            {
                "recording_group": str(record["recording_group"]),
                "anchor_index": int(record["anchor_index"]),
                "scene_token": str(record["scene_token"]),
                "log_name": str(record["log_name"]),
                "metric_cache_sha256": cache_hash,
                "execution_deterministic_replay_bit_exact": True,
                "challenger_eligible": list(stored_eligible),
                "prefix_statuses": statuses,
                "panel_artifact": relative,
                "panel_artifact_sha256": expected_hash,
                "recovery_replay_matches_artifact_bit_exact": True,
            }
        )
        if number % 16 == 0:
            print(
                json.dumps(
                    {
                        "phase": "fit_result_materialization",
                        "completed_anchor_count": number,
                        "total_anchor_count": 672,
                        "field_teacher_outcome_accessed": False,
                        "development_accessed": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = summarize_prefix_qualification(
        rows,
        group_names=groups,
        prefix_indices=prefixes,
        anchors_per_group=16,
        minimum_valid_anchors_per_group=int(stage["minimum_valid_anchors_per_group"]),
    )
    result = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED_AFTER_REGISTERED_SERIALIZATION_RECOVERY",
        "phase": "fit_prefix_selection",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "original_runner_sha256": stage["implementation_sha256"]["runner_source"],
        "materializer_sha256": sha256_file(Path(__file__)),
        "failed_run_log_sha256": sha256_file(args.failed_run_log),
        "panel_manifest_sha256": sha256_file(args.panel_manifest),
        "panel_artifact_count": 672,
        "all_execution_replays_match_artifacts_bit_exact": True,
        "anchor_ledgers": rows,
        "summary": summary,
        "decision": summary["decision"],
        "elapsed_seconds_materialization": time.perf_counter() - started,
        "field_teacher_outcome_accessed": False,
        "development_accessed": False,
        "calibration_test_navdream_or_target_accessed": False,
        "claim_boundary": (
            "Fit-only deterministic support qualification recovered from a complete frozen "
            "artifact bundle after strict-JSON serialization failure; no visual field, teacher, "
            "outcome, decision relevance, driving, safety, NavDream, target, or novelty claim."
        ),
    }
    _write_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--h7b-config", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--navsim-root", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--failed-run-log", type=Path, required=True)
    parser.add_argument("--expected-panel-manifest-sha256", required=True)
    parser.add_argument("--expected-failed-run-log-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()
    result = materialize(args)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "anchor_ledgers"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
