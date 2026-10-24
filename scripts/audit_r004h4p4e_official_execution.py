"""Audit the official NAVSIM desired-reference execution operator on fit anchors."""

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
from aorta_drive.evaluation.h4 import (
    IntentEligibilityLimits,
    qualify_intent_match,
    sample_cached_native_trajectory,
    terminal_segment_average_speed,
)
from aorta_drive.evaluation.navsim_execution import execute_official_navsim_batch
from diagnose_r004h4p2_candidate_supply import (
    _metric_records,
    _selected_anchors,
    _sha256,
)


def _angle_difference(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.asarray(
        np.arctan2(np.sin(left - right), np.cos(left - right)), dtype=np.float64
    )


def _limits() -> IntentEligibilityLimits:
    return IntentEligibilityLimits(
        minimum_native_terminal_displacement_m=2.0,
        minimum_progress_ratio=0.8,
        maximum_progress_ratio=1.2,
        maximum_terminal_cross_track_m=3.5,
        maximum_terminal_heading_difference_rad=0.35,
        maximum_terminal_speed_difference_mps=4.0,
    )


def _verify_sources(config: dict[str, Any], repository: Path, navsim_root: Path) -> None:
    local = {
        "execution_bridge_source_sha256": (
            repository / "src/aorta_drive/evaluation/navsim_execution.py"
        ),
        "native_offset_source_sha256": (
            repository / "src/aorta_drive/candidates/native_offset.py"
        ),
    }
    for key, path in local.items():
        if _sha256(path) != config["implementation"][key]:
            raise ValueError(f"{key} disagrees with registration")
    for relative, expected in config["implementation"]["official_source_sha256"].items():
        path = navsim_root / relative
        if _sha256(path) != expected:
            raise ValueError(f"official source hash mismatch: {relative}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for name, path in (
        ("time_set", args.time_set),
        ("role_manifest", args.role_manifest),
        ("metric_manifest", args.metric_manifest),
    ):
        if _sha256(path) != config["inputs"][f"{name}_sha256"]:
            raise ValueError(f"{name} hash disagrees with registration")
    repository = Path(__file__).resolve().parents[1]
    _verify_sources(config, repository, args.navsim_root)
    anchors = _selected_anchors(args.time_set, args.role_manifest)
    if len(anchors) != 128:
        raise ValueError("execution audit requires eight groups times sixteen anchors")
    records = _metric_records(json.loads(args.metric_manifest.read_text(encoding="utf-8")))
    amplitudes = tuple(float(v) for v in config["desired_reference"]["amplitudes_m"])
    gates = config["interface_gates"]
    comparison_indices = np.arange(0, 41, 5, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    realized_supply = 0
    realized_both_sign = 0
    final_identity_survived = 0
    for index, anchor in enumerate(anchors, start=1):
        key = (anchor["log_name"], anchor["scene_token"])
        record = records.get(key)
        if record is None:
            raise ValueError("metric manifest lacks a registered anchor")
        cache_path = args.metric_root / str(record["relative_cache_path"])
        if _sha256(cache_path) != record["cache_sha256"]:
            raise ValueError("metric cache raw hash mismatch")
        with lzma.open(cache_path, "rb") as stream:
            cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM input
        native = sample_cached_native_trajectory(cache)
        desired_batch = generate_native_offset_candidates(
            native.poses,
            amplitudes_m=amplitudes,
            minimum_reference_displacement_m=1e-6,
            enforce_desired_curve_dynamic_screen=False,
        )
        if len(desired_batch.candidates) != len(amplitudes):
            failures.append(f"{key}: desired provider did not emit six references")
            continue
        desired = np.stack([candidate.poses for candidate in desired_batch.candidates])
        execution = execute_official_navsim_batch(cache, desired)
        shape_ok = execution.realized_states.shape[:2] == (
            int(gates["expected_batch_size"]),
            int(gates["expected_time_samples_including_origin"]),
        )
        if not shape_ok:
            failures.append(f"{key}: official state shape mismatch")
            continue
        candidate_reference_half = execution.reference_local_poses[1:, comparison_indices]
        desired_with_origin = np.concatenate(
            [np.zeros((len(amplitudes), 1, 3)), desired], axis=1
        )
        roundtrip_position = np.linalg.norm(
            candidate_reference_half[:, :, :2] - desired_with_origin[:, :, :2], axis=2
        )
        roundtrip_heading = np.abs(
            _angle_difference(
                candidate_reference_half[:, :, 2], desired_with_origin[:, :, 2]
            )
        )
        maximum_roundtrip_position = float(np.max(roundtrip_position))
        maximum_roundtrip_heading = float(np.max(roundtrip_heading))
        realized_initial_shared = bool(
            np.all(execution.realized_states[:, 0] == execution.realized_states[0, 0])
        )
        interface_ok = (
            execution.deterministic_replay_bit_exact
            and realized_initial_shared
            and maximum_roundtrip_position
            <= float(gates["maximum_desired_roundtrip_position_error_m"])
            and maximum_roundtrip_heading
            <= float(gates["maximum_desired_roundtrip_heading_error_rad"])
        )
        if not interface_ok:
            failures.append(f"{key}: interface gate failure")

        reference_half = execution.reference_local_poses[:, comparison_indices]
        realized_half = execution.realized_local_poses[:, comparison_indices]
        realized_native = realized_half[0]
        accepted_signs: set[float] = set()
        eligible_count = 0
        all_final_equal = True
        for candidate_index, amplitude in enumerate(amplitudes, start=1):
            realized_candidate = realized_half[candidate_index]
            outcome = qualify_intent_match(
                realized_candidate,
                realized_native,
                challenger_terminal_speed_mps=terminal_segment_average_speed(
                    realized_candidate, interval_s=0.5
                ),
                native_terminal_speed_mps=terminal_segment_average_speed(
                    realized_native, interval_s=0.5
                ),
                limits=_limits(),
            )
            eligible_count += int(outcome.eligible)
            if outcome.eligible:
                accepted_signs.add(math.copysign(1.0, amplitude))
            all_final_equal = all_final_equal and bool(
                np.array_equal(realized_candidate[-2:], realized_native[-2:])
            )
        has_supply = eligible_count > 0
        has_both = accepted_signs == {-1.0, 1.0}
        realized_supply += int(has_supply)
        realized_both_sign += int(has_both)
        final_identity_survived += int(all_final_equal)
        desired_realized_position = np.linalg.norm(
            execution.realized_local_poses[:, :, :2]
            - execution.reference_local_poses[:, :, :2],
            axis=2,
        )
        rows.append(
            {
                "recording_group": anchor["recording_group"],
                "anchor_index": int(anchor["anchor_index"]),
                "log_name": anchor["log_name"],
                "scene_token": anchor["scene_token"],
                "shape_ok": shape_ok,
                "deterministic_replay_bit_exact": (
                    execution.deterministic_replay_bit_exact
                ),
                "realized_initial_state_shared_bit_exact": realized_initial_shared,
                "maximum_desired_roundtrip_position_error_m": maximum_roundtrip_position,
                "maximum_desired_roundtrip_heading_error_rad": maximum_roundtrip_heading,
                "maximum_transform_error_m": execution.maximum_transform_error_m,
                "native_desired_to_realized_terminal_position_error_m": float(
                    desired_realized_position[0, -1]
                ),
                "maximum_challenger_desired_to_realized_position_error_m": float(
                    np.max(desired_realized_position[1:])
                ),
                "realized_eligible_candidate_count": eligible_count,
                "realized_has_eligible_candidate": has_supply,
                "realized_has_both_signed_eligible_candidates": has_both,
                "desired_final_identity_survived_execution_for_all_candidates": (
                    all_final_equal
                ),
            }
        )
        if index % 16 == 0:
            print(json.dumps({"anchors_completed": index}), flush=True)

    passed = len(rows) == 128 and not failures
    return {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": (
            "PASS_OFFICIAL_EXECUTION_INTERFACE"
            if passed
            else "STOP_OFFICIAL_EXECUTION_INTERFACE"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": _sha256(args.config),
        "elapsed_seconds": time.perf_counter() - started,
        "data_scope": {
            "role": "fit",
            "recording_group_count": 8,
            "anchor_count": 128,
            "geometry_teacher_prediction_image_or_development_read": False,
        },
        "interface_failures": failures,
        "secondary_diagnostics": {
            "realized_anchors_with_eligible_candidate": realized_supply,
            "realized_anchor_supply_fraction": realized_supply / 128.0,
            "realized_anchors_with_both_signed_eligible_candidates": realized_both_sign,
            "desired_final_identity_survived_execution_anchor_count": (
                final_identity_survived
            ),
        },
        "anchor_ledger": rows,
        "claims": {
            "official_execution_interface_closed": passed,
            "realized_supply_is_decision_endpoint": False,
            "geometry_evaluated": False,
            "teacher_or_prediction_read": False,
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
            {key: value for key, value in result.items() if key != "anchor_ledger"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
