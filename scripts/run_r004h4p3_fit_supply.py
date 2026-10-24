"""Measure native-offset candidate supply on the registered 128 fit anchors."""

from __future__ import annotations

import argparse
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
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.evaluation.h4 import (
    IntentEligibilityLimits,
    qualify_intent_match,
    sample_cached_native_trajectory,
    terminal_segment_average_speed,
)
from diagnose_r004h4p2_candidate_supply import (
    _metric_records,
    _selected_anchors,
    _sha256,
)


def _limits(config: dict[str, Any]) -> IntentEligibilityLimits:
    intent = config["intent_eligibility"]
    return IntentEligibilityLimits(
        minimum_native_terminal_displacement_m=float(
            intent["minimum_native_terminal_displacement_m"]
        ),
        minimum_progress_ratio=float(intent["progress_ratio"][0]),
        maximum_progress_ratio=float(intent["progress_ratio"][1]),
        maximum_terminal_cross_track_m=float(intent["maximum_terminal_cross_track_m"]),
        maximum_terminal_heading_difference_rad=float(
            intent["maximum_terminal_heading_difference_rad"]
        ),
        maximum_terminal_speed_difference_mps=float(
            intent["maximum_terminal_speed_difference_mps"]
        ),
    )


def _provider_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    provider = config["provider"]
    return {
        "interval_s": float(config["native"]["interval_s"]),
        "bump_end_s": float(provider["bump_end_s"]),
        "amplitudes_m": tuple(float(value) for value in provider["amplitudes_m"]),
        "minimum_reference_displacement_m": float(
            config["intent_eligibility"]["minimum_native_terminal_displacement_m"]
        ),
        "maximum_speed_mps": float(provider["maximum_speed_mps"]),
        "maximum_longitudinal_acceleration_mps2": float(
            provider["maximum_longitudinal_acceleration_mps2"]
        ),
        "maximum_lateral_acceleration_mps2": float(
            provider["maximum_lateral_acceleration_mps2"]
        ),
        "maximum_total_acceleration_mps2": float(
            provider["maximum_total_acceleration_mps2"]
        ),
        "bernstein_strict_margin": float(provider["bernstein_strict_margin"]),
        "bernstein_maximum_depth": int(provider["bernstein_maximum_depth"]),
    }


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
    source_paths = {
        "native_offset_source_sha256": (
            repository / "src/aorta_drive/candidates/native_offset.py"
        ),
        "h4_interface_source_sha256": repository / "src/aorta_drive/evaluation/h4.py",
    }
    for name, path in source_paths.items():
        if _sha256(path) != config["implementation"][name]:
            raise ValueError(f"{name} disagrees with registration")

    anchors = _selected_anchors(args.time_set, args.role_manifest)
    if len(anchors) != 128:
        raise ValueError("fit supply requires exactly eight groups times sixteen anchors")
    metrics = _metric_records(json.loads(args.metric_manifest.read_text(encoding="utf-8")))
    limits = _limits(config)
    provider_kwargs = _provider_kwargs(config)
    interval_s = float(config["native"]["interval_s"])
    registered_count = len(provider_kwargs["amplitudes_m"])

    rows: list[dict[str, Any]] = []
    group_supplied: Counter[str] = Counter()
    provider_rejections: Counter[str] = Counter()
    invariant_failures: list[str] = []
    total_accepted = 0
    both_sign_supplied = 0
    for index, anchor in enumerate(anchors, start=1):
        key = (anchor["log_name"], anchor["scene_token"])
        record = metrics.get(key)
        if record is None:
            raise ValueError("metric manifest lacks a registered anchor")
        cache_path = args.metric_root / str(record["relative_cache_path"])
        if _sha256(cache_path) != record["cache_sha256"]:
            raise ValueError("metric cache raw hash mismatch")
        with lzma.open(cache_path, "rb") as stream:
            cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM input
        native = sample_cached_native_trajectory(cache)
        batch = generate_native_offset_candidates(native.poses, **provider_kwargs)
        provider_rejections.update(batch.rejection_counts)
        total_accepted += len(batch.candidates)
        eligible = 0
        eligible_signs: set[float] = set()
        final_exact_count = 0
        candidate_rows: list[dict[str, Any]] = []
        for candidate in batch.candidates:
            challenger = np.vstack((np.zeros((1, 3)), candidate.poses))
            challenger_speed = terminal_segment_average_speed(
                challenger, interval_s=interval_s
            )
            outcome = qualify_intent_match(
                challenger,
                native.poses,
                challenger_terminal_speed_mps=challenger_speed,
                native_terminal_speed_mps=native.terminal_segment_average_speed_mps,
                limits=limits,
            )
            final_exact = bool(np.array_equal(challenger[7:], native.poses[7:]))
            final_exact_count += int(final_exact)
            eligible += int(outcome.eligible)
            if outcome.eligible:
                eligible_signs.add(math.copysign(1.0, candidate.control.amplitude_m))
            endpoint_exact = (
                outcome.progress_ratio == 1.0
                and outcome.terminal_cross_track_m == 0.0
                and outcome.terminal_heading_difference_rad == 0.0
                and outcome.terminal_speed_difference_mps == 0.0
                and final_exact
            )
            if not endpoint_exact:
                invariant_failures.append(
                    f"{anchor['recording_group']}/{anchor['anchor_index']}/"
                    f"{candidate.control.provenance}"
                )
            candidate_rows.append(
                {
                    "control": candidate.control.provenance,
                    "amplitude_m": candidate.control.amplitude_m,
                    "intent_eligible": outcome.eligible,
                    "intent_reason": outcome.reason,
                    "endpoint_invariant_exact": endpoint_exact,
                    "bernstein_subdivision_count": (
                        candidate.bernstein_subdivision_count
                    ),
                }
            )
        supplied = eligible > 0
        both_sign = eligible_signs == {-1.0, 1.0}
        group_supplied[anchor["recording_group"]] += int(supplied)
        both_sign_supplied += int(both_sign)
        rows.append(
            {
                "recording_group": anchor["recording_group"],
                "anchor_index": int(anchor["anchor_index"]),
                "log_name": anchor["log_name"],
                "scene_token": anchor["scene_token"],
                "native_terminal_displacement_m": float(
                    np.linalg.norm(native.poses[-1, :2])
                ),
                "native_sample_sha256": native.canonical_sample_sha256,
                "registered_candidate_count": registered_count,
                "accepted_candidate_count": len(batch.candidates),
                "eligible_candidate_count": eligible,
                "final_exact_candidate_count": final_exact_count,
                "has_eligible_candidate": supplied,
                "has_both_signed_eligible_candidates": both_sign,
                "provider_rejection_counts": batch.rejection_counts,
                "candidates": candidate_rows,
            }
        )
        if index % 16 == 0:
            print(json.dumps({"anchors_completed": index}), flush=True)

    supplied_count = sum(int(row["has_eligible_candidate"]) for row in rows)
    group_fractions = {
        group: group_supplied[group] / 16.0 for group in sorted(group_supplied)
    }
    supply_fraction = supplied_count / len(rows)
    minimum_group_fraction = min(group_fractions.values())
    criteria = config["decision"]["adequate_supply_requires"]
    adequate = (
        supply_fraction >= float(criteria["minimum_anchor_supply_fraction"])
        and minimum_group_fraction
        >= float(criteria["minimum_each_group_supply_fraction"])
        and len(invariant_failures)
        == int(criteria["exact_endpoint_invariant_failure_count"])
    )
    if invariant_failures:
        decision = "STOP_ENDPOINT_INVARIANT_IMPLEMENTATION_FAILURE"
    elif adequate:
        decision = "PASS_TO_REGISTER_FIT_GEOMETRY_PREFLIGHT"
    else:
        decision = "STOP_CANDIDATE_SUPPLY_COVERAGE"
    return {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": decision,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": _sha256(args.config),
        "elapsed_seconds": time.perf_counter() - started,
        "data_scope": {
            "role": "fit",
            "split": "train",
            "recording_group_count": 8,
            "anchor_count": 128,
            "geometry_teacher_prediction_or_image_read": False,
            "development_calibration_test_or_target_accessed": False,
        },
        "supply": {
            "anchors_with_eligible_candidate": supplied_count,
            "anchor_supply_fraction": supply_fraction,
            "group_anchor_supply_fractions": group_fractions,
            "minimum_group_anchor_supply_fraction": minimum_group_fraction,
            "anchors_with_both_signed_eligible_candidates": both_sign_supplied,
            "both_signed_anchor_supply_fraction": both_sign_supplied / len(rows),
        },
        "provider": {
            "registered_constructions": registered_count * len(rows),
            "accepted_constructions": total_accepted,
            "continuous_dynamic_acceptance_fraction": (
                total_accepted / (registered_count * len(rows))
            ),
            "rejection_counts": dict(sorted(provider_rejections.items())),
        },
        "endpoint_invariant_failures": invariant_failures,
        "anchor_ledger": rows,
        "claims": {
            "candidate_supply_adequate_on_registered_fit_sample": adequate,
            "geometry_evaluated": False,
            "teacher_or_prediction_read": False,
            "r004h4_registered": False,
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
