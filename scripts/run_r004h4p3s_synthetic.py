"""Run the registered synthetic qualification for the native-offset provider."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.evaluation.h4 import (
    IntentEligibilityLimits,
    qualify_intent_match,
    terminal_segment_average_speed,
)


TIMES = np.arange(9, dtype=np.float64) * 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _straight_constant_speed() -> npt.NDArray[np.float64]:
    return np.column_stack((5.0 * TIMES, np.zeros(9), np.zeros(9)))


def _circle(sign: float) -> npt.NDArray[np.float64]:
    radius = 50.0
    arc = 5.0 * TIMES
    angle = sign * arc / radius
    return np.column_stack(
        (
            radius * np.sin(arc / radius),
            sign * radius * (1.0 - np.cos(arc / radius)),
            angle,
        )
    )


def _straight_acceleration(acceleration: float) -> npt.NDArray[np.float64]:
    speed0 = 5.0 if acceleration >= 0.0 else 7.0
    distance = speed0 * TIMES + 0.5 * acceleration * TIMES**2
    return np.column_stack((distance, np.zeros(9), np.zeros(9)))


def _smooth_s_curve() -> npt.NDArray[np.float64]:
    x = 5.0 * TIMES
    y = 0.6 * np.sin(math.pi * TIMES / 4.0)
    dx = np.full_like(TIMES, 5.0)
    dy = 0.6 * math.pi / 4.0 * np.cos(math.pi * TIMES / 4.0)
    heading = np.arctan2(dy, dx)
    poses = np.column_stack((x, y, heading))
    poses[0] = 0.0
    return poses


def _cases() -> dict[str, Callable[[], npt.NDArray[np.float64]]]:
    return {
        "straight_constant_speed": _straight_constant_speed,
        "left_constant_curvature": lambda: _circle(1.0),
        "right_constant_curvature": lambda: _circle(-1.0),
        "straight_acceleration": lambda: _straight_acceleration(0.5),
        "straight_deceleration": lambda: _straight_acceleration(-0.5),
        "smooth_s_curve": _smooth_s_curve,
    }


def _limits(config: dict[str, Any]) -> IntentEligibilityLimits:
    values = config["intent_eligibility"]
    return IntentEligibilityLimits(
        minimum_native_terminal_displacement_m=float(
            values["minimum_native_terminal_displacement_m"]
        ),
        minimum_progress_ratio=float(values["progress_ratio"][0]),
        maximum_progress_ratio=float(values["progress_ratio"][1]),
        maximum_terminal_cross_track_m=float(values["maximum_terminal_cross_track_m"]),
        maximum_terminal_heading_difference_rad=float(
            values["maximum_terminal_heading_difference_rad"]
        ),
        maximum_terminal_speed_difference_mps=float(
            values["maximum_terminal_speed_difference_mps"]
        ),
    )


def _provider_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    provider = config["provider"]
    envelope = config["physical_envelope"]
    return {
        "interval_s": float(provider["interval_s"]),
        "bump_end_s": float(provider["bump_end_s"]),
        "amplitudes_m": tuple(
            float(value) for value in config["design"]["positive_control_amplitudes_m"]
        ),
        "maximum_speed_mps": float(envelope["maximum_speed_mps"]),
        "maximum_longitudinal_acceleration_mps2": float(
            envelope["maximum_longitudinal_acceleration_mps2"]
        ),
        "maximum_lateral_acceleration_mps2": float(
            envelope["maximum_lateral_acceleration_mps2"]
        ),
        "maximum_total_acceleration_mps2": float(
            envelope["maximum_total_acceleration_mps2"]
        ),
        "bernstein_strict_margin": float(envelope["bernstein_strict_margin"]),
        "bernstein_maximum_depth": int(envelope["bernstein_maximum_depth"]),
    }


def run(
    config_path: Path, implementation_revision: str, repository_test_log: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    limits = _limits(config)
    provider_kwargs = _provider_kwargs(config)
    test_text = repository_test_log.read_text(encoding="utf-8")
    test_matches = [int(token) for token in re.findall(r"(\d+) passed", test_text)]
    minimum_test_count = int(
        config["decision"]["pass_requires"]["minimum_repository_test_count"]
    )
    repository_tests_pass = (
        bool(test_matches)
        and max(test_matches) >= minimum_test_count
        and " failed" not in test_text
        and " error" not in test_text.lower()
    )
    interval_s = float(config["provider"]["interval_s"])
    rows: list[dict[str, Any]] = []
    invariant_failures: list[str] = []
    both_signs_by_case: dict[str, bool] = {}
    deterministic = True
    for case_name, factory in _cases().items():
        native = factory()
        first = generate_native_offset_candidates(native, **provider_kwargs)
        second = generate_native_offset_candidates(native, **provider_kwargs)
        deterministic = deterministic and len(first.candidates) == len(second.candidates)
        deterministic = deterministic and all(
            left.control.provenance == right.control.provenance
            and np.array_equal(left.poses, right.poses)
            for left, right in zip(first.candidates, second.candidates, strict=True)
        )
        signs = {math.copysign(1.0, item.control.amplitude_m) for item in first.candidates}
        both_signs_by_case[case_name] = signs == {-1.0, 1.0}
        for candidate in first.candidates:
            challenger = np.vstack((np.zeros((1, 3)), candidate.poses))
            native_speed = terminal_segment_average_speed(native, interval_s=interval_s)
            challenger_speed = terminal_segment_average_speed(
                challenger, interval_s=interval_s
            )
            intent = qualify_intent_match(
                challenger,
                native,
                challenger_terminal_speed_mps=challenger_speed,
                native_terminal_speed_mps=native_speed,
                limits=limits,
            )
            final_exact = bool(np.array_equal(challenger[7:], native[7:]))
            middle_displacement = float(
                np.max(np.linalg.norm(challenger[1:7, :2] - native[1:7, :2], axis=1))
            )
            if not intent.eligible:
                invariant_failures.append(f"{case_name}/{candidate.control.provenance}: intent")
            if not final_exact:
                invariant_failures.append(f"{case_name}/{candidate.control.provenance}: final")
            if middle_displacement < 0.1:
                invariant_failures.append(f"{case_name}/{candidate.control.provenance}: middle")
            rows.append(
                {
                    "case": case_name,
                    "control": candidate.control.provenance,
                    "amplitude_m": candidate.control.amplitude_m,
                    "intent_eligible": intent.eligible,
                    "progress_ratio": intent.progress_ratio,
                    "terminal_cross_track_m": intent.terminal_cross_track_m,
                    "terminal_heading_difference_rad": (
                        intent.terminal_heading_difference_rad
                    ),
                    "terminal_speed_difference_mps": intent.terminal_speed_difference_mps,
                    "final_segment_bit_exact": final_exact,
                    "maximum_middle_displacement_m": middle_displacement,
                    "bernstein_subdivision_count": candidate.bernstein_subdivision_count,
                }
            )

    negative = config["design"]["negative_controls"]
    excessive = generate_native_offset_candidates(
        _straight_constant_speed(),
        **{
            **provider_kwargs,
            "amplitudes_m": (float(negative["excessive_amplitude_m"]),),
        },
    )
    insufficient_terminal = float(negative["insufficient_native_terminal_displacement_m"])
    insufficient = np.column_stack(
        (
            insufficient_terminal / 4.0 * TIMES,
            np.zeros(9),
            np.zeros(9),
        )
    )
    insufficient_batch = generate_native_offset_candidates(
        insufficient, **provider_kwargs
    )
    checks = {
        "every_analytic_case_has_both_signed_accepted_candidates": all(
            both_signs_by_case.values()
        ),
        "every_accepted_candidate_is_intent_eligible": all(
            bool(row["intent_eligible"]) for row in rows
        ),
        "every_accepted_candidate_has_bit_exact_final_segment": all(
            bool(row["final_segment_bit_exact"]) for row in rows
        ),
        "every_accepted_candidate_has_nontrivial_mid_horizon_displacement": all(
            float(row["maximum_middle_displacement_m"]) >= 0.1 for row in rows
        ),
        "deterministic_replay_is_bit_exact": deterministic,
        "excessive_amplitude_is_rejected": (
            len(excessive.candidates) == 0
            and excessive.rejection_counts["continuous_dynamic_envelope"] == 1
        ),
        "insufficient_displacement_is_rejected": (
            len(insufficient_batch.candidates) == 0
            and insufficient_batch.rejection_counts["native_reference_displacement"]
            == len(provider_kwargs["amplitudes_m"])
        ),
        "full_repository_tests_pass": repository_tests_pass,
    }
    passed = bool(rows) and all(checks.values()) and not invariant_failures
    return {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": (
            "PASS_TO_REGISTER_R004H4P3_FIT_SUPPLY"
            if passed
            else "STOP_SYNTHETIC_PROVIDER_QUALIFICATION"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": implementation_revision,
        "config_sha256": _sha256(config_path),
        "elapsed_seconds": time.perf_counter() - started,
        "repository_test_log_sha256": _sha256(repository_test_log),
        "repository_test_count": max(test_matches) if test_matches else 0,
        "design": {
            "analytic_case_count": len(_cases()),
            "registered_amplitude_count": len(provider_kwargs["amplitudes_m"]),
            "accepted_candidate_count": len(rows),
            "statistical_inference_authorized": False,
            "fit_development_calibration_test_or_target_accessed": False,
        },
        "checks": checks,
        "both_signs_by_case": both_signs_by_case,
        "invariant_failures": invariant_failures,
        "candidate_ledger": rows,
        "negative_controls": {
            "excessive_amplitude_rejections": excessive.rejection_counts,
            "insufficient_displacement_rejections": insufficient_batch.rejection_counts,
        },
        "claims": {
            "synthetic_provider_interface_qualified": passed,
            "continuous_dynamics_screen_is_formal_real_arithmetic_proof": False,
            "fit_data_read": False,
            "geometry_teacher_or_prediction_read": False,
            "candidate_supply_estimated": False,
            "development_access_authorized": False,
            "h4_registered": False,
            "coverage_established": False,
            "gate_b_passed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--repository-test-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.implementation_revision, args.repository_test_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "candidate_ledger"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if result["decision"] != "PASS_TO_REGISTER_R004H4P3_FIT_SUPPLY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
