"""Diagnose frozen candidate/native intent coverage on eight fit groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import math
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_kinematic_candidates
from aorta_drive.evaluation.h4 import (
    IntentEligibilityLimits,
    qualify_intent_match,
    sample_cached_native_trajectory,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _selected_anchors(time_set: Path, roles: Path) -> list[dict[str, str]]:
    fit_groups = [row["recording_group"] for row in _rows(roles) if row["role"] == "fit"][:8]
    if len(fit_groups) != 8 or len(set(fit_groups)) != 8:
        raise ValueError("role manifest does not provide eight unique leading fit groups")
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _rows(time_set):
        if row["recording_group"] in fit_groups:
            by_group[row["recording_group"]].append(row)
    selected: list[dict[str, str]] = []
    for group in fit_groups:
        group_rows = sorted(by_group[group], key=lambda row: int(row["anchor_index"]))
        if (
            len(group_rows) != 16
            or [int(row["anchor_index"]) for row in group_rows] != list(range(16))
            or any(row["split"] != "train" for row in group_rows)
        ):
            raise ValueError("each selected fit group must contain train anchor indices 0..15")
        selected.extend(group_rows)
    return selected


def _metric_records(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("metric manifest lacks a records list")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for value in records:
        if not isinstance(value, dict):
            raise ValueError("metric manifest contains a non-object record")
        key = (str(value.get("log_name")), str(value.get("scene_token")))
        if key in result:
            raise ValueError("metric manifest contains a duplicate scene key")
        result[key] = value
    return result


def _raw_speeds(log_root: Path, anchors: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    by_log: dict[str, list[str]] = defaultdict(list)
    for anchor in anchors:
        by_log[anchor["log_name"]].append(anchor["scene_token"])
    result: dict[tuple[str, str], float] = {}
    for log_name, tokens in sorted(by_log.items()):
        with (log_root / f"{log_name}.pkl").open("rb") as stream:
            frames = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM input
        requested = set(tokens)
        for frame in frames:
            token = str(frame["token"])
            if token not in requested:
                continue
            dynamic = np.asarray(frame["ego_dynamic_state"], dtype=np.float64)
            if dynamic.ndim != 1 or dynamic.size < 2 or not np.all(np.isfinite(dynamic[:2])):
                raise ValueError("raw ego dynamic state is invalid")
            result[(log_name, token)] = float(np.linalg.norm(dynamic[:2]))
    expected = {(row["log_name"], row["scene_token"]) for row in anchors}
    if set(result) != expected:
        raise ValueError("raw logs do not exactly cover the registered anchors")
    return result


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


def _interval_gap(value: float, low: float, high: float) -> float:
    return max(low - value, value - high, 0.0)


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
    anchors = _selected_anchors(args.time_set, args.role_manifest)
    if len(anchors) != 128:
        raise ValueError("diagnosis requires exactly eight groups times sixteen anchors")
    metrics = _metric_records(json.loads(args.metric_manifest.read_text(encoding="utf-8")))
    speeds = _raw_speeds(args.log_root, anchors)
    limits = _limits(config)

    anchor_rows: list[dict[str, Any]] = []
    first_failures: Counter[str] = Counter()
    independent_passes: Counter[str] = Counter()
    group_supply: Counter[str] = Counter(
        {anchor["recording_group"]: 0 for anchor in anchors}
    )
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
        initial_speed = speeds[key]
        batch = generate_kinematic_candidates(initial_speed)
        eligible = 0
        candidate_metrics: list[dict[str, Any]] = []
        for candidate in batch.candidates:
            candidate_poses = np.vstack((np.zeros((1, 3)), candidate.poses))
            outcome = qualify_intent_match(
                candidate_poses,
                native.poses,
                challenger_terminal_speed_mps=candidate.terminal_speed_mps,
                native_terminal_speed_mps=native.terminal_speed_mps,
                limits=limits,
            )
            progress_pass = (
                math.isfinite(outcome.progress_ratio)
                and limits.minimum_progress_ratio
                <= outcome.progress_ratio
                <= limits.maximum_progress_ratio
            )
            cross_track_pass = (
                math.isfinite(outcome.terminal_cross_track_m)
                and outcome.terminal_cross_track_m <= limits.maximum_terminal_cross_track_m
            )
            heading_pass = (
                math.isfinite(outcome.terminal_heading_difference_rad)
                and outcome.terminal_heading_difference_rad
                <= limits.maximum_terminal_heading_difference_rad
            )
            speed_pass = (
                math.isfinite(outcome.terminal_speed_difference_mps)
                and outcome.terminal_speed_difference_mps
                <= limits.maximum_terminal_speed_difference_mps
            )
            for gate, passed in (
                ("progress", progress_pass),
                ("cross_track", cross_track_pass),
                ("heading", heading_pass),
                ("speed", speed_pass),
            ):
                independent_passes[gate] += int(passed)
            first_failures[outcome.reason] += 1
            eligible += int(outcome.eligible)
            progress_gap = (
                _interval_gap(
                    outcome.progress_ratio,
                    limits.minimum_progress_ratio,
                    limits.maximum_progress_ratio,
                )
                if math.isfinite(outcome.progress_ratio)
                else math.inf
            )
            candidate_metrics.append(
                {
                    "control": candidate.control.provenance,
                    "eligible": outcome.eligible,
                    "first_failure": outcome.reason,
                    "progress_ratio": (
                        outcome.progress_ratio
                        if math.isfinite(outcome.progress_ratio)
                        else None
                    ),
                    "progress_gap": progress_gap if math.isfinite(progress_gap) else None,
                    "cross_track_m": (
                        outcome.terminal_cross_track_m
                        if math.isfinite(outcome.terminal_cross_track_m)
                        else None
                    ),
                    "heading_difference_rad": (
                        outcome.terminal_heading_difference_rad
                        if math.isfinite(outcome.terminal_heading_difference_rad)
                        else None
                    ),
                    "speed_difference_mps": (
                        outcome.terminal_speed_difference_mps
                        if math.isfinite(outcome.terminal_speed_difference_mps)
                        else None
                    ),
                    "_progress_gap_sort": progress_gap,
                }
            )
        supplied = eligible > 0
        group_supply[anchor["recording_group"]] += int(supplied)
        nearest_internal = min(
            candidate_metrics, key=lambda row: float(row["_progress_gap_sort"])
        )
        nearest = {
            key: value
            for key, value in nearest_internal.items()
            if key != "_progress_gap_sort"
        }
        anchor_rows.append(
            {
                "recording_group": anchor["recording_group"],
                "anchor_index": int(anchor["anchor_index"]),
                "log_name": anchor["log_name"],
                "scene_token": anchor["scene_token"],
                "initial_speed_mps": initial_speed,
                "native_terminal_displacement_m": float(
                    np.linalg.norm(native.poses[-1, :2])
                ),
                "native_terminal_speed_mps": native.terminal_speed_mps,
                "native_transform_error_m": native.maximum_transform_error_m,
                "raw_candidate_count": len(batch.candidates),
                "eligible_candidate_count": eligible,
                "has_eligible_candidate": supplied,
                "nearest_progress_candidate": nearest,
            }
        )
        if index % 16 == 0:
            print(json.dumps({"anchors_completed": index}), flush=True)

    supplied_anchors = sum(bool(row["has_eligible_candidate"]) for row in anchor_rows)
    anchor_supply_fraction = supplied_anchors / len(anchor_rows)
    group_fractions = {group: count / 16.0 for group, count in group_supply.items()}
    minimum_group_fraction = min(group_fractions.values())
    raw_candidates = sum(int(row["raw_candidate_count"]) for row in anchor_rows)
    criteria = config["decision"]["adequate_supply_requires"]
    adequate = (
        anchor_supply_fraction >= float(criteria["minimum_anchor_supply_fraction"])
        and minimum_group_fraction >= float(criteria["minimum_each_group_supply_fraction"])
    )
    result: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": (
            "PASS_TO_FIT_GEOMETRY_PREFLIGHT"
            if adequate
            else "STOP_CANDIDATE_SUPPLY_COVERAGE"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": _sha256(args.config),
        "elapsed_seconds": time.perf_counter() - started,
        "data_scope": {
            "role": "fit",
            "split": "train",
            "recording_group_count": 8,
            "anchor_count": 128,
            "development_calibration_test_or_target_accessed": False,
        },
        "supply": {
            "anchors_with_eligible_candidate": supplied_anchors,
            "anchor_supply_fraction": anchor_supply_fraction,
            "group_anchor_supply_fractions": group_fractions,
            "minimum_group_anchor_supply_fraction": minimum_group_fraction,
        },
        "candidate_diagnostics": {
            "raw_candidate_evaluations": raw_candidates,
            "first_failure_counts": dict(sorted(first_failures.items())),
            "independent_gate_pass_fractions": {
                gate: count / raw_candidates for gate, count in sorted(independent_passes.items())
            },
        },
        "anchor_ledger": anchor_rows,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()
    result = run(args)
    summary = {key: value for key, value in result.items() if key != "anchor_ledger"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
