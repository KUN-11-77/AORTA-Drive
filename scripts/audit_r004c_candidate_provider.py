"""Audit the deterministic R004C candidate provider on source metadata only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aorta_drive.candidates import generate_kinematic_candidates

ALLOWED_SPLITS = {"train", "development"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    requested_splits = {part.strip() for part in args.splits.split(",") if part.strip()}
    if not requested_splits or not requested_splits <= ALLOWED_SPLITS:
        raise ValueError("R004C audit may use only train and development splits")
    with args.time_set.open(encoding="utf-8", newline="") as stream:
        anchors = [row for row in csv.DictReader(stream) if row["split"] in requested_splits]
    anchors.sort(key=lambda row: (row["recording_group"], int(row["anchor_index"])))
    if args.max_anchors is not None:
        if args.max_anchors <= 0:
            raise ValueError("max_anchors must be positive")
        anchors = anchors[: args.max_anchors]
    if not anchors:
        raise ValueError("no anchors selected")

    by_log: dict[str, list[dict[str, str]]] = defaultdict(list)
    for anchor in anchors:
        by_log[anchor["log_name"]].append(anchor)

    speeds: list[float] = []
    candidate_counts: list[float] = []
    roi_candidate_counts: list[float] = []
    roi_candidate_fractions: list[float] = []
    endpoint_spreads: list[float] = []
    rejection_counts: Counter[str] = Counter()
    control_frequency: Counter[str] = Counter()
    missing_tokens: list[str] = []
    per_anchor: list[dict[str, object]] = []

    for log_number, log_name in enumerate(sorted(by_log), start=1):
        log_path = args.log_root / f"{log_name}.pkl"
        if not log_path.is_file():
            raise FileNotFoundError(log_path)
        with log_path.open("rb") as stream:
            frames = pickle.load(stream)  # noqa: S301 - official frozen NAVSIM input
        token_to_frame = {str(frame["token"]): frame for frame in frames}
        for anchor in by_log[log_name]:
            token = anchor["scene_token"]
            frame = token_to_frame.get(token)
            if frame is None:
                missing_tokens.append(token)
                continue
            dynamic_state = np.asarray(frame["ego_dynamic_state"], dtype=np.float64)
            if dynamic_state.ndim != 1 or dynamic_state.size < 2:
                raise ValueError(f"invalid ego_dynamic_state for {token}")
            speed = float(np.linalg.norm(dynamic_state[:2]))
            batch = generate_kinematic_candidates(speed)
            replay = generate_kinematic_candidates(speed)
            if len(batch.candidates) != len(replay.candidates):
                raise AssertionError(f"non-deterministic candidate count for {token}")
            for first, second in zip(batch.candidates, replay.candidates, strict=True):
                if first.control != second.control or not np.array_equal(first.poses, second.poses):
                    raise AssertionError(f"non-deterministic candidate replay for {token}")
            if not batch.candidates:
                raise RuntimeError(f"empty candidate library for {token}")

            roi_eligible = 0
            endpoints = []
            for candidate in batch.candidates:
                if candidate.poses.shape != (8, 3) or not np.all(np.isfinite(candidate.poses)):
                    raise AssertionError(f"invalid trajectory payload for {token}")
                maximum_speed = max(speed, candidate.terminal_speed_mps)
                lateral_acceleration = (
                    maximum_speed**2 * abs(candidate.control.curvature_per_m)
                )
                if (
                    abs(candidate.control.acceleration_mps2) > 3.0 + 1e-12
                    or maximum_speed > 20.0 + 1e-12
                    or lateral_acceleration > 4.0 + 1e-12
                ):
                    raise AssertionError(f"dynamic envelope violation for {token}")
                reference_inside_roi = bool(
                    np.all(np.abs(candidate.poses[:, 0]) <= 32.0)
                    and np.all(np.abs(candidate.poses[:, 1]) <= 32.0)
                )
                roi_eligible += int(reference_inside_roi)
                endpoints.append(candidate.poses[-1, :2])
                control_frequency[candidate.control.provenance] += 1

            endpoint_array = np.asarray(endpoints, dtype=np.float64)
            pairwise = endpoint_array[:, None, :] - endpoint_array[None, :, :]
            endpoint_spread = float(np.max(np.linalg.norm(pairwise, axis=2)))
            speeds.append(speed)
            candidate_counts.append(float(len(batch.candidates)))
            roi_candidate_counts.append(float(roi_eligible))
            roi_candidate_fractions.append(roi_eligible / len(batch.candidates))
            endpoint_spreads.append(endpoint_spread)
            rejection_counts.update(batch.rejection_counts)
            per_anchor.append(
                {
                    "scene_token": token,
                    "recording_group": anchor["recording_group"],
                    "split": anchor["split"],
                    "initial_speed_mps": speed,
                    "candidate_count": len(batch.candidates),
                    "reference_point_roi_eligible_count": roi_eligible,
                    "endpoint_spread_m": endpoint_spread,
                    "rejection_counts": batch.rejection_counts,
                }
            )
        print(f"audited log {log_number}/{len(by_log)}: {log_name}", flush=True)

    if missing_tokens:
        raise RuntimeError(f"missing {len(missing_tokens)} frozen anchor tokens")
    if len(per_anchor) != len(anchors):
        raise AssertionError(f"audited {len(per_anchor)} anchors, expected {len(anchors)}")
    if any(count <= 0.0 for count in roi_candidate_counts):
        raise RuntimeError("at least one anchor has no reference-point ROI candidate")
    result: dict[str, object] = {
        "experiment_id": "R004C-candidate-provider-audit",
        "implementation_revision": args.implementation_revision,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "source train/development metadata only; no maps, labels, images, "
            "planner outputs, or scores"
        ),
        "splits": sorted(requested_splits),
        "anchor_count": len(anchors),
        "recording_group_count": len({row["recording_group"] for row in anchors}),
        "log_count": len(by_log),
        "provider": {
            "type": "deterministic_constant_acceleration_curvature_v1",
            "input": "norm of current ego_dynamic_state velocity components",
            "native_included_separately": True,
            "horizon_s": 4.0,
            "interval_s": 0.5,
            "pose_count": 8,
            "accelerations_mps2": [0.0, -1.0, -2.0, -3.0, 1.0],
            "curvatures_per_m": [0.0, -0.01, 0.01, -0.02, 0.02, -0.04, 0.04],
            "maximum_raw_control_count": 35,
            "maximum_speed_mps": 20.0,
            "maximum_longitudinal_acceleration_mps2": 3.0,
            "maximum_lateral_acceleration_mps2": 4.0,
            "reverse_motion": False,
            "randomness": False,
        },
        "initial_speed_mps": _quantiles(speeds),
        "candidate_count": _quantiles(candidate_counts),
        "reference_point_roi_eligible_count": _quantiles(roi_candidate_counts),
        "anchors_with_reference_point_roi_candidate": int(
            np.sum(np.asarray(roi_candidate_counts) > 0.0)
        ),
        "endpoint_spread_m": _quantiles(endpoint_spreads),
        "reference_point_roi_eligible_fraction": _quantiles(roi_candidate_fractions),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "control_acceptance_anchor_counts": dict(sorted(control_frequency.items())),
        "interpretation_limits": {
            "reference_point_roi_is_geometry_certificate": False,
            "candidate_is_better_than_native_evaluated": False,
            "meaningful_non_inclusion_evaluated": False,
            "planner_transfer_evaluated": False,
        },
        "input_hashes": {
            "time_set_sha256": _sha256(args.time_set),
            "audit_script_sha256": _sha256(Path(__file__)),
            "provider_source_sha256": _sha256(
                Path(__file__).parents[1]
                / "src"
                / "aorta_drive"
                / "candidates"
                / "kinematic.py"
            ),
        },
        "per_anchor": per_anchor,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--splits", default="train,development")
    parser.add_argument("--max-anchors", type=int)
    args = parser.parse_args()
    result = audit(args)
    summary = {key: value for key, value in result.items() if key != "per_anchor"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
