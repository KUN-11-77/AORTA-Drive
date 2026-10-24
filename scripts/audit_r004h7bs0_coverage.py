"""Independent mechanical audit of the R004H7-BS0 coverage attribution."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _corners(poses: np.ndarray[Any, np.dtype[np.float64]], vehicle: Mapping[str, Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
    yaw = poses[:, 2]
    forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
    left = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    offsets = (
        (float(vehicle["front_extent_m"]), float(vehicle["half_width_m"])),
        (-float(vehicle["rear_overhang_m"]), float(vehicle["half_width_m"])),
        (-float(vehicle["rear_overhang_m"]), -float(vehicle["half_width_m"])),
        (float(vehicle["front_extent_m"]), -float(vehicle["half_width_m"])),
    )
    return np.stack(
        [poses[:, :2] + longitudinal * forward + lateral * left for longitudinal, lateral in offsets],
        axis=1,
    )


def _assert_close(actual: object, expected: object, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AssertionError(f"mapping mismatch at {path}")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"list mismatch at {path}")
        for index, value in enumerate(expected):
            _assert_close(actual[index], value, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if actual is None or not math.isclose(float(cast(float, actual)), expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise AssertionError(f"float mismatch at {path}: {actual} != {expected}")
        return
    if actual != expected:
        raise AssertionError(f"value mismatch at {path}: {actual} != {expected}")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.result.read_text(encoding="utf-8"))
    attribution = json.loads(args.attribution.read_text(encoding="utf-8"))
    if sha256_file(args.result) != attribution.get("source_result_sha256"):
        raise ValueError("attribution is not bound to the supplied H7-B result")
    if source.get("decision") != "INSUFFICIENT_PANEL_COVERAGE":
        raise ValueError("source result is not the frozen H7-B coverage failure")
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    panel = config["H7B"]["candidate_panel"]
    grid = panel["node_grid"]
    vehicle = panel["vehicle"]
    x_min = float(grid["x_min_m"])
    y_min = float(grid["y_min_m"])
    x_max = x_min + int(grid["columns"]) * float(grid["resolution_m"])
    y_max = y_min + int(grid["rows"]) * float(grid["resolution_m"])

    panel_reasons: Counter[str] = Counter()
    candidate_reasons: Counter[str] = Counter()
    native_boundaries: Counter[str] = Counter()
    candidate_boundaries: Counter[str] = Counter()
    native_indices: list[float] = []
    native_expansion: dict[str, list[float]] = defaultdict(list)
    candidate_expansion: dict[str, list[float]] = defaultdict(list)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    maximum_translation = 0.0
    maximum_yaw = 0.0
    artifact_failures = 0
    invalid_keys: set[tuple[str, int]] = set()

    ledgers = source.get("anchor_ledgers")
    if not isinstance(ledgers, list) or len(ledgers) != 384:
        raise ValueError("source anchor ledger is incomplete")
    for stored in ledgers:
        row = cast(dict[str, Any], stored)
        artifact = args.run_directory / str(row["panel_artifact"])
        if not artifact.is_file() or sha256_file(artifact) != row["panel_artifact_sha256"]:
            artifact_failures += 1
            continue
        with np.load(artifact, allow_pickle=False) as archive:
            poses = np.asarray(archive["realized_local_poses"], dtype=np.float64)
            eligible = np.asarray(archive["challenger_eligible"], dtype=np.bool_)
        stored_candidates = cast(list[dict[str, Any]], row["candidates"])
        if poses.shape != (7, 41, 3) or eligible.shape != (6,) or len(stored_candidates) != 7:
            raise AssertionError("panel artifact shape mismatch")
        for index, trajectory in enumerate(poses):
            maximum_translation = max(maximum_translation, float(np.linalg.norm(trajectory[0, :2])))
            maximum_yaw = max(
                maximum_yaw,
                abs(math.atan2(math.sin(float(trajectory[0, 2])), math.cos(float(trajectory[0, 2])))),
            )
            corners = _corners(trajectory, vehicle)
            margins = np.column_stack(
                (
                    np.min(corners[:, :, 0], axis=1) - x_min,
                    x_max - np.max(corners[:, :, 0], axis=1),
                    np.min(corners[:, :, 1], axis=1) - y_min,
                    y_max - np.max(corners[:, :, 1], axis=1),
                )
            )
            first_indices = np.flatnonzero(np.any(margins < 0.0, axis=1))
            first = int(first_indices[0]) if first_indices.size else None
            names = ("x_min", "x_max", "y_min", "y_max")
            boundary = (
                "+".join(name for idx, name in enumerate(names) if margins[first, idx] < 0.0)
                if first is not None
                else ""
            )
            expansion = {
                name: max(0.0, -float(np.min(margins[:, idx])))
                for idx, name in enumerate(names)
            }
            is_eligible = index == 0 or bool(eligible[index - 1])
            expected_reason = (
                "intent_ineligible"
                if not is_eligible
                else "oriented footprint leaves the frozen ROI"
                if first is not None
                else "valid"
            )
            if stored_candidates[index]["reason"] != expected_reason:
                raise AssertionError("independent candidate reason mismatch")
            candidate_reasons[expected_reason] += 1
            if expected_reason == "oriented footprint leaves the frozen ROI":
                candidate_boundaries[boundary] += 1
                for name, value in expansion.items():
                    candidate_expansion[name].append(value)
                if index == 0:
                    native_boundaries[boundary] += 1
                    native_indices.append(float(cast(int, first)))
                    for name, value in expansion.items():
                        native_expansion[name].append(value)
        reason = str(row["panel_reason"])
        panel_reasons[reason] += 1
        group_counts[str(row["recording_group"])][reason] += 1
        if reason != "valid":
            invalid_keys.add((str(row["recording_group"]), int(row["anchor_index"])))

    if artifact_failures:
        raise AssertionError(f"{artifact_failures} panel artifacts failed provenance")
    recomputed = {
        "panel_reason_counts": dict(panel_reasons),
        "candidate_reason_counts": dict(candidate_reasons),
        "first_native_violation_boundary_counts": dict(native_boundaries),
        "first_candidate_violation_boundary_counts": dict(candidate_boundaries),
        "first_native_violation_index_quantiles": {
            "minimum": float(np.min(native_indices)),
            "q25": float(np.quantile(native_indices, 0.25)),
            "median": float(np.median(native_indices)),
            "q75": float(np.quantile(native_indices, 0.75)),
            "maximum": float(np.max(native_indices)),
        },
        "native_required_expansion_quantiles_m": {
            name: {
                "minimum": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "maximum": float(np.max(values)),
            }
            for name, values in sorted(native_expansion.items())
        },
        "candidate_required_expansion_quantiles_m": {
            name: {
                "minimum": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "maximum": float(np.max(values)),
            }
            for name, values in sorted(candidate_expansion.items())
        },
    }
    for key, expected in recomputed.items():
        _assert_close(attribution[key], expected, key)
    stored_invalid = {
        (str(row["recording_group"]), int(row["anchor_index"]))
        for row in attribution["invalid_anchor_ledger"]
    }
    if invalid_keys != stored_invalid or len(stored_invalid) != 119:
        raise AssertionError("invalid anchor ledger identity mismatch")
    expected_group_counts = {
        group: {
            "valid": counts["valid"],
            "native_invalid": counts["native_invalid"],
            "no_valid_challenger": counts["no_valid_challenger"],
            "invalid_fraction": (counts["native_invalid"] + counts["no_valid_challenger"]) / 16.0,
        }
        for group, counts in sorted(group_counts.items())
    }
    _assert_close(attribution["group_counts"], expected_group_counts, "group_counts")
    binding = attribution["coordinate_binding_audit"]
    _assert_close(binding["maximum_initial_translation_residual_m"], maximum_translation, "origin.translation")
    _assert_close(binding["maximum_initial_yaw_residual_rad"], maximum_yaw, "origin.yaw")
    if attribution["appearance_validity_contract"]["invalid_rows_with_conditions_none"] != 119:
        raise AssertionError("invalid panels unexpectedly contain appearance outcomes")
    return {
        "experiment_id": "R004H7BS0-mechanical-audit-v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_result_sha256": sha256_file(args.result),
        "attribution_result_sha256": sha256_file(args.attribution),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "panel_artifact_count": 384,
        "invalid_anchor_identity_count": 119,
        "artifact_failure_count": 0,
        "independent_attribution_failure_count": 0,
        "recomputed_panel_reason_counts": dict(panel_reasons),
        "external_review_status": "REVIEW_UNAVAILABLE",
        "claim_boundary": (
            "Mechanical attribution of post-hoc H7-B coverage failures only; no visual, "
            "decision, driving, safety, target, or novelty claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
