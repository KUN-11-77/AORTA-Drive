"""Post-hoc, outcome-blind attribution of the frozen R004H7-B coverage failure."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.shadow_decision import rasterize_discrete_footprint_nodes
from aorta_drive.evaluation.support_adequacy import (
    group_equal_descriptive_difference,
    support_profile,
)
from aorta_drive.vision.feature_cache import sha256_file

ORIGIN_TOLERANCE = 1.0e-9
GEOMETRY_KEYS = (
    "path_length_m",
    "terminal_x_m",
    "terminal_y_m",
    "terminal_displacement_m",
    "maximum_absolute_lateral_m",
    "total_yaw_variation_rad",
    "maximum_discrete_curvature_proxy_per_m",
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _grid(config: Mapping[str, Any]) -> RectangularGrid:
    values = config["H7B"]["candidate_panel"]["node_grid"]
    return RectangularGrid(
        x_min=float(values["x_min_m"]),
        y_min=float(values["y_min_m"]),
        columns=int(values["columns"]),
        rows=int(values["rows"]),
        resolution=float(values["resolution_m"]),
    )


def _body(config: Mapping[str, Any]) -> VehicleRectangle:
    values = config["H7B"]["candidate_panel"]["vehicle"]
    return VehicleRectangle(
        rear_overhang=float(values["rear_overhang_m"]),
        front_extent=float(values["front_extent_m"]),
        half_width=float(values["half_width_m"]),
    )


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("minimum", "q25", "median", "q75", "maximum")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.result) != args.expected_result_sha256:
        raise ValueError("R004H7-B result hash does not match the registered failed-coverage run")
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if (
        result.get("status") != "COMPLETED"
        or result.get("decision") != "INSUFFICIENT_PANEL_COVERAGE"
        or result.get("development_anchor_count") != 384
        or result.get("calibration_test_or_target_accessed") is not False
    ):
        raise ValueError("R004H7-B result is not the frozen coverage-failure instance")
    ledgers = result.get("anchor_ledgers")
    if not isinstance(ledgers, list) or len(ledgers) != 384:
        raise ValueError("R004H7-B anchor ledger is incomplete")
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    grid = _grid(config)
    body = _body(config)

    anchor_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    panel_reason_counts: Counter[str] = Counter()
    candidate_reason_counts: Counter[str] = Counter()
    first_native_boundary_counts: Counter[str] = Counter()
    first_candidate_boundary_counts: Counter[str] = Counter()
    first_native_indices: list[float] = []
    initial_translation: list[float] = []
    initial_yaw: list[float] = []
    native_expansion: dict[str, list[float]] = defaultdict(list)
    candidate_expansion: dict[str, list[float]] = defaultdict(list)
    artifact_failures = 0
    ledger_failures = 0

    for stored in ledgers:
        row = cast(dict[str, Any], stored)
        artifact = args.run_directory / str(row["panel_artifact"])
        if not artifact.is_file() or sha256_file(artifact) != row["panel_artifact_sha256"]:
            artifact_failures += 1
            continue
        with np.load(artifact, allow_pickle=False) as archive:
            realized = np.asarray(archive["realized_local_poses"], dtype=np.float64)
            eligible = np.asarray(archive["challenger_eligible"], dtype=np.bool_)
        candidates = row.get("candidates")
        if (
            realized.shape != (7, 41, 3)
            or eligible.shape != (6,)
            or not isinstance(candidates, list)
            or len(candidates) != 7
        ):
            ledger_failures += 1
            continue

        profiles: list[dict[str, Any]] = []
        recomputed_valid: list[bool] = []
        for index, poses in enumerate(realized):
            profile = support_profile(poses, grid=grid, body=body)
            profile_dict = asdict(profile)
            is_eligible = index == 0 or bool(eligible[index - 1])
            reason = "valid"
            valid = False
            if not bool(row["execution_deterministic_replay_bit_exact"]):
                reason = "execution_not_bit_exact"
            elif not is_eligible:
                reason = "intent_ineligible"
            else:
                try:
                    rasterize_discrete_footprint_nodes(poses, grid=grid, body=body)
                except ValueError as error:
                    reason = str(error)
                else:
                    valid = True
            stored_candidate = cast(dict[str, Any], candidates[index])
            if valid != bool(stored_candidate["valid"]) or reason != stored_candidate["reason"]:
                ledger_failures += 1
            candidate_reason_counts[reason] += 1
            recomputed_valid.append(valid)
            profiles.append(
                {
                    "registered_index": index,
                    "identifier": stored_candidate["identifier"],
                    "eligible": is_eligible,
                    "recomputed_valid": valid,
                    "recomputed_reason": reason,
                    **profile_dict,
                }
            )
            initial_translation.append(profile.initial_translation_residual_m)
            initial_yaw.append(profile.initial_yaw_residual_rad)
            if reason == "oriented footprint leaves the frozen ROI":
                boundary = "+".join(profile.first_violation_boundaries)
                first_candidate_boundary_counts[boundary] += 1
                for name, value in profile.required_expansion_m.items():
                    candidate_expansion[name].append(value)

        panel_valid = recomputed_valid[0] and any(recomputed_valid[1:])
        panel_reason = (
            "native_invalid"
            if not recomputed_valid[0]
            else "valid"
            if any(recomputed_valid[1:])
            else "no_valid_challenger"
        )
        if panel_valid != bool(row["panel_valid"]) or panel_reason != row["panel_reason"]:
            ledger_failures += 1
        native = profiles[0]
        group = str(row["recording_group"])
        panel_reason_counts[panel_reason] += 1
        group_counts[group][panel_reason] += 1
        anchor = {
            "recording_group": group,
            "anchor_index": int(row["anchor_index"]),
            "scene_token": str(row["scene_token"]),
            "panel_valid": panel_valid,
            "panel_reason": panel_reason,
            **{key: native[key] for key in GEOMETRY_KEYS},
        }
        anchor_rows.append(anchor)
        if not panel_valid:
            invalid_rows.append({**anchor, "native": native, "candidates": profiles})
        if panel_reason == "native_invalid":
            boundary = "+".join(cast(list[str], native["first_violation_boundaries"]))
            first_native_boundary_counts[boundary] += 1
            first_index = native["first_violation_index"]
            if first_index is not None:
                first_native_indices.append(float(first_index))
            for name, value in cast(dict[str, float], native["required_expansion_m"]).items():
                native_expansion[name].append(float(value))

    if artifact_failures or ledger_failures or len(anchor_rows) != 384:
        raise AssertionError(
            f"coverage attribution replay failed: artifacts={artifact_failures} "
            f"ledgers={ledger_failures} anchors={len(anchor_rows)}"
        )
    if panel_reason_counts != Counter({"valid": 265, "native_invalid": 110, "no_valid_challenger": 9}):
        raise AssertionError("recomputed panel reason counts disagree with frozen H7-B")

    invalid_reasons = frozenset(("native_invalid", "no_valid_challenger"))
    native_invalid = frozenset(("native_invalid",))
    group_summary = {
        group: {
            "valid": counts["valid"],
            "native_invalid": counts["native_invalid"],
            "no_valid_challenger": counts["no_valid_challenger"],
            "invalid_fraction": (counts["native_invalid"] + counts["no_valid_challenger"]) / 16.0,
        }
        for group, counts in sorted(group_counts.items())
    }
    result_summary: dict[str, Any] = {
        "experiment_id": "R004H7BS0-posthoc-coverage-attribution-v1",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "posthoc_descriptive_outcome_blind",
        "source_result_sha256": sha256_file(args.result),
        "config_sha256": sha256_file(args.config),
        "anchor_count": 384,
        "invalid_anchor_count": 119,
        "panel_reason_counts": dict(panel_reason_counts),
        "candidate_reason_counts": dict(candidate_reason_counts),
        "first_native_violation_boundary_counts": dict(first_native_boundary_counts),
        "first_candidate_violation_boundary_counts": dict(first_candidate_boundary_counts),
        "first_native_violation_index_quantiles": _quantiles(first_native_indices),
        "native_required_expansion_quantiles_m": {
            name: _quantiles(values) for name, values in sorted(native_expansion.items())
        },
        "candidate_required_expansion_quantiles_m": {
            name: _quantiles(values) for name, values in sorted(candidate_expansion.items())
        },
        "coordinate_binding_audit": {
            "origin_tolerance": ORIGIN_TOLERANCE,
            "maximum_initial_translation_residual_m": max(initial_translation),
            "maximum_initial_yaw_residual_rad": max(initial_yaw),
            "origin_contract_pass": max(initial_translation) <= ORIGIN_TOLERANCE
            and max(initial_yaw) <= ORIGIN_TOLERANCE,
            "artifact_failure_count": artifact_failures,
            "ledger_recomputation_failure_count": ledger_failures,
        },
        "group_counts": group_summary,
        "group_equal_descriptive_differences_all_invalid_minus_valid": {
            key: group_equal_descriptive_difference(
                anchor_rows, value_key=key, invalid_reasons=invalid_reasons
            )
            for key in GEOMETRY_KEYS
        },
        "group_equal_descriptive_differences_native_invalid_minus_valid": {
            key: group_equal_descriptive_difference(
                anchor_rows, value_key=key, invalid_reasons=native_invalid
            )
            for key in GEOMETRY_KEYS
        },
        "appearance_validity_contract": {
            "panel_built_once_before_condition_fields": True,
            "appearance_specific_coverage_attribution_permitted": False,
            "invalid_rows_with_conditions_none": sum(
                not bool(row["panel_valid"]) and row.get("conditions") is None for row in ledgers
            ),
        },
        "visual_evidence_status": (
            "NOT_A_CAUSE_OF_RECORDED_PANEL_INVALIDITY__FUTURE_EVIDENCE_AVAILABILITY_UNMEASURED"
        ),
        "calibration_test_navdream_or_target_accessed": False,
        "claim_boundary": (
            "Post-hoc geometry and selection-bias attribution on already observed H7-B panels; "
            "no decision relevance, visual reliability, driving, safety, target, or novelty claim."
        ),
        "invalid_anchor_ledger": invalid_rows,
    }
    _write_json(args.output, result_summary)
    return result_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "invalid_anchor_ledger"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
