"""Run the preregistered read-only R004H7-B shadow-decision diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import pickle
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.certification.geometry import RectangularGrid
from aorta_drive.evaluation.h4 import (
    IntentEligibilityLimits,
    qualify_intent_match,
    sample_cached_native_trajectory,
    terminal_segment_average_speed,
)
from aorta_drive.evaluation.heldout_visual import validate_serialized_teacher_projection
from aorta_drive.evaluation.navsim_execution import execute_official_navsim_batch
from aorta_drive.evaluation.navsim_geometry import vehicle_geometry_contract
from aorta_drive.evaluation.shadow_decision import (
    ShadowPanel,
    build_shadow_panel,
    evaluate_shadow_condition,
    summarize_shadow_ledgers,
)
from aorta_drive.training.r004ec import load_joined_anchors, load_teacher_arrays
from aorta_drive.vision.feature_cache import sha256_file


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes its declared root")
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _development_groups(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = tuple(row["recording_group"] for row in rows if row["split"] == "development")
    if len(groups) != 24 or len(set(groups)) != 24:
        raise ValueError("source split manifest does not define 24 unique development groups")
    return groups


def _records_by_scene(
    manifest: Mapping[str, Any], *, name: str
) -> dict[tuple[str, str], dict[str, Any]]:
    raw = manifest.get("records")
    if not isinstance(raw, list):
        raise ValueError(f"{name} manifest lacks records")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError(f"{name} manifest contains a non-object record")
        if name == "label":
            parts = str(value.get("cache_id", "")).split("/")
            if len(parts) != 2:
                raise ValueError("label cache_id is invalid")
            key = (parts[0], parts[1])
        else:
            key = (str(value.get("log_name")), str(value.get("scene_token")))
        if key in result:
            raise ValueError(f"{name} manifest has duplicate scene keys")
        result[key] = cast(dict[str, Any], value)
    return result


def _intent_limits(config: Mapping[str, Any]) -> IntentEligibilityLimits:
    values = config["H7B"]["candidate_panel"]["intent_eligibility"]
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


def _grid(config: Mapping[str, Any]) -> RectangularGrid:
    values = config["H7B"]["candidate_panel"]["node_grid"]
    return RectangularGrid(
        x_min=float(values["x_min_m"]),
        y_min=float(values["y_min_m"]),
        columns=int(values["columns"]),
        rows=int(values["rows"]),
        resolution=float(values["resolution_m"]),
    )


def _vehicle_matches(config: Mapping[str, Any], contract: Any) -> bool:
    expected = config["H7B"]["candidate_panel"]["vehicle"]
    return bool(
        contract.width_m == float(expected["width_m"])
        and contract.length_m == float(expected["length_m"])
        and contract.rear_axle_to_center_m == float(expected["rear_axle_to_center_m"])
        and contract.body.rear_overhang == float(expected["rear_overhang_m"])
        and contract.body.front_extent == float(expected["front_extent_m"])
        and contract.body.half_width == float(expected["half_width_m"])
    )


def _verify_implementation(
    config: Mapping[str, Any], *, repository: Path, navsim_root: Path
) -> None:
    frozen = config["H7B"]["implementation"]
    local = {
        "shadow_decision_source_sha256": repository
        / "src/aorta_drive/evaluation/shadow_decision.py",
        "navsim_execution_source_sha256": repository
        / "src/aorta_drive/evaluation/navsim_execution.py",
        "native_offset_source_sha256": repository
        / "src/aorta_drive/candidates/native_offset.py",
        "intent_filter_source_sha256": repository / "src/aorta_drive/evaluation/h4.py",
        "runner_source_sha256": Path(__file__).resolve(),
    }
    for key, path in local.items():
        if sha256_file(path) != frozen[key]:
            raise ValueError(f"H7B local implementation hash mismatch: {key}")
    for relative, expected in frozen["official_source_sha256"].items():
        if sha256_file(navsim_root / relative) != expected:
            raise ValueError(f"H7B official source hash mismatch: {relative}")


def _panel_ledger(panel: ShadowPanel) -> list[dict[str, Any]]:
    return [
        {
            "registered_index": row.registered_index,
            "identifier": row.identifier,
            "valid": row.valid,
            "reason": row.reason,
            "node_count": int(row.node_indices.size),
        }
        for row in panel.candidates
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage = config["H7B"]
    frozen = stage["frozen_inputs"]
    _verify_implementation(
        config,
        repository=Path(__file__).resolve().parents[1],
        navsim_root=args.navsim_root,
    )
    paths = {
        "r004h6_result_sha256": args.h6_result,
        "r004h6_exact_teacher_audit_sha256": args.h6_exact_teacher_audit,
        "source_split_manifest_sha256": args.source_split_manifest,
        "feature_manifest_sha256": args.feature_manifest,
        "label_manifest_sha256": args.label_manifest,
        "metric_manifest_sha256": args.metric_manifest,
    }
    for hash_key, path in paths.items():
        if sha256_file(path) != frozen[hash_key]:
            raise ValueError(f"frozen input hash mismatch: {hash_key}")

    h6 = json.loads(args.h6_result.read_text(encoding="utf-8"))
    h6_audit = json.loads(args.h6_exact_teacher_audit.read_text(encoding="utf-8"))
    if (
        h6.get("status") != "COMPLETED"
        or h6.get("decision") != "SUPPORTED"
        or h6.get("development_anchor_count") != 384
    ):
        raise ValueError("R004H6 result is not the frozen completed supported run")
    if (
        h6_audit.get("status") != "PASS"
        or h6_audit.get("result_sha256") != frozen["r004h6_result_sha256"]
        or h6_audit.get("recomputed_decision") != "SUPPORTED"
    ):
        raise ValueError("R004H6 exact-teacher audit did not bind the frozen result")
    h6_ledgers = h6.get("anchor_ledgers")
    if not isinstance(h6_ledgers, list) or len(h6_ledgers) != 384:
        raise ValueError("R004H6 anchor ledger is incomplete")

    development_groups = _development_groups(args.source_split_manifest)
    group_order = {group: index for index, group in enumerate(development_groups)}
    joined = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
        allowed_splits=("train", "development"),
    )
    records = {
        (row.recording_group, row.anchor_index): row
        for row in joined
        if row.recording_group in group_order
    }
    if len(records) != 384:
        raise ValueError("exact teacher join does not contain the 384-anchor development panel")
    metric_manifest = json.loads(args.metric_manifest.read_text(encoding="utf-8"))
    label_manifest = json.loads(args.label_manifest.read_text(encoding="utf-8"))
    metric_by_scene = _records_by_scene(metric_manifest, name="metric")
    label_by_scene = _records_by_scene(label_manifest, name="label")
    ledger_by_key = {
        (str(row["recording_group"]), int(row["anchor_index"])): row for row in h6_ledgers
    }
    if len(ledger_by_key) != 384 or set(ledger_by_key) != set(records):
        raise ValueError("R004H6 and exact-teacher development panels disagree")

    grid = _grid(config)
    limits = _intent_limits(config)
    amplitudes = tuple(
        float(value) for value in stage["candidate_panel"]["challenger_amplitudes_m"]
    )
    conditions = tuple(str(value) for value in stage["appearances"])
    if conditions != ("clean", "low_exposure", "low_contrast", "cool_cast"):
        raise ValueError("H7B confirmatory appearance contract changed")

    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)
    panel_root = args.output_directory / "panels"
    panel_root.mkdir()
    output_ledgers: list[dict[str, Any]] = []
    ordered_keys = sorted(records, key=lambda key: (group_order[key[0]], key[1]))
    comparison_indices = np.arange(0, 41, 5, dtype=np.int64)
    for number, key in enumerate(ordered_keys, start=1):
        record = records[key]
        h6_row = ledger_by_key[key]
        if h6_row.get("scene_token") != record.scene_token:
            raise ValueError("R004H6 ledger scene token disagrees with exact teacher join")
        scene_key = (record.log_name, record.scene_token)
        metric = metric_by_scene.get(scene_key)
        label = label_by_scene.get(scene_key)
        if metric is None or label is None:
            raise ValueError("metric or label manifest lacks a registered H7B scene")
        if (
            metric.get("recording_group") != record.recording_group
            or int(metric.get("anchor_index", -1)) != record.anchor_index
            or metric.get("split") != "development"
        ):
            raise ValueError("metric record role or identity mismatch")
        cache_path = _inside(args.metric_root, str(metric["relative_cache_path"]))
        cache_hash = sha256_file(cache_path)
        if cache_hash != metric.get("cache_sha256") or cache_hash != label.get("cache_sha256"):
            raise ValueError("metric cache is not exactly bound to the teacher label")
        with lzma.open(cache_path, "rb") as stream:
            cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM metric cache
        contract = vehicle_geometry_contract(cache.ego_state.car_footprint.vehicle_parameters)
        if not _vehicle_matches(config, contract):
            raise ValueError("official vehicle geometry changed")

        native = sample_cached_native_trajectory(cache)
        desired = generate_native_offset_candidates(
            native.poses,
            amplitudes_m=amplitudes,
            minimum_reference_displacement_m=1.0e-6,
            enforce_desired_curve_dynamic_screen=False,
        )
        if not desired.candidates:
            output_ledgers.append(
                {
                    "recording_group": record.recording_group,
                    "anchor_index": record.anchor_index,
                    "scene_token": record.scene_token,
                    "panel_valid": False,
                    "panel_reason": "provider_emitted_no_challenger",
                    "provider_rejection_counts": dict(desired.rejection_counts),
                    "candidates": [],
                    "conditions": None,
                }
            )
            continue
        desired_array = np.stack([candidate.poses for candidate in desired.candidates])
        execution = execute_official_navsim_batch(cache, desired_array)
        half = execution.realized_local_poses[:, comparison_indices]
        native_half = half[0]
        eligibility = [
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
        ]
        identifiers = ("native",) + tuple(
            f"normal_offset_{candidate.control.amplitude_m:+.1f}m"
            for candidate in desired.candidates
        )
        panel = build_shadow_panel(
            execution.realized_local_poses,
            challenger_eligible=tuple(row.eligible for row in eligibility),
            identifiers=identifiers,
            deterministic_execution=execution.deterministic_replay_bit_exact,
            grid=grid,
            body=contract.body,
        )
        panel_path = panel_root / f"{record.scene_token}.npz"
        np.savez_compressed(
            panel_path,
            native_desired_poses=native.poses,
            challenger_desired_poses=desired_array,
            realized_local_poses=execution.realized_local_poses,
            challenger_eligible=np.asarray([row.eligible for row in eligibility], dtype=np.bool_),
            candidate_amplitudes_m=np.asarray(
                [candidate.control.amplitude_m for candidate in desired.candidates],
                dtype=np.float64,
            ),
        )
        row: dict[str, Any] = {
            "recording_group": record.recording_group,
            "anchor_index": record.anchor_index,
            "scene_token": record.scene_token,
            "log_name": record.log_name,
            "metric_cache_sha256": cache_hash,
            "panel_valid": panel.valid,
            "panel_reason": panel.reason,
            "provider_rejection_counts": dict(desired.rejection_counts),
            "intent_eligibility": [_json_safe(asdict(value)) for value in eligibility],
            "execution_deterministic_replay_bit_exact": execution.deterministic_replay_bit_exact,
            "maximum_realized_transform_error_m": execution.maximum_transform_error_m,
            "candidates": _panel_ledger(panel),
            "panel_artifact": panel_path.relative_to(args.output_directory).as_posix(),
            "panel_artifact_sha256": sha256_file(panel_path),
            "conditions": None,
        }
        if panel.valid:
            field_path = _inside(args.h6_run_root, str(h6_row["field_artifact"]))
            if sha256_file(field_path) != h6_row.get("field_artifact_sha256"):
                raise ValueError("R004H6 field artifact hash mismatch")
            exact_teacher = load_teacher_arrays(record).label_m
            with np.load(field_path, allow_pickle=False) as fields:
                teacher = validate_serialized_teacher_projection(fields["teacher"], exact_teacher)
                condition_results = {}
                for condition in conditions:
                    drift = float(h6_row["perturbation_ledger"][condition]["dense_feature_rms"])
                    condition_results[condition] = evaluate_shadow_condition(
                        fields[condition], teacher, panel, dino_drift=drift
                    )
            row["conditions"] = condition_results
        output_ledgers.append(row)

        elapsed = time.perf_counter() - started
        output_mib = (
            sum(
                path.stat().st_size
                for path in args.output_directory.rglob("*")
                if path.is_file()
            )
            / 2**20
        )
        resources = stage["resources"]
        if elapsed > float(resources["maximum_wall_minutes"]) * 60:
            raise RuntimeError("H7B exceeded the frozen wall-time cap")
        if output_mib > float(resources["maximum_output_mib"]):
            raise RuntimeError("H7B exceeded the frozen output-size cap")
        if number % 16 == 0:
            progress = {
                "experiment_id": config["experiment_id"],
                "stage": "H7B",
                "status": "IN_PROGRESS",
                "completed_anchor_count": number,
                "total_anchor_count": 384,
                "calibration_test_or_target_accessed": False,
            }
            _write_json(args.output_directory / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)

    summary = summarize_shadow_ledgers(
        output_ledgers,
        conditions=conditions,
        seed=int(stage["bootstrap_seed"]),
        replicates=int(stage["bootstrap_replicates"]),
        expected_groups=int(stage["groups"]),
        anchors_per_group=int(stage["anchors_per_group"]),
        minimum_valid_anchors_per_group=int(
            stage["panel_validity"]["minimum_valid_anchors_per_group"]
        ),
        minimum_valid_bootstrap_fraction=float(stage["minimum_valid_bootstrap_fraction"]),
    )
    result = {
        "experiment_id": config["experiment_id"],
        "stage": "H7B",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "runner_sha256": sha256_file(Path(__file__)),
        "input_hashes": {key: sha256_file(path) for key, path in paths.items()},
        "development_group_count": 24,
        "development_anchor_count": 384,
        "conditions": list(conditions),
        "anchor_ledgers": output_ledgers,
        "summary": summary,
        "decision": summary["decision"],
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_test_or_target_accessed": False,
        "claim_boundary": (
            "Historically exposed development shadow decision proxy only; no training, "
            "continuous geometry, driving, safety, test, NavDream, target, or novelty claim."
        ),
    }
    safe_result = cast(dict[str, Any], _json_safe(result))
    _write_json(args.output_directory / "result.json", safe_result)
    return safe_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--h6-result", type=Path, required=True)
    parser.add_argument("--h6-run-root", type=Path, required=True)
    parser.add_argument("--h6-exact-teacher-audit", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--navsim-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "anchor_ledgers"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
