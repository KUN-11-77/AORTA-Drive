"""Independent mechanical recomputation for a completed R004H7-B run."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.candidates import generate_native_offset_candidates
from aorta_drive.evaluation.h4 import qualify_intent_match, terminal_segment_average_speed
from aorta_drive.evaluation.heldout_visual import validate_serialized_teacher_projection
from aorta_drive.evaluation.shadow_decision import (
    build_shadow_panel,
    evaluate_shadow_condition,
    summarize_shadow_ledgers,
)
from aorta_drive.training.r004ec import load_joined_anchors, load_teacher_arrays
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7b_shadow_decision import (
    _development_groups,
    _grid,
    _inside,
    _intent_limits,
    _json_safe,
    _panel_ledger,
    _records_by_scene,
    _write_json,
)


def _assert_close(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            raise AssertionError(f"key mismatch at {path}")
        for key in expected:
            _assert_close(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise AssertionError(f"length mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_close(left, right, f"{path}[{index}]")
    elif expected is None or actual is None:
        if expected is not actual:
            raise AssertionError(f"null mismatch at {path}")
    elif isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            raise AssertionError(f"boolean mismatch at {path}")
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not np.isclose(float(expected), float(actual), rtol=1.0e-12, atol=1.0e-12):
            raise AssertionError(f"numeric mismatch at {path}: {expected} != {actual}")
    elif expected != actual:
        raise AssertionError(f"value mismatch at {path}: {expected!r} != {actual!r}")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage = config["H7B"]
    frozen = stage["frozen_inputs"]
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETED" or result.get("development_anchor_count") != 384:
        raise ValueError("H7B result is incomplete")
    if result.get("config_sha256") != sha256_file(args.config):
        raise ValueError("H7B result config hash mismatch")
    if result.get("runner_sha256") != sha256_file(args.runner):
        raise ValueError("H7B result runner hash mismatch")

    bound_inputs = {
        "r004h6_result_sha256": args.h6_result,
        "r004h6_exact_teacher_audit_sha256": args.h6_exact_teacher_audit,
        "source_split_manifest_sha256": args.source_split_manifest,
        "feature_manifest_sha256": args.feature_manifest,
        "label_manifest_sha256": args.label_manifest,
        "metric_manifest_sha256": args.metric_manifest,
    }
    for hash_key, path in bound_inputs.items():
        digest = sha256_file(path)
        if digest != frozen[hash_key] or result["input_hashes"].get(hash_key) != digest:
            raise ValueError(f"H7B frozen input mismatch: {hash_key}")
    h6 = json.loads(args.h6_result.read_text(encoding="utf-8"))
    h6_audit = json.loads(args.h6_exact_teacher_audit.read_text(encoding="utf-8"))
    if h6_audit.get("status") != "PASS" or h6_audit.get(
        "result_sha256"
    ) != sha256_file(args.h6_result):
        raise ValueError("upstream exact-teacher audit is not bound to H6")
    h6_ledgers = h6.get("anchor_ledgers")
    output_ledgers = result.get("anchor_ledgers")
    if not isinstance(h6_ledgers, list) or not isinstance(output_ledgers, list):
        raise ValueError("H6 or H7B lacks anchor ledgers")
    if len(h6_ledgers) != 384 or len(output_ledgers) != 384:
        raise ValueError("H6 or H7B anchor ledger is incomplete")
    h6_by_key = {
        (str(row["recording_group"]), int(row["anchor_index"])): row for row in h6_ledgers
    }
    h7_by_key = {
        (str(row["recording_group"]), int(row["anchor_index"])): row for row in output_ledgers
    }
    if len(h6_by_key) != 384 or set(h6_by_key) != set(h7_by_key):
        raise ValueError("H6 and H7B panels do not align")

    groups = _development_groups(args.source_split_manifest)
    group_order = {group: index for index, group in enumerate(groups)}
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
    if set(records) != set(h7_by_key):
        raise ValueError("exact teacher join does not align with H7B")
    metric_manifest = json.loads(args.metric_manifest.read_text(encoding="utf-8"))
    label_manifest = json.loads(args.label_manifest.read_text(encoding="utf-8"))
    metric_by_scene = _records_by_scene(metric_manifest, name="metric")
    label_by_scene = _records_by_scene(label_manifest, name="label")
    grid = _grid(config)
    limits = _intent_limits(config)
    amplitudes = tuple(
        float(value) for value in stage["candidate_panel"]["challenger_amplitudes_m"]
    )
    conditions = tuple(str(value) for value in stage["appearances"])
    comparison_indices = np.arange(0, 41, 5, dtype=np.int64)
    recomputed: list[dict[str, Any]] = []
    artifact_failures = 0
    metric_failures = 0
    for key in sorted(records, key=lambda value: (group_order[value[0]], value[1])):
        record = records[key]
        stored = h7_by_key[key]
        h6_row = h6_by_key[key]
        scene_key = (record.log_name, record.scene_token)
        metric = metric_by_scene.get(scene_key)
        label = label_by_scene.get(scene_key)
        if metric is None or label is None:
            raise ValueError("metric or label record missing in audit")
        cache_path = _inside(args.metric_root, str(metric["relative_cache_path"]))
        cache_hash = sha256_file(cache_path)
        if (
            cache_hash != metric.get("cache_sha256")
            or cache_hash != label.get("cache_sha256")
            or cache_hash != stored.get("metric_cache_sha256")
        ):
            raise ValueError("metric-cache/label/result binding failed")

        if stored.get("panel_reason") == "provider_emitted_no_challenger":
            raise AssertionError("registered provider unexpectedly emitted no challenger")
        panel_path = _inside(args.run_directory, str(stored["panel_artifact"]))
        if sha256_file(panel_path) != stored.get("panel_artifact_sha256"):
            artifact_failures += 1
            continue
        with np.load(panel_path, allow_pickle=False) as panel_arrays:
            required = {
                "native_desired_poses",
                "challenger_desired_poses",
                "realized_local_poses",
                "challenger_eligible",
                "candidate_amplitudes_m",
            }
            if set(panel_arrays.files) != required:
                raise ValueError("panel artifact array contract changed")
            native_desired = np.asarray(panel_arrays["native_desired_poses"], dtype=np.float64)
            challenger_desired = np.asarray(
                panel_arrays["challenger_desired_poses"], dtype=np.float64
            )
            realized = np.asarray(panel_arrays["realized_local_poses"], dtype=np.float64)
            stored_eligible = np.asarray(panel_arrays["challenger_eligible"], dtype=np.bool_)
            stored_amplitudes = np.asarray(panel_arrays["candidate_amplitudes_m"], dtype=np.float64)
        regenerated = generate_native_offset_candidates(
            native_desired,
            amplitudes_m=amplitudes,
            minimum_reference_displacement_m=1.0e-6,
            enforce_desired_curve_dynamic_screen=False,
        )
        regenerated_poses = np.stack([candidate.poses for candidate in regenerated.candidates])
        regenerated_amplitudes = np.asarray(
            [candidate.control.amplitude_m for candidate in regenerated.candidates],
            dtype=np.float64,
        )
        if not np.array_equal(regenerated_poses, challenger_desired) or not np.array_equal(
            regenerated_amplitudes, stored_amplitudes
        ):
            raise AssertionError("panel artifact does not reproduce the frozen candidate provider")
        half = realized[:, comparison_indices]
        eligibility = [
            qualify_intent_match(
                half[index],
                half[0],
                challenger_terminal_speed_mps=terminal_segment_average_speed(
                    half[index], interval_s=0.5
                ),
                native_terminal_speed_mps=terminal_segment_average_speed(
                    half[0], interval_s=0.5
                ),
                limits=limits,
            )
            for index in range(1, half.shape[0])
        ]
        recomputed_eligible = np.asarray([value.eligible for value in eligibility], dtype=np.bool_)
        if not np.array_equal(recomputed_eligible, stored_eligible):
            raise AssertionError("stored eligibility does not replay from realized trajectories")
        identifiers = ("native",) + tuple(
            f"normal_offset_{value:+.1f}m" for value in stored_amplitudes
        )
        # Recover the frozen body without trusting result metadata. The exact
        # dimensions are already bound in the config and checked below.
        from aorta_drive.certification.geometry import VehicleRectangle

        expected_vehicle = stage["candidate_panel"]["vehicle"]
        body = VehicleRectangle(
            rear_overhang=float(expected_vehicle["rear_overhang_m"]),
            front_extent=float(expected_vehicle["front_extent_m"]),
            half_width=float(expected_vehicle["half_width_m"]),
        )
        panel = build_shadow_panel(
            realized,
            challenger_eligible=tuple(bool(value) for value in recomputed_eligible),
            identifiers=identifiers,
            deterministic_execution=bool(stored["execution_deterministic_replay_bit_exact"]),
            grid=grid,
            body=body,
        )
        replay: dict[str, Any] = {
            **{
                key_name: stored[key_name]
                for key_name in stored
                if key_name not in {"conditions", "candidates", "intent_eligibility"}
            },
            "provider_rejection_counts": dict(regenerated.rejection_counts),
            "intent_eligibility": [_json_safe(asdict(value)) for value in eligibility],
            "candidates": _panel_ledger(panel),
            "conditions": None,
        }
        if panel.valid:
            field_path = _inside(args.h6_run_root, str(h6_row["field_artifact"]))
            if sha256_file(field_path) != h6_row.get("field_artifact_sha256"):
                raise ValueError("H6 field artifact hash mismatch in H7B audit")
            exact_teacher = load_teacher_arrays(record).label_m
            with np.load(field_path, allow_pickle=False) as fields:
                teacher = validate_serialized_teacher_projection(fields["teacher"], exact_teacher)
                replay["conditions"] = {
                    condition: evaluate_shadow_condition(
                        fields[condition],
                        teacher,
                        panel,
                        dino_drift=float(
                            h6_row["perturbation_ledger"][condition]["dense_feature_rms"]
                        ),
                    )
                    for condition in conditions
                }
        replay["panel_valid"] = panel.valid
        replay["panel_reason"] = panel.reason
        try:
            _assert_close(stored, cast(dict[str, Any], _json_safe(replay)), f"anchor[{key}]")
        except AssertionError:
            metric_failures += 1
        recomputed.append(cast(dict[str, Any], _json_safe(replay)))

    if artifact_failures or metric_failures or len(recomputed) != 384:
        raise AssertionError(
            f"H7B audit failed: artifacts={artifact_failures} metrics={metric_failures} "
            f"recomputed={len(recomputed)}"
        )
    summary = summarize_shadow_ledgers(
        recomputed,
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
    safe_summary = _json_safe(summary)
    _assert_close(result["summary"], safe_summary, "summary")
    if result["decision"] != safe_summary["decision"]:
        raise AssertionError("stored H7B decision disagrees with independent recomputation")
    return {
        "experiment_id": "R004H7B-mechanical-recomputation-audit-v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_sha256": sha256_file(args.result),
        "config_sha256": sha256_file(args.config),
        "runner_sha256": sha256_file(args.runner),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "panel_artifact_count": 384,
        "artifact_failure_count": 0,
        "metric_recomputation_failure_count": 0,
        "recomputed_decision": safe_summary["decision"],
        "external_review_status": "REVIEW_UNAVAILABLE",
        "claim_boundary": (
            "Mechanical fidelity of the frozen development shadow proxy only; "
            "no continuous geometry, driving, safety, test, target, or novelty claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
