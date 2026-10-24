"""Mechanical field-level audit for a completed R004H6 result."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aorta_drive.evaluation.appearance_mechanism import measure_field_mechanism, oriented_grid_edges
from aorta_drive.evaluation.heldout_visual import (
    deterministic_group_derangement,
    summarize_heldout_ledgers,
)
from aorta_drive.vision.appearance import H6_APPEARANCE_CONDITIONS
from aorta_drive.vision.feature_cache import sha256_file


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
    elif isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            raise AssertionError(f"boolean mismatch at {path}")
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not np.isclose(float(expected), float(actual), rtol=1.0e-6, atol=1.0e-6):
            raise AssertionError(f"numeric mismatch at {path}: {expected} != {actual}")
    elif expected != actual:
        raise AssertionError(f"value mismatch at {path}: {expected!r} != {actual!r}")


def _salient_edge_mae(prediction: np.ndarray[Any, Any], teacher: np.ndarray[Any, Any]) -> float:
    predicted_edges = oriented_grid_edges(prediction)
    teacher_edges = oriented_grid_edges(teacher)
    selected = np.abs(teacher_edges) >= 0.50
    if not np.any(selected):
        raise ValueError("registered salient-edge set is empty")
    return float(np.mean(np.abs(predicted_edges[selected] - teacher_edges[selected])))


def _assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_json(child, f"{path}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"nonfinite value at {path}")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETED" or result.get("development_anchor_count") != 384:
        raise ValueError("R004H6 result is not complete")
    if result.get("config_sha256") != sha256_file(args.config):
        raise ValueError("R004H6 config hash mismatch")
    if result.get("script_sha256") != sha256_file(args.runner):
        raise ValueError("R004H6 runner hash mismatch")
    ledgers = result.get("anchor_ledgers")
    if not isinstance(ledgers, list) or len(ledgers) != 384:
        raise ValueError("R004H6 anchor ledger is incomplete")
    _assert_finite_json(result)
    groups = {str(row["recording_group"]) for row in ledgers}
    expected_donors = deterministic_group_derangement(groups)
    if result.get("donor_derangement") != expected_donors:
        raise ValueError("stored donor derangement is not the registered mapping")

    recomputed_ledgers: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    artifact_failures = 0
    metric_failures = 0
    for ledger in ledgers:
        key = (str(ledger["recording_group"]), int(ledger["anchor_index"]))
        if key in seen:
            raise ValueError("duplicate group/anchor in R004H6 ledger")
        seen.add(key)
        if ledger.get("donor_recording_group") != expected_donors[key[0]]:
            raise ValueError("anchor donor disagrees with registered derangement")
        artifact = (args.run_directory / str(ledger["field_artifact"])).resolve()
        if not artifact.is_relative_to(args.run_directory.resolve()) or sha256_file(artifact) != ledger["field_artifact_sha256"]:
            artifact_failures += 1
            continue
        with np.load(artifact, allow_pickle=False) as arrays:
            teacher = arrays["teacher"]
            clean = arrays["clean"]
            conditions = {}
            pooled = {}
            for condition in H6_APPEARANCE_CONDITIONS:
                condition_metrics = asdict(measure_field_mechanism(arrays[condition], teacher, clean))
                condition_metrics["salient_edge_mae_m"] = _salient_edge_mae(arrays[condition], teacher)
                conditions[condition] = condition_metrics
                pooled_metrics = asdict(measure_field_mechanism(arrays[f"spatial_pool_{condition}"], teacher, clean))
                pooled_metrics["salient_edge_mae_m"] = _salient_edge_mae(arrays[f"spatial_pool_{condition}"], teacher)
                pooled[condition] = pooled_metrics
            prior = asdict(measure_field_mechanism(arrays["fit_teacher_prior"], teacher, clean))
            mismatch = asdict(measure_field_mechanism(arrays["mismatched_clean"], teacher, clean))
            removed = asdict(measure_field_mechanism(arrays["image_removed"], teacher, clean))
            uniform = asdict(measure_field_mechanism(arrays["uniform_fit_mean_DINO"], teacher, clean))
            prior["salient_edge_mae_m"] = _salient_edge_mae(arrays["fit_teacher_prior"], teacher)
            mismatch["salient_edge_mae_m"] = _salient_edge_mae(arrays["mismatched_clean"], teacher)
            removed["salient_edge_mae_m"] = _salient_edge_mae(arrays["image_removed"], teacher)
            uniform["salient_edge_mae_m"] = _salient_edge_mae(arrays["uniform_fit_mean_DINO"], teacher)
            zero = float(np.mean(np.abs(oriented_grid_edges(teacher))))
        try:
            _assert_close(ledger["conditions"], conditions, "conditions")
            _assert_close(ledger["fit_teacher_prior"], prior, "fit_teacher_prior")
            _assert_close(ledger["mismatched_clean"], mismatch, "mismatched_clean")
            _assert_close(ledger["image_removed"], removed, "image_removed")
            _assert_close(ledger["spatial_pool"], pooled, "spatial_pool")
            _assert_close(ledger["uniform_fit_mean_DINO"], uniform, "uniform_fit_mean_DINO")
            _assert_close(ledger["zero_task_mae_m"], zero, "zero_task_mae_m")
        except AssertionError:
            metric_failures += 1
        replay = dict(ledger)
        replay["conditions"] = conditions
        replay["fit_teacher_prior"] = prior
        replay["mismatched_clean"] = mismatch
        replay["image_removed"] = removed
        replay["spatial_pool"] = pooled
        replay["uniform_fit_mean_DINO"] = uniform
        replay["zero_task_mae_m"] = zero
        recomputed_ledgers.append(replay)
    if artifact_failures or metric_failures or len(recomputed_ledgers) != 384:
        raise AssertionError(f"field audit failed: artifacts={artifact_failures} metrics={metric_failures}")
    summary = summarize_heldout_ledgers(recomputed_ledgers, all_conditions=H6_APPEARANCE_CONDITIONS)
    _assert_close(result["macro_group_summary"], summary, "macro_group_summary")
    if result["decision"] != summary["decision"]:
        raise AssertionError("stored decision disagrees with recomputation")
    return {
        "experiment_id": "R004H6-mechanical-result-audit-v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_sha256": sha256_file(args.result),
        "config_sha256": sha256_file(args.config),
        "runner_sha256": sha256_file(args.runner),
        "field_artifact_count": 384,
        "artifact_hash_failure_count": 0,
        "metric_recomputation_failure_count": 0,
        "recomputed_decision": summary["decision"],
        "claim_boundary": "Mechanical fidelity only; this audit does not expand the R004H6 scientific scope.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
