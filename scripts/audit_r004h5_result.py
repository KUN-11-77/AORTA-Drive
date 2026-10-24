"""Mechanical, payload-reopening audit for a completed R004H5 result."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aorta_drive.evaluation.appearance_mechanism import (
    measure_field_mechanism,
    oriented_grid_edges,
)
from aorta_drive.vision.appearance import APPEARANCE_CONDITIONS
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h5_appearance_mechanism import _macro_group_summary


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("result artifact path escapes its declared root")
    return path


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _close_tree(left: Any, right: Any, *, tolerance: float = 1.0e-10) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _close_tree(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close_tree(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return bool(left == right)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    raw = json.loads(args.result.read_text(encoding="utf-8"))
    failures: list[str] = []
    if raw.get("status") != "COMPLETED":
        failures.append("raw result is not completed")
    if raw.get("config_sha256") != sha256_file(args.config):
        failures.append("config hash mismatch")
    if raw.get("script_sha256") != sha256_file(args.runner):
        failures.append("runner hash mismatch")
    if raw.get("prohibited_payload_accessed") is not False:
        failures.append("prohibited payload access flag is not false")
    if raw.get("clean_replay_mismatch_count") != 0:
        failures.append("clean DINO replay mismatch is nonzero")
    ledgers = raw.get("anchor_ledgers")
    if not isinstance(ledgers, list) or len(ledgers) != 672:
        failures.append("anchor ledger is not exactly 672 rows")
        ledgers = []
    keys = [
        (str(row.get("recording_group")), int(row.get("anchor_index", -1)))
        for row in ledgers
        if isinstance(row, dict)
    ]
    counts = Counter(group for group, _anchor in keys)
    if len(keys) != 672 or len(set(keys)) != 672 or set(counts.values()) != {16}:
        failures.append("recording-group/anchor ledger is incomplete or duplicated")

    expected_probes = {
        *(f"spatial_pool_{name}" for name in APPEARANCE_CONDITIONS),
        "uniform_fit_mean_DINO",
        *(f"drop_camera_{index}" for index in range(3)),
        *(f"drop_history_{index}" for index in range(4)),
    }
    reopened = 0
    for row in ledgers:
        if not isinstance(row, dict):
            failures.append("invalid anchor ledger row")
            continue
        conditions = row.get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != set(APPEARANCE_CONDITIONS):
            failures.append("condition ledger mismatch")
            continue
        probes = row.get("probes")
        if not isinstance(probes, dict) or set(probes) != expected_probes:
            failures.append("probe ledger mismatch")
        if not _finite_tree(row):
            failures.append("nonfinite anchor ledger value")
        path = _inside(args.result_root, str(row.get("field_artifact")))
        if not path.is_file() or sha256_file(path) != row.get("field_artifact_sha256"):
            failures.append("field artifact missing or hash-mismatched")
            continue
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"teacher", *APPEARANCE_CONDITIONS}:
                failures.append("field artifact keys mismatch")
                continue
            fields = {name: np.asarray(archive[name]) for name in archive.files}
        if any(
            field.dtype != np.float32
            or field.shape != (64, 64)
            or not np.all(np.isfinite(field))
            for field in fields.values()
        ):
            failures.append("field artifact shape/dtype/finiteness failure")
            continue
        clean = fields["clean"]
        teacher = fields["teacher"]
        for condition in APPEARANCE_CONDITIONS:
            recomputed = measure_field_mechanism(fields[condition], teacher, clean)
            stored = conditions[condition]
            if not all(
                math.isclose(
                    float(getattr(recomputed, name)),
                    float(stored[name]),
                    rel_tol=1.0e-5,
                    abs_tol=1.0e-5,
                )
                for name in recomputed.__dataclass_fields__
            ):
                failures.append("stored field metric disagrees with reopened payload")
                break
        zero = float(np.mean(np.abs(oriented_grid_edges(teacher))))
        if not math.isclose(
            zero,
            float(row.get("zero_task_mae_m", math.nan)),
            rel_tol=1.0e-5,
            abs_tol=1.0e-5,
        ):
            failures.append("zero-field metric disagrees with reopened teacher")
        reopened += 1

    if ledgers:
        recomputed_summary = _macro_group_summary(ledgers)
        if not _close_tree(recomputed_summary, raw.get("macro_group_summary")):
            failures.append("macro-group summary does not recompute from anchor ledgers")
        expected_decision = (
            "MECHANISM_SUPPORTED_ON_FIT_CONTROLLED_APPEARANCE"
            if recomputed_summary["all_nonclean_conditions_supported"]
            else "MECHANISM_UNSUPPORTED_ON_FIT_CONTROLLED_APPEARANCE"
        )
        if raw.get("decision") != expected_decision:
            failures.append("decision disagrees with the registered gate")

    return {
        "experiment_id": "R004H5-mechanical-result-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_result_sha256": sha256_file(args.result),
        "reopened_field_artifact_count": reopened,
        "failure_count": len(failures),
        "failures": failures,
        "external_cross_model_review": "REVIEW_UNAVAILABLE",
        "claim_boundary": (
            "Mechanical replay and consistency audit only; external reviewer backend was "
            "unavailable and no self-acquittal was substituted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

