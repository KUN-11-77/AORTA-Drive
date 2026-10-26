#!/usr/bin/env python
"""Mechanical integrity audit for the R006V0S qualification smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_STAGES = {
    "c1_early_visual",
    "c2_fused_bev",
    "c3_planner_memory",
    "c4_trajectory_query",
}
EXPECTED_C1_CUT = "first_image_encoder_return_layer_before_any_cross_modal_fusion"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": message, "pass": bool(condition)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--retained-failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    for path in (args.synthetic, args.primary, args.replay, args.retained_failure):
        require(path.is_file(), f"artifact exists: {path}", checks)
    if not all(item["pass"] for item in checks):
        raise FileNotFoundError("one or more required artifacts are missing")

    synthetic = json.loads(args.synthetic.read_text(encoding="utf-8"))
    primary_bytes = args.primary.read_bytes()
    replay_bytes = args.replay.read_bytes()
    primary = json.loads(primary_bytes)
    replay = json.loads(replay_bytes)
    retained_failure = json.loads(args.retained_failure.read_text(encoding="utf-8"))

    require(bool(synthetic.get("pass")), "synthetic dense-Jacobian oracle passes", checks)
    require(primary_bytes == replay_bytes, "primary and replay are byte exact", checks)
    require(primary == replay, "primary and replay payloads are equal", checks)
    require(primary.get("status") == "PASS", "qualification status is PASS", checks)
    require(
        primary.get("scientific_metrics_emitted") is False,
        "no scientific metrics emitted",
        checks,
    )
    require(
        primary.get("development_calibration_test_navdream_target_accessed") is False,
        "sealed roles remain untouched",
        checks,
    )
    require(primary.get("anchors") == 2, "exactly two source-fit anchors used", checks)
    require(primary.get("c1_cut_definition") == EXPECTED_C1_CUT, "C1 is a complete pre-fusion causal cut", checks)
    require(retained_failure.get("status") == "FAIL", "incomplete-cut failure is retained", checks)

    provenance = primary.get("provenance", {})
    path_hash_pairs = (
        ("checkpoint", "checkpoint_sha256"),
        ("instrumentation", "instrumentation_sha256"),
        ("imported_model_path", "imported_model_sha256"),
        ("imported_builder_path", "imported_builder_sha256"),
    )
    for path_key, hash_key in path_hash_pairs:
        path = Path(provenance.get(path_key, ""))
        require(path.is_file(), f"provenance file exists: {path_key}", checks)
        if path.is_file():
            require(
                sha256_file(path) == provenance.get(hash_key),
                f"provenance hash matches: {hash_key}",
                checks,
            )
    require(
        provenance.get("planner_source", {}).get("commit") is not None,
        "planner commit recorded",
        checks,
    )

    require(len(primary.get("anchor_reports", [])) == 2, "two anchor reports present", checks)
    for anchor_index, anchor in enumerate(primary.get("anchor_reports", [])):
        prefix = f"anchor[{anchor_index}]"
        require(anchor.get("role") == "fit", f"{prefix} has fit role", checks)
        require(bool(anchor.get("clean_replay_exact")), f"{prefix} clean replay exact", checks)
        require(bool(anchor.get("activation_replay_exact")), f"{prefix} activation replay exact", checks)
        stages = anchor.get("stages", [])
        require({stage.get("stage") for stage in stages} == EXPECTED_STAGES, f"{prefix} stage set complete", checks)
        for stage in stages:
            stage_prefix = f"{prefix}.{stage.get('stage')}"
            require(bool(stage.get("identity_exact")), f"{stage_prefix} identity exact", checks)
            require(bool(stage.get("finite")), f"{stage_prefix} values finite", checks)
            require(bool(stage.get("pass")), f"{stage_prefix} restoration passes", checks)

    passed = all(item["pass"] for item in checks)
    report = {
        "artifact": "R006V0S_MECHANICAL_AUDIT_V1",
        "status": "PASS" if passed else "FAIL",
        "integrity_scope": "mechanical qualification only; not scientific efficacy",
        "primary_sha256": hashlib.sha256(primary_bytes).hexdigest(),
        "replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "check_count": len(checks),
        "failed_check_count": sum(not item["pass"] for item in checks),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(checks)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
