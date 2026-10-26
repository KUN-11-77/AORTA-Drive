#!/usr/bin/env python
"""Mechanical audit for the real-LTF matrix-free operator qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


QUALIFIED = ["c3_planner_memory", "c4_trajectory_query"]
EXCLUDED = ["c1_early_visual", "c2_fused_bev"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--all-stage-failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary_bytes = args.primary.read_bytes()
    replay_bytes = args.replay.read_bytes()
    primary = json.loads(primary_bytes)
    replay = json.loads(replay_bytes)
    failure = json.loads(args.all_stage_failure.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def check(name: str, value: bool) -> None:
        checks.append({"check": name, "pass": bool(value)})

    check("primary/replay byte exact", primary_bytes == replay_bytes)
    check("primary/replay payload equal", primary == replay)
    check("qualified primary PASS", primary.get("status") == "PASS")
    check("selected stage roster exact", primary.get("selected_stages") == QUALIFIED)
    check("sealed roles untouched", primary.get("development_calibration_test_navdream_target_accessed") is False)
    check("no scientific metrics emitted", primary.get("scientific_metrics_emitted") is False)
    check("math backend semantics", bool(primary.get("backend_semantics_pass")))
    check("out-of-place head semantics", bool(primary.get("autodiff_head_semantics_pass")))
    check("direct tail semantics", bool(primary.get("direct_tail_semantics_pass")))
    direct_tail_differences = primary.get("direct_tail_max_abs_difference", {})
    for stage in QUALIFIED:
        check(
            f"{stage}: direct tail tolerance",
            stage in direct_tail_differences
            and float(direct_tail_differences[stage])
            <= float(primary.get("backend_output_tolerance", 1e-5)),
        )
    check("all-stage qualification failure retained", failure.get("status") == "FAIL")

    qualified_reports = {row["stage"]: row for row in primary.get("stages", [])}
    failure_reports = {row["stage"]: row for row in failure.get("stages", [])}
    for stage in QUALIFIED:
        row = qualified_reports.get(stage, {})
        check(f"{stage}: present", bool(row))
        check(f"{stage}: finite", row.get("finite") is True)
        check(f"{stage}: basis gradient nonzero", row.get("basis_gradient_norm", 0.0) > 0.0)
        check(f"{stage}: adjoint tolerance", row.get("adjoint_relative_error", 1.0) <= 1e-4 or row.get("adjoint_absolute_error", 1.0) <= 1e-6)
        check(f"{stage}: pass", row.get("pass") is True)
    for stage in EXCLUDED:
        row = failure_reports.get(stage, {})
        check(f"{stage}: exclusion evidence present", bool(row))
        check(f"{stage}: failed frozen operator gate", row.get("pass") is False)
        check(f"{stage}: finite/nonzero despite exclusion", row.get("finite") is True and row.get("basis_gradient_norm", 0.0) > 0.0)

    passed = all(bool(row["pass"]) for row in checks)
    report = {
        "artifact": "R006V0J_MATRIX_FREE_OPERATOR_AUDIT_V2",
        "status": "PASS" if passed else "FAIL",
        "integrity_scope": "mechanical operator qualification only",
        "qualified_stages": QUALIFIED,
        "mechanically_excluded_stages": EXCLUDED,
        "primary_sha256": hashlib.sha256(primary_bytes).hexdigest(),
        "replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "check_count": len(checks),
        "failed_check_count": sum(not bool(row["pass"]) for row in checks),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(checks)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
