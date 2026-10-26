#!/usr/bin/env python
"""Mechanical audit for the outcome-free R006V0 streaming resource benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.input.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def check(name: str, value: bool) -> None:
        checks.append({"check": name, "pass": bool(value)})

    check(
        "artifact", result.get("artifact") == "R006V0_STREAMING_RESOURCE_BENCHMARK_V1"
    )
    check("runner PASS", result.get("status") == "PASS")
    check(
        "sealed roles untouched",
        result.get("development_calibration_test_navdream_target_accessed") is False,
    )
    check("no scientific metrics", result.get("scientific_metrics_emitted") is False)
    check("fixed rank", result.get("rank") == 32)
    check("fixed repeats", result.get("repeats") == 3)
    check("fixed memory ceiling", result.get("memory_ceiling_bytes") == 18 * 1024**3)
    measurements = result.get("measurements", {})
    check(
        "stage roster",
        set(measurements) == {"c3_planner_memory", "c4_trajectory_query"},
    )
    passing: set[int] | None = None
    for stage, rows in measurements.items():
        sizes = [int(row["batch_size"]) for row in rows]
        check(f"{stage}: batch roster", sizes == [16, 32, 64, 128])
        stage_passing = set()
        for row in rows:
            status = row.get("status")
            check(
                f"{stage}/{row['batch_size']}: status valid",
                status in {"PASS", "MEMORY_CEILING_FAIL", "OOM"},
            )
            if status == "PASS":
                finite = all(
                    math.isfinite(float(row[key]))
                    for key in (
                        "median_step_seconds",
                        "samples_per_second",
                        "peak_allocated_bytes",
                    )
                )
                check(f"{stage}/{row['batch_size']}: finite resources", finite)
                check(
                    f"{stage}/{row['batch_size']}: below ceiling",
                    int(row["peak_allocated_bytes"]) <= 18 * 1024**3,
                )
                stage_passing.add(int(row["batch_size"]))
        passing = stage_passing if passing is None else passing & stage_passing
    check("at least one joint passing batch", bool(passing))
    selected = int(result.get("selected_batch_size", -1))
    check(
        "selection rule exact",
        result.get("selection_rule")
        == "largest_registered_batch_below_18GiB_on_both_stages",
    )
    check("selected largest joint pass", bool(passing) and selected == max(passing))
    provenance = result.get("provenance", {})
    check(
        "provenance hashes",
        all(
            bool(provenance.get(key))
            for key in ("checkpoint_sha256", "cache_shard_sha256", "runner_sha256")
        ),
    )

    passed = all(bool(row["pass"]) for row in checks)
    report = {
        "artifact": "R006V0_STREAMING_RESOURCE_AUDIT_V1",
        "status": "PASS" if passed else "FAIL",
        "check_count": len(checks),
        "failed_check_count": sum(not bool(row["pass"]) for row in checks),
        "checks": checks,
        "claim_boundary": "resource qualification only; no mediation or method claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(checks)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
