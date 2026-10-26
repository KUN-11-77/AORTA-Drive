#!/usr/bin/env python
"""Compare two complete R006V0 caches without reading scientific endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-cache", type=Path, required=True)
    parser.add_argument("--replay-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary = json.loads(
        (args.primary_cache / "manifest.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (args.replay_cache / "manifest.json").read_text(encoding="utf-8")
    )
    checks: list[dict[str, object]] = []

    def check(name: str, value: bool) -> None:
        checks.append({"check": name, "pass": bool(value)})

    for name, manifest in (("primary", primary), ("replay", replay)):
        check(f"{name}: PASS", manifest.get("status") == "PASS")
        check(f"{name}: complete", manifest.get("complete_formal_cache") is True)
        check(f"{name}: 42 groups", manifest.get("groups") == 42)
        check(f"{name}: 672 anchors", manifest.get("anchors") == 672)
        check(
            f"{name}: sealed roles",
            manifest.get("development_calibration_test_navdream_target_accessed")
            is False,
        )
        check(
            f"{name}: no science metrics",
            manifest.get("scientific_metrics_emitted") is False,
        )
    for key in ("conditions", "qualified_stages", "schema", "stage_site_semantics"):
        check(f"manifest {key} exact", primary.get(key) == replay.get(key))
    for key in ("checkpoint_sha256", "group_roles_sha256", "time_set_sha256"):
        check(
            f"provenance {key} exact",
            primary.get("provenance", {}).get(key)
            == replay.get("provenance", {}).get(key),
        )
    replay_source = replay.get("provenance", {}).get("planner_source", {})
    check(
        "replay planner execution subtree clean",
        replay_source.get("execution_subtree_dirty") is False,
    )
    check(
        "replay planner execution tree recorded",
        bool(replay_source.get("execution_subtree_tree")),
    )

    primary_shards = {row["recording_group"]: row for row in primary.get("shards", [])}
    replay_shards = {row["recording_group"]: row for row in replay.get("shards", [])}
    check("group roster exact", set(primary_shards) == set(replay_shards))
    exact_shards = 0
    for group in sorted(set(primary_shards) | set(replay_shards)):
        left = primary_shards.get(group)
        right = replay_shards.get(group)
        exact = bool(
            left
            and right
            and left.get("sha256") == right.get("sha256")
            and left.get("anchors") == right.get("anchors") == 16
        )
        check(f"{group}: shard hash exact", exact)
        exact_shards += int(exact)
    check("42 exact shard hashes", exact_shards == 42)

    passed = all(bool(row["pass"]) for row in checks)
    report = {
        "artifact": "R006V0_ACTIVATION_CACHE_REPLAY_AUDIT_V1",
        "status": "PASS" if passed else "FAIL",
        "scientific_metrics_emitted": False,
        "check_count": len(checks),
        "failed_check_count": sum(not bool(row["pass"]) for row in checks),
        "exact_shard_count": exact_shards,
        "checks": checks,
        "claim_boundary": "cache determinism and provenance only; no mediation or method claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"status": report["status"], "exact_shards": exact_shards}, indent=2)
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
