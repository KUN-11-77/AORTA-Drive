#!/usr/bin/env python
"""Fail-closed mechanical and replay audit for R006V0 fold basis selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aorta_drive.vision.r006_basis_audit import audit_fold_basis_result
from aorta_drive.vision.r006_cache import sha256_file
from aorta_drive.vision.r006_protocol import audit_r006_nested_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--nested-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path)
    args = parser.parse_args()

    result_path = args.result_dir / "fold_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    nested = json.loads(args.nested_manifest.read_text(encoding="utf-8"))
    audit_r006_nested_manifest(nested, expected_groups=nested["groups"])
    fold_id = int(result.get("outer_fold", -1))
    fold = next(
        (row for row in nested["outer_folds"] if int(row["outer_fold"]) == fold_id),
        None,
    )
    if fold is None:
        raise ValueError("result outer fold is not registered")
    checks, _ = audit_fold_basis_result(result, result_dir=args.result_dir, expected_fold=fold)

    def check(name: str, value: bool) -> None:
        checks.append({"check": name, "pass": bool(value)})

    provenance = result.get("provenance", {})
    expected_hashes = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "cache_manifest_sha256": sha256_file(args.cache_dir / "manifest.json"),
        "nested_manifest_sha256": sha256_file(args.nested_manifest),
        "config_sha256": sha256_file(args.config),
        "runner_sha256": sha256_file(args.source_root / "scripts/run_r006v0_fold_basis.py"),
        "basis_module_sha256": sha256_file(
            args.source_root / "src/aorta_drive/vision/r006_basis.py"
        ),
        "solver_module_sha256": sha256_file(
            args.source_root / "src/aorta_drive/vision/planner_tangent.py"
        ),
    }
    for key, expected in expected_hashes.items():
        check(f"provenance exact: {key}", provenance.get(key) == expected)
    check("source commit exact", provenance.get("source_commit") == args.expected_source_commit)
    cache_manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    check(
        "cache checkpoint binding",
        provenance.get("cache_checkpoint_sha256")
        == cache_manifest.get("provenance", {}).get("checkpoint_sha256"),
    )
    check(
        "cache planner tree binding",
        provenance.get("cache_planner_execution_tree")
        == cache_manifest.get("provenance", {})
        .get("planner_source", {})
        .get("execution_subtree_tree"),
    )

    replay_exact = None
    if args.replay_dir is not None:
        replay_path = args.replay_dir / "fold_result.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        replay_checks, _ = audit_fold_basis_result(
            replay,
            result_dir=args.replay_dir,
            expected_fold=fold,
        )
        check(
            "replay independently passes mechanics",
            all(bool(row["pass"]) for row in replay_checks),
        )
        check("primary/replay result JSON exact", result == replay)
        primary_bases = sorted((args.result_dir / "bases").glob("*.npy"))
        replay_bases = sorted((args.replay_dir / "bases").glob("*.npy"))
        replay_exact = (
            [path.name for path in primary_bases] == [path.name for path in replay_bases]
            and all(
                sha256_file(left) == sha256_file(right)
                for left, right in zip(primary_bases, replay_bases, strict=True)
            )
        )
        check("primary/replay basis bytes exact", replay_exact)

    passed = all(bool(row["pass"]) for row in checks)
    report = {
        "artifact": "R006V0_FOLD_BASIS_AUDIT_V1",
        "status": "PASS" if passed else "FAIL",
        "outer_fold": fold_id,
        "check_count": len(checks),
        "failed_check_count": sum(not bool(row["pass"]) for row in checks),
        "primary_replay_exact": replay_exact,
        "checks": checks,
        "claim_boundary": (
            "fold-level inner selection mechanics only; no outer-confirmation, "
            "mediation, DINO, CPTR, NavDream, behavior, latency, or safety claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(checks)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
