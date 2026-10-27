#!/usr/bin/env python3
"""Independently audit and aggregate the completed R006V0 basis chain."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aorta_drive.vision.r006_basis_audit import audit_fold_basis_result
from aorta_drive.vision.r006_cache import sha256_file
from aorta_drive.vision.r006_protocol import (
    InterfaceSelection,
    aggregate_cross_fold_interfaces,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--nested-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def audit_chain(
    result_root: Path,
    nested_manifest_path: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool) -> None:
        checks.append({"check": name, "pass": bool(passed)})

    exit_path = result_root / "exit_code.txt"
    check("top-level exit code exists", exit_path.is_file())
    check(
        "top-level exit code zero",
        exit_path.is_file() and exit_path.read_text(encoding="utf-8").strip() == "0",
    )
    progress_path = result_root / "progress.json"
    check("progress exists", progress_path.is_file())
    progress = _load_json(progress_path) if progress_path.is_file() else {}
    check("progress complete", progress.get("status") == "COMPLETE")
    check("six folds complete", progress.get("completed_fold_count") == 6)
    check("zero failed folds", progress.get("failed_fold_count") == 0)
    check("zero running folds", progress.get("running_fold_count") == 0)
    check("sealed roles untouched", progress.get("development_calibration_test_navdream_target_accessed") is False)
    check("basis science emitted", progress.get("scientific_metrics_emitted") is True)

    manifest = _load_json(nested_manifest_path)
    expected_folds = {
        int(row["outer_fold"]): row for row in manifest.get("outer_folds", [])
    }
    check("nested manifest has six folds", set(expected_folds) == set(range(6)))
    observed_fold_dirs = {
        int(path.name.removeprefix("fold_"))
        for path in result_root.glob("fold_*")
        if path.is_dir() and path.name.removeprefix("fold_").isdigit()
    }
    check("result fold roster exact", observed_fold_dirs == set(range(6)))

    selections: list[InterfaceSelection | None] = []
    selected_rows: list[dict[str, object]] = []
    fold_audit_summaries: list[dict[str, object]] = []
    provenance_values: dict[str, set[str]] = {
        key: set()
        for key in (
            "basis_module_sha256",
            "cache_checkpoint_sha256",
            "cache_manifest_sha256",
            "cache_planner_execution_tree",
            "checkpoint_sha256",
            "config_sha256",
            "nested_manifest_sha256",
            "runner_sha256",
            "solver_module_sha256",
            "source_commit",
        )
    }
    qualification_counts: dict[str, dict[int, int]] = {
        "planner_memory": {rank: 0 for rank in (4, 8, 16, 32)},
        "trajectory_query_secondary": {rank: 0 for rank in (4, 8, 16, 32)},
    }

    for fold in range(6):
        fold_root = result_root / f"fold_{fold}"
        primary_dir = fold_root / "primary"
        replay_dir = fold_root / "replay"
        primary_path = primary_dir / "fold_result.json"
        replay_path = replay_dir / "fold_result.json"
        stored_audit_path = fold_root / "audit.json"
        check(f"fold {fold}: primary result exists", primary_path.is_file())
        check(f"fold {fold}: replay result exists", replay_path.is_file())
        check(f"fold {fold}: stored audit exists", stored_audit_path.is_file())
        if not (primary_path.is_file() and replay_path.is_file() and stored_audit_path.is_file()):
            selections.append(None)
            continue

        primary_hashes = _tree_hashes(primary_dir)
        replay_hashes = _tree_hashes(replay_dir)
        check(f"fold {fold}: primary replay file roster exact", set(primary_hashes) == set(replay_hashes))
        check(f"fold {fold}: primary replay byte exact", primary_hashes == replay_hashes)

        primary = _load_json(primary_path)
        replay = _load_json(replay_path)
        check(f"fold {fold}: result JSON exact", primary == replay)
        recomputed_checks, _ = audit_fold_basis_result(
            primary,
            result_dir=primary_dir,
            expected_fold=expected_folds[fold],
        )
        failed_recomputed = [item for item in recomputed_checks if not item["pass"]]
        check(f"fold {fold}: independent fold audit", not failed_recomputed)

        stored_audit = _load_json(stored_audit_path)
        stored_checks = stored_audit.get("checks", [])
        check(f"fold {fold}: stored audit status", stored_audit.get("status") == "PASS")
        check(f"fold {fold}: stored audit failed count", stored_audit.get("failed_check_count") == 0)
        check(f"fold {fold}: stored audit checks all pass", bool(stored_checks) and all(item.get("pass") is True for item in stored_checks))
        check(f"fold {fold}: stored exact replay", stored_audit.get("primary_replay_exact") is True)

        selected = primary.get("selected_interface")
        if isinstance(selected, dict):
            selection = InterfaceSelection(
                layer=str(selected["layer"]),
                rank=int(selected["rank"]),
                group_equal_mean=float(selected["group_equal_mean"]),
                worst_group_mean=float(selected["worst_group_mean"]),
                rule=str(selected["rule"]),
            )
            selections.append(selection)
            selected_rows.append({"outer_fold": fold, **asdict(selection)})
        else:
            selections.append(None)
            selected_rows.append({"outer_fold": fold, "selection": None})

        for layer, ranks in primary.get("candidates", {}).items():
            for rank_text, row in ranks.items():
                summary = row.get("selection_summary", {})
                if summary.get("qualifies") is True:
                    qualification_counts[str(layer)][int(rank_text)] += 1

        provenance = primary.get("provenance", {})
        check(f"fold {fold}: source commit", provenance.get("source_commit") == expected_source_commit)
        for key in provenance_values:
            provenance_values[key].add(str(provenance.get(key)))
        fold_audit_summaries.append(
            {
                "outer_fold": fold,
                "stored_check_count": stored_audit.get("check_count"),
                "recomputed_check_count": len(recomputed_checks),
                "recomputed_failed_checks": [item["check"] for item in failed_recomputed],
                "primary_replay_exact": primary_hashes == replay_hashes,
            }
        )

    for key, values in provenance_values.items():
        check(f"cross-fold provenance stable: {key}", len(values) == 1)

    consensus = aggregate_cross_fold_interfaces(selections)
    failed_checks = [item for item in checks if not item["pass"]]
    mechanical_status = "PASS" if not failed_checks else "FAIL"
    scientific_decision = consensus.status if mechanical_status == "PASS" else None
    return {
        "artifact": "R006V0_BASIS_CHAIN_AUDIT_V1",
        "mechanical_status": mechanical_status,
        "scientific_decision": scientific_decision,
        "cross_fold_consensus": asdict(consensus),
        "selected_interfaces": selected_rows,
        "qualification_counts_by_layer_rank": qualification_counts,
        "fold_audits": fold_audit_summaries,
        "provenance_distinct_values": {
            key: sorted(values) for key, values in provenance_values.items()
        },
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "claim_boundary": (
            "This stage evaluates inner-held-out basis qualification and the "
            "preregistered six-fold interface-consensus rule only. It does not "
            "read outer-confirmation endpoints or support finite mediation, DINO "
            "recoverability, CPTR, OOD, utility, latency, or safety claims."
        ),
    }


def main() -> int:
    args = parse_args()
    result = audit_chain(
        args.result_root.resolve(),
        args.nested_manifest.resolve(),
        args.expected_source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("mechanical_status", "scientific_decision", "check_count", "failed_check_count")}, sort_keys=True))
    return 0 if result["mechanical_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
