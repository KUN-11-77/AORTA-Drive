#!/usr/bin/env python3
"""Independently audit and aggregate the completed R006V0F chain."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from aorta_drive.vision.r006_failure_diagnostic import (
    R006V0F_RANKS,
    R006V0F_SEEDS,
    audit_r006v0f_fold_result,
    minimum_qualifying_rank,
    normalized_projector_overlap,
    route_as_dict,
    route_failure_decomposition,
    summarize_drift_denominators,
)
from aorta_drive.vision.r006_cache import sha256_file
from aorta_drive.vision.r006_protocol import audit_r006_nested_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--nested-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty diagnostic vector")
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
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
    progress_path = result_root / "progress.json"
    check(
        "top-level exit code zero",
        exit_path.is_file() and exit_path.read_text().strip() == "0",
    )
    progress = _load_json(progress_path) if progress_path.is_file() else {}
    check("progress complete", progress.get("status") == "COMPLETE")
    check("six folds complete", progress.get("completed_fold_count") == 6)
    check("zero failed folds", progress.get("failed_fold_count") == 0)
    check("zero running folds", progress.get("running_fold_count") == 0)
    check(
        "sealed roles untouched",
        progress.get("development_calibration_test_navdream_target_accessed") is False,
    )
    check(
        "diagnostic science emitted", progress.get("scientific_metrics_emitted") is True
    )

    manifest = _load_json(nested_manifest_path)
    audit_r006_nested_manifest(manifest, expected_groups=manifest["groups"])
    expected_folds = {int(row["outer_fold"]): row for row in manifest["outer_folds"]}
    check("nested manifest has six folds", set(expected_folds) == set(range(6)))

    fold_payloads: dict[int, dict[str, Any]] = {}
    bases: dict[tuple[int, int, int], np.ndarray] = {}
    provenance: dict[str, set[str]] = {}
    fold_audits: list[dict[str, Any]] = []
    denominator_diagnostics: dict[str, Any] = {}
    expected_gpu_by_fold = {0: "0", 1: "1", 2: "6", 3: "7", 4: "0", 5: "1"}
    for fold in range(6):
        fold_root = result_root / f"fold_{fold}"
        primary_dir = fold_root / "primary"
        primary_path = primary_dir / "fold_result.json"
        replay_path = fold_root / "replay" / "fold_result.json"
        stored_audit_path = fold_root / "audit.json"
        check(f"fold {fold}: primary exists", primary_path.is_file())
        check(f"fold {fold}: replay exists", replay_path.is_file())
        check(f"fold {fold}: stored audit exists", stored_audit_path.is_file())
        if not (
            primary_path.is_file()
            and replay_path.is_file()
            and stored_audit_path.is_file()
        ):
            continue
        primary = _load_json(primary_path)
        replay = _load_json(replay_path)
        stored_audit = _load_json(stored_audit_path)
        check(f"fold {fold}: result JSON exact", primary == replay)
        check(f"fold {fold}: stored audit pass", stored_audit.get("status") == "PASS")
        check(
            f"fold {fold}: stored replay exact",
            stored_audit.get("primary_replay_exact") is True,
        )
        recomputed = audit_r006v0f_fold_result(
            primary, result_dir=primary_dir, expected_fold=expected_folds[fold]
        )
        failed = [row["check"] for row in recomputed if not row["pass"]]
        check(f"fold {fold}: independent result audit", not failed)
        fold_audits.append(
            {
                "outer_fold": fold,
                "stored_check_count": stored_audit.get("check_count"),
                "recomputed_check_count": len(recomputed),
                "recomputed_failed_checks": failed,
            }
        )
        fold_payloads[fold] = primary
        for rank in R006V0F_RANKS:
            for seed in R006V0F_SEEDS:
                row = primary["candidates"][str(rank)][str(seed)]
                basis_path = primary_dir / row["basis_file"]
                bases[(fold, rank, seed)] = np.load(
                    basis_path, allow_pickle=False
                ).astype(np.float64)
        for key, value in primary.get("provenance", {}).items():
            if key in {"device", "cuda_visible_devices"}:
                continue
            provenance.setdefault(key, set()).add(str(value))
        check(
            f"fold {fold}: source commit",
            primary.get("provenance", {}).get("source_commit")
            == expected_source_commit,
        )
        check(
            f"fold {fold}: frozen physical GPU assignment",
            str(primary.get("provenance", {}).get("cuda_visible_devices"))
            == expected_gpu_by_fold[fold],
        )

        reference_samples = primary["candidates"][str(R006V0F_RANKS[0])][
            str(R006V0F_SEEDS[0])
        ]["selection_samples"]
        reference_keys = [
            (
                str(row["recording_group"]),
                int(row["anchor_index"]),
                str(row["condition"]),
            )
            for row in reference_samples
        ]
        reference_totals = np.asarray(
            [float(row["total_task_drift_sq"]) for row in reference_samples],
            dtype=np.float64,
        )
        invariant = True
        for rank in R006V0F_RANKS:
            for seed in R006V0F_SEEDS:
                samples = primary["candidates"][str(rank)][str(seed)][
                    "selection_samples"
                ]
                keys = [
                    (
                        str(row["recording_group"]),
                        int(row["anchor_index"]),
                        str(row["condition"]),
                    )
                    for row in samples
                ]
                totals = np.asarray(
                    [float(row["total_task_drift_sq"]) for row in samples],
                    dtype=np.float64,
                )
                invariant = (
                    invariant
                    and keys == reference_keys
                    and np.array_equal(totals, reference_totals)
                )
        check(f"fold {fold}: coverage denominator invariant", invariant)
        denominator_diagnostics[str(fold)] = {
            **vars(summarize_drift_denominators(reference_totals, epsilon=1e-12)),
            "interpretation": (
                "descriptive denominator-health diagnostic only; the frozen "
                "coverage formula and route are unchanged"
            ),
        }

    check("all fold payloads available", set(fold_payloads) == set(range(6)))
    for key, values in sorted(provenance.items()):
        check(f"cross-fold provenance stable: {key}", len(values) == 1)

    seed_minimum_ranks: dict[int, dict[int, int | None]] = {}
    qualification_flip_folds: list[int] = []
    qualification_by_fold_rank_seed: dict[str, Any] = {}
    for fold, payload in sorted(fold_payloads.items()):
        seed_minimum_ranks[fold] = {}
        fold_rows: dict[str, Any] = {}
        fold_has_flip = False
        for rank in R006V0F_RANKS:
            values = {
                seed: bool(
                    payload["candidates"][str(rank)][str(seed)]["selection_summary"][
                        "qualifies"
                    ]
                )
                for seed in R006V0F_SEEDS
            }
            fold_rows[str(rank)] = {str(seed): value for seed, value in values.items()}
            fold_has_flip = fold_has_flip or len(set(values.values())) > 1
        if fold_has_flip:
            qualification_flip_folds.append(fold)
        for seed in R006V0F_SEEDS:
            seed_minimum_ranks[fold][seed] = minimum_qualifying_rank(
                {rank: bool(fold_rows[str(rank)][str(seed)]) for rank in R006V0F_RANKS}
            )
        qualification_by_fold_rank_seed[str(fold)] = fold_rows

    route = route_failure_decomposition(seed_minimum_ranks, qualification_flip_folds)
    geometry: dict[str, Any] = {}
    for rank in R006V0F_RANKS:
        within_fold_seed: list[float] = []
        for fold in range(6):
            for left_seed, right_seed in combinations(R006V0F_SEEDS, 2):
                within_fold_seed.append(
                    normalized_projector_overlap(
                        bases[(fold, rank, left_seed)],
                        bases[(fold, rank, right_seed)],
                    )
                )
        across_fold_same_seed: list[float] = []
        for seed in R006V0F_SEEDS:
            for left_fold, right_fold in combinations(range(6), 2):
                across_fold_same_seed.append(
                    normalized_projector_overlap(
                        bases[(left_fold, rank, seed)],
                        bases[(right_fold, rank, seed)],
                    )
                )
        geometry[str(rank)] = {
            "within_fold_across_seed_overlap": _summary(within_fold_seed),
            "across_fold_same_seed_overlap": _summary(across_fold_same_seed),
        }

    failed_checks = [row for row in checks if not row["pass"]]
    mechanical_status = "PASS" if not failed_checks else "FAIL"
    return {
        "artifact": "R006V0F_CHAIN_AUDIT_V1",
        "mechanical_status": mechanical_status,
        "diagnostic_route": (
            route_as_dict(route) if mechanical_status == "PASS" else None
        ),
        "qualification_flip_folds": qualification_flip_folds,
        "seed_minimum_ranks": {
            str(fold): {str(seed): rank for seed, rank in values.items()}
            for fold, values in seed_minimum_ranks.items()
        },
        "qualification_by_fold_rank_seed": qualification_by_fold_rank_seed,
        "projector_geometry": geometry,
        "coverage_denominator_diagnostics": denominator_diagnostics,
        "fold_audits": fold_audits,
        "provenance_distinct_values": {
            key: sorted(values) for key, values in provenance.items()
        },
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "auditor_provenance": {
            "script_sha256": sha256_file(Path(__file__)),
            "scope": "post-run independent aggregate audit; not a scientific runner",
        },
        "claim_boundary": (
            "Post-failure inner diagnostic only. The parent STOP is immutable; "
            "folds and seeds are blocked/repeated diagnostic factors rather than "
            "independent scientific replicates. No outer mediation or downstream "
            "method claim is supported."
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
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "mechanical_status": result["mechanical_status"],
                "diagnostic_route": (
                    result["diagnostic_route"]["status"]
                    if result["diagnostic_route"] is not None
                    else None
                ),
                "check_count": result["check_count"],
                "failed_check_count": result["failed_check_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["mechanical_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
