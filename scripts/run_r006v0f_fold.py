#!/usr/bin/env python
"""Run one R006V0F inner-only repeated-initialization diagnostic fold."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from aorta_drive.vision.ltf_causal_runtime import (
    CONTROLLED_APPEARANCES,
    LTFCausalRuntime,
    build_frozen_ltf,
    load_ltf_bindings,
)
from aorta_drive.vision.planner_tangent import solve_streaming_matrix_free_basis
from aorta_drive.vision.r006_basis import (
    evaluate_basis_coverage,
    matrix_free_batch_objective,
)
from aorta_drive.vision.r006_cache import (
    STAGE_SITE_SEMANTICS,
    sha256_file,
    validate_group_shard,
)
from aorta_drive.vision.r006_failure_diagnostic import R006V0F_RANKS, R006V0F_SEEDS
from aorta_drive.vision.r006_protocol import (
    audit_r006_nested_manifest,
    summarize_basis_coverage,
)


STAGE = "c4_trajectory_query"
LAYER = "trajectory_query_secondary"


def load_cache(cache_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the already-qualified source-fit cache with strict provenance checks."""

    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("complete_formal_cache") is not True
    ):
        raise RuntimeError("formal cache is not complete and passing")
    if (
        manifest.get("development_calibration_test_navdream_target_accessed")
        is not False
    ):
        raise RuntimeError("formal cache reports sealed-role access")
    if manifest.get("scientific_metrics_emitted") is not False:
        raise RuntimeError("formal cache must not contain scientific endpoints")
    planner_source = manifest.get("provenance", {}).get("planner_source", {})
    if planner_source.get(
        "execution_subtree_dirty"
    ) is not False or not planner_source.get("execution_subtree_tree"):
        raise RuntimeError("planner execution-subtree provenance is not qualified")

    by_group: dict[str, dict[str, Any]] = {}
    for record in manifest.get("shards", []):
        path = cache_dir / record["file"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"cache shard provenance failed: {record['file']}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_group_shard(payload)
        group = str(payload["recording_group"])
        if group in by_group:
            raise RuntimeError(f"duplicate cache group {group}")
        by_group[group] = payload
    if len(by_group) != 42:
        raise RuntimeError("formal cache must contain exactly 42 groups")
    return manifest, by_group


def collect_samples(
    cache: dict[str, dict[str, Any]], groups: list[str]
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Collect C4 shifted bases and clean-minus-shift deltas in fixed order."""

    bases = []
    deltas = []
    records: list[dict[str, Any]] = []
    for group in sorted(groups):
        payload = cache.get(group)
        if payload is None:
            raise RuntimeError(f"group missing from cache: {group}")
        for anchor in sorted(
            payload["anchors"], key=lambda row: int(row["anchor_index"])
        ):
            clean = anchor["conditions"]["clean"]["activations"][STAGE].squeeze(0)
            for condition in CONTROLLED_APPEARANCES:
                shifted = anchor["conditions"][condition]["activations"][STAGE].squeeze(
                    0
                )
                bases.append(shifted)
                deltas.append(clean - shifted)
                records.append(
                    {
                        "recording_group": group,
                        "anchor_index": int(anchor["anchor_index"]),
                        "scene_token": str(anchor["scene_token"]),
                        "condition": condition,
                    }
                )
    expected = len(groups) * 16 * len(CONTROLLED_APPEARANCES)
    if len(records) != expected:
        raise RuntimeError("stage sample cardinality mismatch")
    return torch.stack(bases), torch.stack(deltas), records


def validate_config(config: dict[str, Any]) -> None:
    """Fail closed if any registered diagnostic factor changed."""

    qualification = config.get("basis_qualification", {})
    solver = qualification.get("solver", {})
    if config.get("layer") != LAYER or config.get("stage") != STAGE:
        raise ValueError("diagnostic layer or stage changed")
    if tuple(config.get("ranks", [])) != R006V0F_RANKS:
        raise ValueError("rank dictionary changed")
    if tuple(config.get("random_only_initialization_seeds", [])) != R006V0F_SEEDS:
        raise ValueError("seed roster changed")
    expected = {
        "include_pca_initialization": False,
        "one_random_initialization_per_seed_job": True,
        "streaming_batch_size": 128,
        "retraction": "qr",
        "max_line_search_steps": 50,
        "relative_improvement_tolerance": 1e-6,
        "patience": 20,
        "max_steps": 500,
    }
    for key, value in expected.items():
        if solver.get(key) != value:
            raise ValueError(f"solver contract changed at {key}")
    if solver.get("line_search") != {
        "type": "armijo",
        "initial_step": 1.0,
        "shrink": 0.5,
        "sufficient_decrease": 1e-4,
    }:
        raise ValueError("line-search contract changed")
    if qualification.get("selection_mean_coverage_min") != 0.90:
        raise ValueError("mean coverage gate changed")
    if qualification.get("selection_worst_group_mean_coverage_min") != 0.50:
        raise ValueError("worst-group coverage gate changed")
    if qualification.get("coverage_epsilon") != 1e-12:
        raise ValueError("coverage epsilon changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--planner-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--nested-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if len(args.source_commit) != 40:
        raise ValueError("source commit must be a full SHA")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    nested = json.loads(args.nested_manifest.read_text(encoding="utf-8"))
    audit_r006_nested_manifest(nested, expected_groups=nested["groups"])
    if args.outer_fold not in range(6):
        raise ValueError("outer fold must lie in [0,5]")
    fold = next(
        row
        for row in nested["outer_folds"]
        if int(row["outer_fold"]) == args.outer_fold
    )
    discovery_groups = list(fold["basis_discovery_groups"])
    selection_groups = list(fold["layer_rank_selection_groups"])
    outer_groups = list(fold["outer_confirmation_groups"])
    if set(discovery_groups) & set(selection_groups) or (
        set(discovery_groups) | set(selection_groups)
    ) & set(outer_groups):
        raise RuntimeError("nested roles overlap")
    if (len(discovery_groups), len(selection_groups), len(outer_groups)) != (28, 7, 7):
        raise RuntimeError("nested role cardinalities changed")

    cache_manifest, cache = load_cache(args.cache_dir)
    torch.manual_seed(20261027)
    np.random.seed(20261027)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("an explicitly selected CUDA device is required")

    bindings = load_ltf_bindings(args.planner_root)
    model, _ = build_frozen_ltf(bindings, args.checkpoint, device)
    runtime = LTFCausalRuntime(model)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("output directory must be absent or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basis_dir = args.output_dir / "bases"
    basis_dir.mkdir(exist_ok=True)

    discovery_bases_cpu, discovery_deltas_cpu, discovery_records = collect_samples(
        cache, discovery_groups
    )
    selection_bases_cpu, selection_deltas_cpu, selection_records = collect_samples(
        cache, selection_groups
    )
    discovery_bases = discovery_bases_cpu.to(device)
    discovery_deltas = discovery_deltas_cpu.to(device)
    selection_bases = selection_bases_cpu.to(device)
    selection_deltas = selection_deltas_cpu.to(device)
    mask = torch.ones(
        int(STAGE_SITE_SEMANTICS[STAGE]["sites"]), dtype=torch.bool, device=device
    )
    function = lambda value: runtime.trajectory_xy_from_qualified_stage(STAGE, value)
    candidates: dict[str, dict[str, Any]] = {}
    total_jobs = len(R006V0F_RANKS) * len(R006V0F_SEEDS)
    completed_jobs = 0

    for rank in R006V0F_RANKS:
        rank_rows: dict[str, Any] = {}
        for seed in R006V0F_SEEDS:

            def batch_objective(
                basis: torch.Tensor, start: int, end: int
            ) -> torch.Tensor:
                return matrix_free_batch_objective(
                    function,
                    discovery_bases[start:end],
                    discovery_deltas[start:end],
                    basis,
                    mask,
                )

            solved = solve_streaming_matrix_free_basis(
                discovery_deltas,
                rank,
                batch_objective,
                batch_size=128,
                random_seeds=(seed,),
                include_pca_initialization=False,
                initial_step=1.0,
                armijo_shrink=0.5,
                armijo_c=1e-4,
                relative_tolerance=1e-6,
                patience=20,
                max_steps=500,
                max_line_search_steps=50,
            )
            basis = solved.basis.detach()
            total, uncovered = evaluate_basis_coverage(
                function,
                selection_bases,
                selection_deltas,
                basis,
                mask,
                batch_size=128,
            )
            group_ids = [str(row["recording_group"]) for row in selection_records]
            summary = summarize_basis_coverage(
                total.cpu(),
                uncovered.cpu(),
                group_ids,
                rank=rank,
                epsilon=1e-12,
                mean_threshold=0.90,
                worst_group_threshold=0.50,
            )
            basis_path = (
                basis_dir / f"fold_{args.outer_fold}_rank_{rank}_seed_{seed}.npy"
            )
            np.save(basis_path, basis.float().cpu().numpy(), allow_pickle=False)
            samples = [
                {
                    **record,
                    "total_task_drift_sq": float(total_value),
                    "uncovered_task_drift_sq": float(uncovered_value),
                }
                for record, total_value, uncovered_value in zip(
                    selection_records,
                    total.cpu().tolist(),
                    uncovered.cpu().tolist(),
                    strict=True,
                )
            ]
            rank_rows[str(seed)] = {
                "basis_file": str(basis_path.relative_to(args.output_dir)),
                "basis_sha256": sha256_file(basis_path),
                "basis_shape": [256, rank],
                "discovery_objective": solved.objective,
                "solver_steps": solved.steps,
                "solver_initialization": solved.initialization,
                "discovery_sample_count": len(discovery_records),
                "selection_sample_count": len(selection_records),
                "selection_summary": {
                    "rank": summary.rank,
                    "group_count": summary.group_count,
                    "group_equal_mean": summary.group_equal_mean,
                    "worst_group_mean": summary.worst_group_mean,
                    "qualifies": summary.qualifies,
                    "group_means": summary.group_means,
                },
                "selection_samples": samples,
            }
            completed_jobs += 1
            print(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "outer_fold": args.outer_fold,
                        "completed_job_count": completed_jobs,
                        "total_job_count": total_jobs,
                        "scientific_metrics_emitted": False,
                    }
                ),
                flush=True,
            )
        candidates[str(rank)] = rank_rows

    result = {
        "artifact": "R006V0F_FOLD_FAILURE_DECOMPOSITION_V1",
        "status": "PASS_DIAGNOSTIC_MATERIALIZED",
        "mechanical_status": "PASS",
        "scientific_metrics_emitted": True,
        "outer_confirmation_metrics_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
        "outer_fold": args.outer_fold,
        "roles": {
            "basis_discovery_groups": discovery_groups,
            "layer_rank_selection_groups": selection_groups,
            "outer_confirmation_groups": outer_groups,
        },
        "conditions": list(CONTROLLED_APPEARANCES),
        "registered_layer": LAYER,
        "registered_ranks": list(R006V0F_RANKS),
        "registered_seeds": list(R006V0F_SEEDS),
        "streaming_batch_size": 128,
        "output_metric": "W=I_(2T)/T",
        "candidates": candidates,
        "provenance": {
            "source_commit": args.source_commit,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "cache_manifest_sha256": sha256_file(args.cache_dir / "manifest.json"),
            "nested_manifest_sha256": sha256_file(args.nested_manifest),
            "config_sha256": sha256_file(args.config),
            "runner_sha256": sha256_file(Path(__file__)),
            "diagnostic_module_sha256": sha256_file(
                Path(__file__).parents[1]
                / "src/aorta_drive/vision/r006_failure_diagnostic.py"
            ),
            "basis_module_sha256": sha256_file(
                Path(__file__).parents[1] / "src/aorta_drive/vision/r006_basis.py"
            ),
            "solver_module_sha256": sha256_file(
                Path(__file__).parents[1] / "src/aorta_drive/vision/planner_tangent.py"
            ),
            "cache_checkpoint_sha256": cache_manifest["provenance"][
                "checkpoint_sha256"
            ],
            "cache_planner_execution_tree": cache_manifest["provenance"][
                "planner_source"
            ]["execution_subtree_tree"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "claim_boundary": (
            "post-failure inner diagnostic only; parent STOP immutable; no outer "
            "mediation, DINO, CPTR, NavDream, behavior, latency, or safety claim"
        ),
    }
    output_path = args.output_dir / "fold_result.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "output": str(output_path)}))


if __name__ == "__main__":
    main()
