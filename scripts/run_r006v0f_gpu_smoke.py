#!/usr/bin/env python
"""Real-GPU mechanical smoke for the frozen R006V0F diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from aorta_drive.vision.ltf_causal_runtime import (
    LTFCausalRuntime,
    build_frozen_ltf,
    load_ltf_bindings,
)
from aorta_drive.vision.planner_tangent import solve_streaming_matrix_free_basis
from aorta_drive.vision.r006_basis import matrix_free_batch_objective
from aorta_drive.vision.r006_cache import STAGE_SITE_SEMANTICS, sha256_file
from aorta_drive.vision.r006_failure_diagnostic import audit_smoke_replay
from aorta_drive.vision.r006_protocol import audit_r006_nested_manifest
from run_r006v0f_fold import STAGE, collect_samples, load_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--planner-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--nested-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if len(args.source_commit) != 40:
        raise ValueError("source commit must be a full SHA")
    nested = json.loads(args.nested_manifest.read_text(encoding="utf-8"))
    audit_r006_nested_manifest(nested, expected_groups=nested["groups"])
    fold = next(row for row in nested["outer_folds"] if row["outer_fold"] == 0)
    discovery_group = [sorted(fold["basis_discovery_groups"])[0]]
    selection_group = [sorted(fold["layer_rank_selection_groups"])[0]]
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
    discovery_bases, discovery_deltas, _ = collect_samples(cache, discovery_group)
    selection_bases, selection_deltas, _ = collect_samples(cache, selection_group)
    discovery_bases = discovery_bases.to(device)
    discovery_deltas = discovery_deltas.to(device)
    selection_bases = selection_bases.to(device)
    selection_deltas = selection_deltas.to(device)
    if (
        not torch.isfinite(selection_bases).all()
        or not torch.isfinite(selection_deltas).all()
    ):
        raise RuntimeError("smoke selection tensors are non-finite")
    mask = torch.ones(
        int(STAGE_SITE_SEMANTICS[STAGE]["sites"]), dtype=torch.bool, device=device
    )
    function = lambda value: runtime.trajectory_xy_from_qualified_stage(STAGE, value)

    def solve_once() -> np.ndarray:
        def batch_objective(basis: torch.Tensor, start: int, end: int) -> torch.Tensor:
            return matrix_free_batch_objective(
                function,
                discovery_bases[start:end],
                discovery_deltas[start:end],
                basis,
                mask,
            )

        solved = solve_streaming_matrix_free_basis(
            discovery_deltas,
            4,
            batch_objective,
            batch_size=48,
            random_seeds=(2026102701,),
            include_pca_initialization=False,
            initial_step=1.0,
            armijo_shrink=0.5,
            armijo_c=1e-4,
            relative_tolerance=1e-6,
            patience=2,
            max_steps=2,
            max_line_search_steps=2,
        )
        if solved.initialization != "random_2026102701":
            raise RuntimeError(
                "smoke did not use the frozen random-only initialization"
            )
        return solved.basis.detach().float().cpu().numpy()

    primary = solve_once()
    replay = solve_once()
    audit = audit_smoke_replay(primary, replay, expected_rank=4)
    artifact = {
        "artifact": "R006V0F_REAL_GPU_SMOKE_V1",
        **audit,
        "scientific_metrics_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
        "outer_confirmation_accessed": False,
        "scope": "one discovery group and one selection group; mechanical only",
        "solver": {
            "rank": 4,
            "seed": 2026102701,
            "max_steps": 2,
            "max_line_search_steps": 2,
            "include_pca_initialization": False,
        },
        "provenance": {
            "source_commit": args.source_commit,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "cache_manifest_sha256": sha256_file(args.cache_dir / "manifest.json"),
            "nested_manifest_sha256": sha256_file(args.nested_manifest),
            "runner_sha256": sha256_file(Path(__file__)),
            "cache_checkpoint_sha256": cache_manifest["provenance"][
                "checkpoint_sha256"
            ],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "claim_boundary": "mechanical smoke only; no scientific or behavioral claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": artifact["status"], "output": str(args.output)}))
    if artifact["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
