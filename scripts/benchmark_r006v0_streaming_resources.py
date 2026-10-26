#!/usr/bin/env python
"""Outcome-free GPU resource benchmark for the R006V0 streaming JVP solver.

The benchmark uses one cached shifted activation only as a shape-correct base
point and deterministic synthetic tangents.  It emits timing and memory, never
the task objective or a mediation endpoint.  The largest candidate batch that
stays below the preregistered memory ceiling on both qualified stages is chosen.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from aorta_drive.vision.ltf_causal_runtime import (
    LTFCausalRuntime,
    build_frozen_ltf,
    load_ltf_bindings,
)
from aorta_drive.vision.planner_tangent import orthonormalize
from aorta_drive.vision.r006_cache import STAGE_SITE_SEMANTICS, sha256_file


def site_mask(stage: str, device: torch.device) -> torch.Tensor:
    semantics = STAGE_SITE_SEMANTICS[stage]
    mask = torch.zeros(int(semantics["sites"]), dtype=torch.bool, device=device)
    mask[semantics["repairable_site_indices"]] = True
    return mask


def repeated_base(
    value: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    if value.shape[0] != 1:
        raise ValueError("resource benchmark expects one cached anchor activation")
    return value.to(device).repeat(batch_size, *([1] * (value.ndim - 1)))


def one_step(
    runtime: LTFCausalRuntime,
    stage: str,
    base: torch.Tensor,
    rank: int,
    seed: int,
) -> None:
    generator = torch.Generator(device=base.device)
    generator.manual_seed(seed)
    tangent = torch.randn(
        base.shape,
        generator=generator,
        dtype=base.dtype,
        device=base.device,
    )
    mask = site_mask(stage, base.device).to(dtype=base.dtype).view(1, -1, 1)
    channels = base.shape[-1]
    basis = orthonormalize(
        torch.randn(
            channels,
            rank,
            generator=generator,
            dtype=base.dtype,
            device=base.device,
        )
    ).requires_grad_(True)
    projected = ((tangent @ basis) @ basis.transpose(0, 1)) * mask
    residual = tangent - projected
    function = lambda value: runtime.trajectory_xy_from_qualified_stage(stage, value)
    _, product = torch.autograd.functional.jvp(
        function, base, residual, create_graph=True, strict=True
    )
    steps = product.shape[1] // 2
    objective = torch.sum(product * product, dim=1).mean() / steps
    (gradient,) = torch.autograd.grad(objective, basis)
    if not torch.isfinite(gradient).all():
        raise RuntimeError("non-finite synthetic resource gradient")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--planner-root", type=Path, required=True)
    parser.add_argument("--cache-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-ceiling-gib", type=float, default=18.0)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()

    if sorted(set(args.batch_sizes)) != args.batch_sizes or min(args.batch_sizes) <= 0:
        raise ValueError("batch sizes must be unique, positive, and increasing")
    if args.rank != 32 or args.repeats != 3 or args.memory_ceiling_gib != 18.0:
        raise ValueError("formal resource benchmark contract changed")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("an explicitly selected CUDA device is required")

    payload = torch.load(args.cache_shard, map_location="cpu", weights_only=False)
    anchor = payload["anchors"][0]
    bindings = load_ltf_bindings(args.planner_root)
    model, _ = build_frozen_ltf(bindings, args.checkpoint, device)
    runtime = LTFCausalRuntime(model)
    ceiling_bytes = int(args.memory_ceiling_gib * 1024**3)
    measurements: dict[str, list[dict[str, object]]] = {}

    for stage in ("c3_planner_memory", "c4_trajectory_query"):
        cached = anchor["conditions"]["low_exposure"]["activations"][stage]
        stage_rows = []
        for batch_size in args.batch_sizes:
            try:
                base = repeated_base(cached, batch_size, device)
                one_step(runtime, stage, base, args.rank, args.seed + batch_size)
                torch.cuda.synchronize(device)
                durations = []
                torch.cuda.reset_peak_memory_stats(device)
                for repeat in range(args.repeats):
                    start = time.perf_counter()
                    one_step(
                        runtime,
                        stage,
                        base,
                        args.rank,
                        args.seed + 1000 * (repeat + 1) + batch_size,
                    )
                    torch.cuda.synchronize(device)
                    durations.append(time.perf_counter() - start)
                peak_bytes = int(torch.cuda.max_memory_allocated(device))
                median_seconds = float(np.median(durations))
                row = {
                    "batch_size": batch_size,
                    "status": (
                        "PASS" if peak_bytes <= ceiling_bytes else "MEMORY_CEILING_FAIL"
                    ),
                    "median_step_seconds": median_seconds,
                    "samples_per_second": batch_size / median_seconds,
                    "peak_allocated_bytes": peak_bytes,
                }
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = {
                    "batch_size": batch_size,
                    "status": "OOM",
                    "median_step_seconds": None,
                    "samples_per_second": None,
                    "peak_allocated_bytes": None,
                }
            stage_rows.append(row)
        measurements[stage] = stage_rows

    passing = []
    for batch_size in args.batch_sizes:
        if all(
            next(row for row in measurements[stage] if row["batch_size"] == batch_size)[
                "status"
            ]
            == "PASS"
            for stage in measurements
        ):
            passing.append(batch_size)
    if not passing:
        raise RuntimeError("no registered streaming batch size satisfies both stages")
    selected = max(passing)
    result = {
        "artifact": "R006V0_STREAMING_RESOURCE_BENCHMARK_V1",
        "status": "PASS",
        "scientific_metrics_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
        "selection_rule": "largest_registered_batch_below_18GiB_on_both_stages",
        "selected_batch_size": selected,
        "rank": args.rank,
        "repeats": args.repeats,
        "memory_ceiling_bytes": ceiling_bytes,
        "measurements": measurements,
        "provenance": {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "cache_shard_sha256": sha256_file(args.cache_shard),
            "runner_sha256": sha256_file(Path(__file__)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "seed": args.seed,
        },
        "claim_boundary": "resource qualification only; no mediation or method claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "selected_batch_size": selected}, indent=2))


if __name__ == "__main__":
    main()
