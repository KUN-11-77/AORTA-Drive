"""Synthetic shape, matched-compute, and scorer-only resource preflight for R005E2."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager, cast

import numpy as np
import torch
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from aorta_drive.models.full_context_scorer import FullContextSetScorer
from aorta_drive.vision.feature_cache import sha256_file


ARM_NAMES = ("V0_removed", "V1_mismatch", "V2_correct")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _autocast(device: torch.device, enabled: bool) -> ContextManager[Any]:
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model(seed: int, device: torch.device) -> FullContextSetScorer:
    torch.manual_seed(seed)
    return FullContextSetScorer().to(device).eval()


def _parameter_schema(model: FullContextSetScorer) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "requires_grad": bool(parameter.requires_grad),
            "count": parameter.numel(),
        }
        for name, parameter in model.named_parameters()
    ]


def _synthetic_inputs(device: torch.device, seed: int) -> dict[str, tuple[Tensor, ...]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, candidates, time = 1, 7, 41
    clock = torch.linspace(0.0, 4.0, time)
    candidate = torch.zeros(batch, candidates, time, 3)
    for index in range(candidates):
        candidate[0, index, :, 0] = 3.0 * clock
        candidate[0, index, :, 1] = (index - 3) * 0.25 * clock
        candidate[0, index, :, 2] = (index - 3) * 0.01 * clock
    candidate_mask = torch.ones(batch, candidates, dtype=torch.bool)
    previous = candidate[:, 0].clone()
    correct = torch.randn(
        batch,
        12,
        18,
        32,
        1280,
        generator=generator,
        dtype=torch.bfloat16,
    )
    mismatch = torch.roll(correct, shifts=1, dims=1)
    removed = correct.mean(dim=(2, 3), keepdim=True).expand_as(correct).clone()
    visual_mask = torch.ones(batch, 12, dtype=torch.bool)
    common = (
        candidate.to(device),
        candidate_mask.to(device),
        previous.to(device),
        visual_mask.to(device),
    )
    return {
        "V0_removed": (common[0], common[1], common[2], removed.to(device), common[3]),
        "V1_mismatch": (common[0], common[1], common[2], mismatch.to(device), common[3]),
        "V2_correct": (common[0], common[1], common[2], correct.to(device), common[3]),
    }


def _forward(
    model: FullContextSetScorer,
    inputs: tuple[Tensor, ...],
    *,
    device: torch.device,
    mixed_precision: bool,
) -> Tensor:
    with torch.inference_mode(), _autocast(device, mixed_precision):
        return cast(Tensor, model(*inputs))


def _profile_flops(
    model: FullContextSetScorer,
    inputs: tuple[Tensor, ...],
    *,
    device: torch.device,
    mixed_precision: bool,
) -> int:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, with_flops=True, record_shapes=True) as profiler:
        output = _forward(
            model,
            inputs,
            device=device,
            mixed_precision=mixed_precision,
        )
        if not torch.isfinite(output).all():
            raise FloatingPointError("scorer output is nonfinite during FLOPs profiling")
    return int(sum(int(event.flops or 0) for event in profiler.key_averages()))


def _latency_ms(
    model: FullContextSetScorer,
    inputs: tuple[Tensor, ...],
    *,
    device: torch.device,
    mixed_precision: bool,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        _forward(model, inputs, device=device, mixed_precision=mixed_precision)
    _synchronize(device)
    values: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = _forward(model, inputs, device=device, mixed_precision=mixed_precision)
        _synchronize(device)
        elapsed = (time.perf_counter() - started) * 1000.0
        if not torch.isfinite(output).all():
            raise FloatingPointError("scorer output is nonfinite during latency rehearsal")
        values.append(elapsed)
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError("warmup must be positive and repeats must be at least two")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)

    models = {name: _model(args.seed, device) for name in ARM_NAMES}
    schemas = {name: _parameter_schema(model) for name, model in models.items()}
    reference_schema = schemas["V2_correct"]
    if any(schema != reference_schema for schema in schemas.values()):
        raise ValueError("arm parameter schemas differ")
    parameter_counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    if len(set(parameter_counts.values())) != 1:
        raise ValueError("arm parameter counts differ")

    inputs = _synthetic_inputs(device, args.seed)
    outputs: dict[str, Tensor] = {}
    flops: dict[str, int] = {}
    latency: dict[str, dict[str, float]] = {}
    peak_memory: dict[str, int] = {}
    for name in ARM_NAMES:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        first = _forward(
            models[name],
            inputs[name],
            device=device,
            mixed_precision=args.mixed_precision,
        )
        second = _forward(
            models[name],
            inputs[name],
            device=device,
            mixed_precision=args.mixed_precision,
        )
        if not torch.equal(first, second):
            raise ValueError(f"{name} inference replay is not bit exact")
        if not torch.equal(first[:, 0], torch.zeros_like(first[:, 0])):
            raise ValueError(f"{name} native margin is not exact zero")
        outputs[name] = first.detach().cpu()
        flops[name] = _profile_flops(
            models[name],
            inputs[name],
            device=device,
            mixed_precision=args.mixed_precision,
        )
        values = _latency_ms(
            models[name],
            inputs[name],
            device=device,
            mixed_precision=args.mixed_precision,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        latency[name] = {
            "p50_ms": float(np.quantile(values, 0.50)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "mean_ms": float(np.mean(values)),
            "maximum_ms": float(np.max(values)),
        }
        peak_memory[name] = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
    minimum_flops = min(flops.values())
    if minimum_flops <= 0:
        raise ValueError("profiler did not report positive FLOPs")
    flop_ratio = max(flops.values()) / minimum_flops
    parameter_ratio = max(parameter_counts.values()) / min(parameter_counts.values())
    if not math.isclose(parameter_ratio, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("parameter ratio is not exactly one")
    if flop_ratio > 1.01:
        raise ValueError("arm forward FLOPs ratio exceeds 1.01")

    scorer_source = Path(FullContextSetScorer.__module__.replace(".", "/"))
    source_file = Path(__file__).resolve().parents[1] / "src" / scorer_source.with_suffix(".py")
    payload: dict[str, Any] = {
        "schema": "aorta.r005e2.full_context_scorer.preflight.v1",
        "status": "PASS",
        "scientific_metrics_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
        "device": str(device),
        "mixed_precision": bool(args.mixed_precision),
        "seed": args.seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scorer_source_sha256": sha256_file(source_file),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "batch_size": 1,
        "candidate_count": 7,
        "trajectory_sample_count": 41,
        "visual_shape": [1, 12, 18, 32, 1280],
        "visual_token_count": 12 * 18 * 32,
        "parameter_counts": parameter_counts,
        "parameter_ratio": parameter_ratio,
        "flops": flops,
        "flops_ratio": flop_ratio,
        "latency_scorer_only": latency,
        "peak_memory_bytes": peak_memory,
        "inference_replay_bit_exact": True,
        "native_margin_exact_zero": True,
        "all_outputs_finite": all(torch.isfinite(output).all() for output in outputs.values()),
        "claim_boundary": (
            "Synthetic scorer-only compute qualification; excludes frozen DINO backbone, "
            "data loading, scientific training, visual recoverability, and driving utility."
        ),
    }
    _write_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
