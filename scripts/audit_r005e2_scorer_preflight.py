"""Independent contract audit for the R005E2 synthetic scorer preflight."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file


ARMS = {"V0_removed", "V1_mismatch", "V2_correct"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    result_hash = sha256_file(args.result)
    if args.expected_result_sha256 and result_hash != args.expected_result_sha256:
        raise ValueError("scorer preflight result hash mismatch")
    result = cast(dict[str, Any], json.loads(args.result.read_text(encoding="utf-8")))
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    if result.get("status") != "PASS" or result.get("scientific_metrics_emitted") is not False:
        raise ValueError("preflight status or scientific-metric boundary is invalid")
    if result.get("development_calibration_test_navdream_target_accessed") is not False:
        raise ValueError("sealed role accessed during scorer preflight")
    probe = cast(dict[str, Any], config["probe"])
    training = cast(dict[str, Any], config["training"])
    expected_shape = [1, *cast(list[int], probe["visual_input_shape"])]
    if result["visual_shape"] != expected_shape:
        raise ValueError("visual shape differs from the frozen config")
    if int(result["visual_token_count"]) != int(probe["visual_token_count"]):
        raise ValueError("visual token count differs from the frozen config")
    if result["cublas_workspace_config"] != training["cublas_workspace_config"]:
        raise ValueError("CuBLAS deterministic workspace contract drift")
    if result.get("mixed_precision") is not True or training["mixed_precision"] != "bfloat16":
        raise ValueError("mixed-precision contract drift")

    counts = cast(dict[str, int], result["parameter_counts"])
    flops = cast(dict[str, int], result["flops"])
    latency = cast(dict[str, dict[str, float]], result["latency_scorer_only"])
    memory = cast(dict[str, int], result["peak_memory_bytes"])
    if not (set(counts) == set(flops) == set(latency) == set(memory) == ARMS):
        raise ValueError("arm ledger mismatch")
    if len(set(counts.values())) != 1 or float(result["parameter_ratio"]) != 1.0:
        raise ValueError("parameter matching failed")
    if min(flops.values()) <= 0:
        raise ValueError("FLOPs must be positive")
    recomputed_flop_ratio = max(flops.values()) / min(flops.values())
    if not math.isclose(recomputed_flop_ratio, float(result["flops_ratio"]), abs_tol=0.0):
        raise ValueError("FLOPs ratio recomputation mismatch")
    if recomputed_flop_ratio > float(config["arms"]["measured_forward_FLOPs_max_ratio"]):
        raise ValueError("FLOPs ratio exceeds the frozen gate")
    for arm in ARMS:
        if memory[arm] <= 0:
            raise ValueError("GPU peak memory must be positive")
        for name in ("p50_ms", "p95_ms", "mean_ms", "maximum_ms"):
            value = float(latency[arm][name])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("latency ledger must be finite and positive")
        if latency[arm]["p95_ms"] > latency[arm]["maximum_ms"]:
            raise ValueError("latency quantiles are inconsistent")
    for field in ("inference_replay_bit_exact", "native_margin_exact_zero", "all_outputs_finite"):
        if result.get(field) is not True:
            raise ValueError(f"required scorer invariant failed: {field}")

    scorer_source = args.repository / "src/aorta_drive/models/full_context_scorer.py"
    runner_source = args.repository / "scripts/preflight_r005e2_scorer.py"
    if result["scorer_source_sha256"] != sha256_file(scorer_source):
        raise ValueError("scorer runtime source hash mismatch")
    if result["runner_source_sha256"] != sha256_file(runner_source):
        raise ValueError("preflight runner source hash mismatch")
    return {
        "schema": "aorta.r005e2.full_context_scorer.preflight.audit.v1",
        "status": "PASS",
        "result_sha256": result_hash,
        "config_sha256": sha256_file(args.config),
        "runtime_source_hashes_exact": True,
        "sealed_roles_unaccessed": True,
        "scientific_metrics_emitted": False,
        "visual_shape_exact": True,
        "visual_token_count_exact": True,
        "parameter_counts_exact": True,
        "parameter_ratio": 1.0,
        "flops_positive": True,
        "flops_ratio": recomputed_flop_ratio,
        "flops_gate": float(config["arms"]["measured_forward_FLOPs_max_ratio"]),
        "deterministic_inference_replay": True,
        "native_margin_exact_zero": True,
        "all_outputs_finite": True,
        "claim_boundary": (
            "Synthetic scorer-only architecture/compute/resource audit; no DINO latency, "
            "data loading, scientific training, visual recoverability, or driving claim."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = audit(arguments)
    _write_json(arguments.output, output)
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
