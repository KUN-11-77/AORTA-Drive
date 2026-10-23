"""Resource-scale structural audit for the frozen R004D predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Required by torch deterministic algorithms for CUDA >= 10.2. This must be
# present before the first cuBLAS operation, so the audit owns the setting.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from aorta_drive.models.quotient_field import QuotientFieldPredictor, quotient_gradient_loss


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = QuotientFieldPredictor().to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    features = torch.randn(
        1, 4096, 12, 1280, generator=generator, device=device, dtype=torch.float32
    )
    mask = torch.rand(1, 4096, 12, generator=generator, device=device) > 0.42
    mask[:, :16, :] = False
    features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))
    label = torch.randn(1, 64, 64, generator=generator, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    prediction = model(features, mask)
    replay = model(features, mask)
    forward_seconds = time.perf_counter() - started
    replay_max_error = float(torch.max(torch.abs(prediction - replay)).item())
    zero_gauge_error = float(torch.max(torch.abs(prediction.mean(dim=(-2, -1)))).item())

    invalid_variant = torch.where(
        mask.unsqueeze(-1), features, torch.full_like(features, 1000.0)
    )
    invariant_prediction = model(invalid_variant, mask)
    invalid_storage_max_error = float(
        torch.max(torch.abs(prediction - invariant_prediction)).item()
    )
    loss = quotient_gradient_loss(prediction, label)
    torch.autograd.backward(loss.total)
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    nonzero_gradient_count = sum(
        int(bool(torch.count_nonzero(gradient).item())) for gradient in gradients
    )
    output_finite = bool(torch.isfinite(prediction).all().item())
    passed = (
        tuple(prediction.shape) == (1, 64, 64)
        and output_finite
        and finite_gradients
        and nonzero_gradient_count > 0
        and replay_max_error == 0.0
        and invalid_storage_max_error == 0.0
        and zero_gauge_error <= 1e-6
    )
    result = {
        "experiment_id": "R004D-predictor-structural-audit",
        "status": "PASS" if passed else "FAIL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "script_sha256": _sha256(Path(__file__)),
        "torch_version": torch.__version__,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cuda_device_name": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "input_shape": list(features.shape),
        "mask_shape": list(mask.shape),
        "all_missing_node_count": int((~mask.any(dim=-1)).sum().item()),
        "output_shape": list(prediction.shape),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "output_finite": output_finite,
        "finite_gradients": finite_gradients,
        "nonzero_gradient_tensor_count": nonzero_gradient_count,
        "zero_gauge_max_absolute_mean_m": zero_gauge_error,
        "deterministic_replay_max_absolute_error_m": replay_max_error,
        "invalid_storage_max_absolute_error_m": invalid_storage_max_error,
        "loss": {
            "total": float(loss.total.item()),
            "mean_huber": float(loss.mean_huber.item()),
            "centered_smooth_max": float(loss.centered_smooth_max.item()),
            "maximum_residual": float(loss.maximum_residual.item()),
        },
        "forward_two_pass_seconds": forward_seconds,
        "forward_backward_total_seconds": elapsed_seconds,
        "peak_cuda_bytes": peak_cuda_bytes,
        "training_performed": False,
        "data_used": "deterministic_synthetic_tensor_matching_the_frozen_R004B_shape",
        "calibration_test_or_target_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
