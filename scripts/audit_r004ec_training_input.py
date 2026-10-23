"""C0 audit: qualify one real R004E-C feature/label pair through backward."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.models.quotient_field import (
    QuotientFieldPredictor,
    oriented_grid_gradients,
    quotient_gradient_loss,
)
from aorta_drive.training.r004ec import (
    load_joined_anchors,
    load_training_anchor,
    sample_dense_features_torch,
)
from aorta_drive.vision.feature_cache import bilinear_sample_dense_numpy, sha256_file


def _sha256_text(path: Path) -> str:
    return sha256_file(path)


def _hash_named_files(paths: tuple[Path, ...]) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/r004ec_training.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    threshold = config["stages"]["C0_real_input_contract"]["acceptance"]
    expected_hashes = config["frozen_inputs"]
    feature_manifest_hash = _sha256_text(args.feature_manifest)
    label_manifest_hash = _sha256_text(args.label_manifest)
    if feature_manifest_hash != expected_hashes["feature_manifest_sha256"]:
        raise ValueError("feature manifest hash disagrees with frozen C0 config")
    if label_manifest_hash != expected_hashes["label_manifest_sha256"]:
        raise ValueError("label manifest hash disagrees with frozen C0 config")

    records = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
        allowed_splits=("train",),
    )
    selected = records[0]
    if selected.anchor_index != 0:
        raise ValueError("lexicographically first frozen train group must contain anchor 0")
    started = time.perf_counter()
    loaded = load_training_anchor(selected)
    if sha256_file(selected.label_path) != selected.label_artifact_sha256:
        raise ValueError("selected label artifact hash mismatch")

    numpy_features = bilinear_sample_dense_numpy(
        loaded.dense_features, loaded.patch_xy, loaded.mask
    )
    device = torch.device(args.device)
    torch.manual_seed(20261023)
    torch.use_deterministic_algorithms(True)
    dense = torch.from_numpy(loaded.dense_features).to(device)
    patch_xy = torch.from_numpy(loaded.patch_xy).to(device)
    mask = torch.from_numpy(loaded.mask).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    materialized = sample_dense_features_torch(dense, patch_xy, mask)
    oracle_error = float(
        np.max(np.abs(materialized.detach().cpu().numpy() - numpy_features))
    )
    invalid_exact_zero = bool(torch.count_nonzero(materialized[~mask]).item() == 0)

    model = QuotientFieldPredictor().to(device)
    model.eval()
    features = materialized.unsqueeze(0)
    masks = mask.unsqueeze(0)
    labels = torch.from_numpy(loaded.label_m).to(device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(features, masks)
        replay = model(features, masks)
        invalid_variant = torch.where(
            masks.unsqueeze(-1), features, torch.full_like(features, 1000.0)
        )
        invariant = model(invalid_variant, masks)
    replay_error = float(torch.max(torch.abs(prediction - replay)).item())
    invalid_storage_error = float(torch.max(torch.abs(prediction - invariant)).item())
    zero_gauge_error = float(torch.max(torch.abs(prediction.mean(dim=(-2, -1)))).item())

    model.train()
    model.zero_grad(set_to_none=True)
    training_prediction = model(features, masks)
    loss = quotient_gradient_loss(training_prediction, labels)
    torch.autograd.backward(loss.total)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    nonzero_gradient_tensors = sum(
        int(bool(torch.count_nonzero(gradient).item())) for gradient in gradients
    )
    edge_count = int(oriented_grid_gradients(labels).shape[-1])
    values_finite = all(
        bool(torch.isfinite(value).all().item())
        for value in (materialized, prediction, training_prediction, loss.total)
    )
    torch.cuda.synchronize(device)
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.perf_counter() - started
    passed = (
        oracle_error <= float(threshold["sampler_max_absolute_error_at_most"])
        and zero_gauge_error
        <= float(threshold["output_zero_gauge_max_absolute_mean_m_at_most"])
        and replay_error
        == float(threshold["deterministic_replay_max_absolute_error_m"])
        and invalid_storage_error
        == float(threshold["invalid_slot_storage_max_absolute_error_m"])
        and edge_count == int(threshold["edge_count"])
        and invalid_exact_zero
        and values_finite
        and finite_gradients
        and nonzero_gradient_tensors > 0
    )
    result = {
        "experiment_id": "R004E-C0-real-training-input-contract",
        "status": "PASS" if passed else "FAIL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": _sha256_text(args.config),
        "script_sha256": _sha256_text(Path(__file__)),
        "feature_manifest_sha256": feature_manifest_hash,
        "label_manifest_sha256": label_manifest_hash,
        "selected_anchor": {
            "recording_group": selected.recording_group,
            "anchor_index": selected.anchor_index,
            "log_name": selected.log_name,
            "scene_token": selected.scene_token,
        },
        "selected_feature_artifact_sha256": _hash_named_files(
            selected.dense_feature_paths
        ),
        "selected_projection_sha256": _hash_named_files(
            (selected.projection_xy_path, selected.projection_mask_path)
        ),
        "selected_label_artifact_sha256": sha256_file(selected.label_path),
        "sampler_max_absolute_error": oracle_error,
        "invalid_sampled_values_exact_zero": invalid_exact_zero,
        "materialized_shape": list(materialized.shape),
        "label_shape": list(labels.shape),
        "edge_count": edge_count,
        "zero_gauge_max_absolute_mean_m": zero_gauge_error,
        "deterministic_replay_max_absolute_error_m": replay_error,
        "invalid_slot_storage_max_absolute_error_m": invalid_storage_error,
        "loss": {
            "total": float(loss.total.item()),
            "mean_huber": float(loss.mean_huber.item()),
            "centered_smooth_max": float(loss.centered_smooth_max.item()),
            "maximum_residual_m": float(loss.maximum_residual.item()),
        },
        "values_finite": values_finite,
        "finite_gradients": finite_gradients,
        "nonzero_gradient_tensor_count": nonzero_gradient_tensors,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": peak_cuda_bytes,
        "elapsed_seconds": elapsed,
        "training_updates": 0,
        "calibration_test_or_target_accessed": False,
        "claim_boundary": (
            "Qualifies one real frozen feature/label pair through runtime sampling, "
            "prediction, loss, and backward only. It establishes no learned accuracy, "
            "generalization, calibration, certificate utility, OOD robustness, or Gate B pass."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
