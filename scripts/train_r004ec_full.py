"""C2: train the sole registered source-only quotient-field probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.models.quotient_field import (
    DevelopmentCheckpoint,
    QuotientFieldPredictor,
    quotient_gradient_loss,
    select_development_checkpoint,
)
from aorta_drive.training.r004ec import (
    JoinedAnchor,
    deterministic_interleaved_epoch,
    learning_rate_at_update,
    load_joined_anchors,
    load_training_anchor,
    sample_dense_features_torch,
)
from aorta_drive.vision.feature_cache import sha256_file


def _load_tensors(record: JoinedAnchor, device: torch.device) -> tuple[torch.Tensor, ...]:
    loaded = load_training_anchor(record)
    dense = torch.from_numpy(loaded.dense_features).to(device)
    patch_xy = torch.from_numpy(loaded.patch_xy).to(device)
    mask = torch.from_numpy(loaded.mask).to(device)
    features = sample_dense_features_torch(dense, patch_xy, mask).unsqueeze(0)
    labels = torch.from_numpy(loaded.label_m).to(device=device, dtype=torch.float32).unsqueeze(0)
    return features, mask.unsqueeze(0), labels


def _development_group_maxima(
    model: QuotientFieldPredictor,
    records: Sequence[JoinedAnchor],
    device: torch.device,
) -> dict[str, float]:
    maxima: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for record in records:
            features, mask, labels = _load_tensors(record, device)
            prediction = model(features, mask)
            loss = quotient_gradient_loss(prediction, labels)
            residual = float(loss.maximum_residual.item())
            if not math.isfinite(residual) or residual < 0.0:
                raise FloatingPointError("nonfinite development residual")
            maxima[record.recording_group] = max(maxima[record.recording_group], residual)
            counts[record.recording_group] += 1
    if len(maxima) != 24 or any(count != 16 for count in counts.values()):
        raise ValueError("development evaluation must contain 24 complete groups")
    return dict(sorted(maxima.items()))


def _save_checkpoint(
    path: Path,
    *,
    model: QuotientFieldPredictor,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    update_count: int,
    config_sha256: str,
    feature_manifest_sha256: str,
    label_manifest_sha256: str,
    selection_key: tuple[float, float, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_epoch": completed_epoch,
            "update_count": update_count,
            "config_sha256": config_sha256,
            "feature_manifest_sha256": feature_manifest_sha256,
            "label_manifest_sha256": label_manifest_sha256,
            "selection_key": selection_key,
        },
        temporary,
    )
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/r004ec2_full_training.yaml")
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    optimization = config["optimization"]
    config_hash = sha256_file(args.config)
    feature_hash = sha256_file(args.feature_manifest)
    label_hash = sha256_file(args.label_manifest)
    if feature_hash != inputs["feature_manifest_sha256"]:
        raise ValueError("feature manifest hash disagrees with frozen C2 config")
    if label_hash != inputs["label_manifest_sha256"]:
        raise ValueError("label manifest hash disagrees with frozen C2 config")
    all_records = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
    )
    train_records = tuple(record for record in all_records if record.split == "train")
    development_records = tuple(
        record for record in all_records if record.split == "development"
    )
    train_group_count = len({record.recording_group for record in train_records})
    development_group_count = len(
        {record.recording_group for record in development_records}
    )
    if len(train_records) != 1056 or train_group_count != 66:
        raise ValueError("C2 requires 66 complete train groups and 1056 anchors")
    if len(development_records) != 384 or development_group_count != 24:
        raise ValueError("C2 requires 24 complete development groups and 384 anchors")

    seed = int(optimization["seed"])
    maximum_epochs = int(optimization["maximum_epochs"])
    warmup_epochs = int(optimization["warmup_epochs"])
    total_updates = maximum_epochs * len(train_records)
    warmup_updates = warmup_epochs * len(train_records)
    base_learning_rate = float(optimization["base_learning_rate"])
    betas_raw = optimization["betas"]
    if not isinstance(betas_raw, list) or len(betas_raw) != 2:
        raise ValueError("C2 AdamW betas must contain exactly two values")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = QuotientFieldPredictor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_learning_rate,
        betas=(float(betas_raw[0]), float(betas_raw[1])),
        weight_decay=float(optimization["weight_decay"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["model"]["parameter_count"]):
        raise ValueError("model parameter count disagrees with frozen C2 config")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    result_path = args.output_directory / "result.json"
    latest_path = args.output_directory / "latest.pt"
    best_path = args.output_directory / "best.pt"
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    epoch_ledger: list[dict[str, object]] = []
    candidates: list[DevelopmentCheckpoint] = []
    update_count = 0
    best_key: tuple[float, float, int] | None = None

    for epoch_index in range(maximum_epochs):
        epoch_started = time.perf_counter()
        model.train()
        train_totals: list[float] = []
        train_hubers: list[float] = []
        train_tails: list[float] = []
        train_maximum = 0.0
        gradient_norm_maximum = 0.0
        schedule = deterministic_interleaved_epoch(
            train_records, seed=seed, epoch=epoch_index
        )
        for record in schedule:
            learning_rate = learning_rate_at_update(
                update_count,
                total_updates=total_updates,
                warmup_updates=warmup_updates,
                base_learning_rate=base_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            features, mask, labels = _load_tensors(record, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, mask)
            loss = quotient_gradient_loss(prediction, labels)
            torch.autograd.backward(loss.total)
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or not all(
                bool(torch.isfinite(gradient).all().item()) for gradient in gradients
            ):
                raise FloatingPointError("nonfinite C2 gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optimization["gradient_clip_global_norm"])
            )
            scalar_values = (
                float(loss.total.item()),
                float(loss.mean_huber.item()),
                float(loss.centered_smooth_max.item()),
                float(loss.maximum_residual.item()),
                float(gradient_norm.item()),
                learning_rate,
            )
            if any(not math.isfinite(value) for value in scalar_values):
                raise FloatingPointError("nonfinite C2 training scalar")
            optimizer.step()
            update_count += 1
            train_totals.append(scalar_values[0])
            train_hubers.append(scalar_values[1])
            train_tails.append(scalar_values[2])
            train_maximum = max(train_maximum, scalar_values[3])
            gradient_norm_maximum = max(gradient_norm_maximum, scalar_values[4])

        development = _development_group_maxima(model, development_records, device)
        candidate = DevelopmentCheckpoint(
            epoch=epoch_index + 1,
            session_max_residuals_m=tuple(development.values()),
        )
        candidates.append(candidate)
        selection = select_development_checkpoint(candidates)
        current_values = sorted(development.values(), reverse=True)
        current_tail = sum(current_values[:5]) / 5
        current_mean = sum(current_values) / 24
        current_key = (current_tail, current_mean, epoch_index + 1)
        epoch_record: dict[str, object] = {
            "epoch": epoch_index + 1,
            "updates_completed": update_count,
            "learning_rate_first": learning_rate_at_update(
                epoch_index * len(train_records),
                total_updates=total_updates,
                warmup_updates=warmup_updates,
                base_learning_rate=base_learning_rate,
            ),
            "learning_rate_last": learning_rate_at_update(
                update_count - 1,
                total_updates=total_updates,
                warmup_updates=warmup_updates,
                base_learning_rate=base_learning_rate,
            ),
            "train_mean_objective": sum(train_totals) / len(train_totals),
            "train_mean_huber": sum(train_hubers) / len(train_hubers),
            "train_mean_centered_smooth_max": sum(train_tails) / len(train_tails),
            "train_anchor_max_residual_m": train_maximum,
            "preclip_gradient_norm_maximum": gradient_norm_maximum,
            "development_group_max_residuals_m": development,
            "development_worst5_mean_m": current_tail,
            "development_all24_mean_m": current_mean,
            "selected_epoch_so_far": selection.epoch,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        epoch_ledger.append(epoch_record)
        _save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            completed_epoch=epoch_index + 1,
            update_count=update_count,
            config_sha256=config_hash,
            feature_manifest_sha256=feature_hash,
            label_manifest_sha256=label_hash,
            selection_key=current_key,
        )
        if best_key is None or current_key < best_key:
            best_key = current_key
            _save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                completed_epoch=epoch_index + 1,
                update_count=update_count,
                config_sha256=config_hash,
                feature_manifest_sha256=feature_hash,
                label_manifest_sha256=label_hash,
                selection_key=current_key,
            )
        progress: dict[str, object] = {
            "experiment_id": "R004E-C2-full-source-only-training",
            "status": "IN_PROGRESS",
            "implementation_revision": args.implementation_revision,
            "config_sha256": config_hash,
            "feature_manifest_sha256": feature_hash,
            "label_manifest_sha256": label_hash,
            "seed": seed,
            "completed_epochs": epoch_index + 1,
            "maximum_epochs": maximum_epochs,
            "selected_epoch_so_far": selection.epoch,
            "epoch_ledger": epoch_ledger,
            "calibration_test_or_target_accessed": False,
        }
        _write_json(result_path, progress)
        print(
            json.dumps(
                {
                    "epoch": epoch_index + 1,
                    "train_mean_objective": epoch_record["train_mean_objective"],
                    "development_worst5_mean_m": current_tail,
                    "development_all24_mean_m": current_mean,
                    "selected_epoch_so_far": selection.epoch,
                    "epoch_seconds": epoch_record["epoch_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selected = select_development_checkpoint(candidates)
    if best_key is None or selected.epoch != best_key[2]:
        raise AssertionError("persisted best checkpoint disagrees with frozen selector")
    final_result: dict[str, object] = {
        "experiment_id": "R004E-C2-full-source-only-training",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": config_hash,
        "script_sha256": sha256_file(Path(__file__)),
        "feature_manifest_sha256": feature_hash,
        "label_manifest_sha256": label_hash,
        "seed": seed,
        "parameter_count": parameter_count,
        "train_group_count": train_group_count,
        "development_group_count": development_group_count,
        "train_anchor_count": len(train_records),
        "development_anchor_count": len(development_records),
        "completed_epochs": maximum_epochs,
        "optimizer_updates": update_count,
        "selected_epoch": selected.epoch,
        "selected_development_worst5_mean_m": selected.tail_mean_m,
        "selected_development_all24_mean_m": selected.mean_m,
        "epoch_ledger": epoch_ledger,
        "best_checkpoint_sha256": sha256_file(best_path),
        "latest_checkpoint_sha256": sha256_file(latest_path),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_test_or_target_accessed": False,
        "claim_boundary": (
            "Closes one registered train/development optimization and checkpoint-selection "
            "run only. It establishes no held-out test accuracy, calibration coverage, "
            "certificate nontriviality, driving utility, OOD robustness, or Gate B pass."
        ),
    }
    _write_json(result_path, final_result)
    print(json.dumps(final_result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
