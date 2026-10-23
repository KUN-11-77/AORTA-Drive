"""Train and select the sole registered fresh fit-only R004H1 Stage-P model."""

from __future__ import annotations

import argparse
import csv
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

from aorta_drive.certification.heterogeneous import deterministic_group_roles
from aorta_drive.models.quotient_field import (
    QuotientFieldPredictor,
    oriented_grid_gradients,
    quotient_gradient_loss,
)
from aorta_drive.training.r004ec import (
    JoinedAnchor,
    deterministic_interleaved_epoch,
    learning_rate_at_update,
    load_joined_anchors,
    load_training_anchor,
    sample_dense_features_torch,
)
from aorta_drive.training.r004h import (
    StagePCheckpointCandidate,
    qualifies_stage_p_signal,
    select_stage_p_checkpoint,
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


def _read_roles(path: Path) -> dict[str, tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"recording_group", "role", "historical_exposure"}
    if not rows or set(rows[0]) != required:
        raise ValueError("role manifest columns disagree with the registered contract")
    if len(rows) != 66 or len({row["recording_group"] for row in rows}) != 66:
        raise ValueError("role manifest must contain 66 unique groups")
    if {row["historical_exposure"] for row in rows} != {
        "previously_used_by_v8_3_training_or_method_development"
    }:
        raise ValueError("every role must disclose the frozen historical exposure")
    roles = {
        role: tuple(row["recording_group"] for row in rows if row["role"] == role)
        for role in ("fit", "selection", "proxy_cal")
    }
    if any(row["role"] not in roles for row in rows):
        raise ValueError("role manifest contains an unknown role")
    if tuple(len(roles[role]) for role in ("fit", "selection", "proxy_cal")) != (
        42,
        8,
        16,
    ):
        raise ValueError("role counts disagree with the registered contract")
    return roles


def _source_train_groups(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = tuple(row["recording_group"] for row in rows if row["split"] == "train")
    if len(groups) != 66 or len(set(groups)) != 66:
        raise ValueError("source manifest must contain 66 unique train groups")
    return groups


def _selection_metrics(
    model: QuotientFieldPredictor | None,
    records: Sequence[JoinedAnchor],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    maxima: dict[str, float] = defaultdict(float)
    if model is not None:
        model.eval()
    with torch.no_grad():
        for record in records:
            features, mask, labels = _load_tensors(record, device)
            prediction = torch.zeros_like(labels) if model is None else model(features, mask)
            residual = torch.abs(
                oriented_grid_gradients(prediction) - oriented_grid_gradients(labels)
            )
            if not bool(torch.isfinite(residual).all().item()):
                raise FloatingPointError("nonfinite Stage-P selection residual")
            value_max = float(residual.max().item())
            sums[record.recording_group] += float(
                residual.sum(dtype=torch.float64).item()
            )
            counts[record.recording_group] += residual.numel()
            maxima[record.recording_group] = max(
                maxima[record.recording_group], value_max
            )
    if len(sums) != 8 or any(count != 16 * 8064 for count in counts.values()):
        raise ValueError("selection evaluation must contain 8 complete groups")
    means = {group: sums[group] / counts[group] for group in sorted(sums)}
    ordered_maxima = {group: maxima[group] for group in sorted(maxima)}
    return means, ordered_maxima


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
    role_manifest_sha256: str,
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
            "role_manifest_sha256": role_manifest_sha256,
            "selection_key": selection_key,
            "fresh_random_initialization": True,
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


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--recording-group-manifest", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/r004h1_stage_p.yaml")
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    optimization = config["optimization"]
    expected_hashes = {
        "recording_group_manifest_sha256": sha256_file(args.recording_group_manifest),
        "role_manifest_sha256": sha256_file(args.role_manifest),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "label_manifest_sha256": sha256_file(args.label_manifest),
    }
    for name, actual in expected_hashes.items():
        if actual != inputs[name]:
            raise ValueError(f"{name} disagrees with the frozen H1 config")

    roles = _read_roles(args.role_manifest)
    regenerated = deterministic_group_roles(
        _source_train_groups(args.recording_group_manifest),
        namespace="AORTA-R004EG-roles-v1",
        fit_count=42,
        selection_count=8,
        proxy_cal_count=16,
    )
    if roles["fit"] != regenerated.fit:
        raise ValueError("fit role order is not a deterministic replay")
    if roles["selection"] != regenerated.selection:
        raise ValueError("selection role order is not a deterministic replay")
    if roles["proxy_cal"] != regenerated.proxy_cal:
        raise ValueError("proxy-cal role order is not a deterministic replay")

    train_records = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
        allowed_splits=("train",),
    )
    by_group: dict[str, list[JoinedAnchor]] = defaultdict(list)
    for record in train_records:
        by_group[record.recording_group].append(record)
    if set(by_group) != set(roles["fit"] + roles["selection"] + roles["proxy_cal"]):
        raise ValueError("joined train groups disagree with the frozen role ledger")
    if any(len(records) != 16 for records in by_group.values()):
        raise ValueError("every role group must contain exactly 16 anchors")
    fit_records = tuple(record for group in roles["fit"] for record in by_group[group])
    selection_records = tuple(
        record for group in roles["selection"] for record in by_group[group]
    )
    if len(fit_records) != 672 or len(selection_records) != 128:
        raise ValueError("fit/selection anchor counts disagree with registration")

    seed = int(optimization["seed"])
    maximum_epochs = int(optimization["maximum_epochs"])
    warmup_epochs = int(optimization["warmup_epochs"])
    total_updates = maximum_epochs * len(fit_records)
    warmup_updates = warmup_epochs * len(fit_records)
    base_learning_rate = float(optimization["base_learning_rate"])
    betas_raw = optimization["betas"]
    if not isinstance(betas_raw, list) or len(betas_raw) != 2:
        raise ValueError("H1 AdamW betas must contain exactly two values")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("R004H1 requires the registered CUDA device")
    torch.cuda.init()  # type: ignore[no-untyped-call]
    model = QuotientFieldPredictor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_learning_rate,
        betas=(float(betas_raw[0]), float(betas_raw[1])),
        weight_decay=float(optimization["weight_decay"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["model"]["parameter_count"]):
        raise ValueError("model parameter count disagrees with the frozen H1 config")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    result_path = args.output_directory / "result.json"
    latest_path = args.output_directory / "latest.pt"
    best_path = args.output_directory / "best.pt"
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    zero_means, zero_maxima = _selection_metrics(None, selection_records, device)
    zero_primary = sum(zero_means.values()) / 8
    zero_worst_two = sum(sorted(zero_maxima.values(), reverse=True)[:2]) / 2
    epoch_ledger: list[dict[str, object]] = []
    candidates: list[StagePCheckpointCandidate] = []
    update_count = 0
    best_key: tuple[float, float, int] | None = None

    objective = config["objective"]
    for epoch_index in range(maximum_epochs):
        epoch_started = time.perf_counter()
        model.train()
        train_totals: list[float] = []
        train_hubers: list[float] = []
        train_tails: list[float] = []
        train_maximum = 0.0
        gradient_norm_maximum = 0.0
        schedule = deterministic_interleaved_epoch(
            fit_records,
            seed=seed,
            epoch=epoch_index,
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
            loss = quotient_gradient_loss(
                prediction,
                labels,
                edge_scale=float(objective["edge_scale_m"]),
                huber_delta_m=float(objective["huber_delta_m"]),
                tail_temperature_m=float(objective["tail_temperature_m"]),
                tail_weight=float(objective["tail_weight"]),
            )
            torch.autograd.backward(loss.total)
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or not all(
                bool(torch.isfinite(gradient).all().item()) for gradient in gradients
            ):
                raise FloatingPointError("nonfinite H1 gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(optimization["gradient_clip_global_norm"]),
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
                raise FloatingPointError("nonfinite H1 training scalar")
            optimizer.step()
            update_count += 1
            train_totals.append(scalar_values[0])
            train_hubers.append(scalar_values[1])
            train_tails.append(scalar_values[2])
            train_maximum = max(train_maximum, scalar_values[3])
            gradient_norm_maximum = max(gradient_norm_maximum, scalar_values[4])

        selection_means, selection_maxima = _selection_metrics(
            model,
            selection_records,
            device,
        )
        candidate = StagePCheckpointCandidate(
            epoch=epoch_index + 1,
            group_mean_edge_mae_m=tuple(selection_means.values()),
            group_max_residual_m=tuple(selection_maxima.values()),
        )
        candidates.append(candidate)
        selected_so_far = select_stage_p_checkpoint(candidates)
        current = select_stage_p_checkpoint((candidate,))
        current_key = (
            current.macro_group_mean_edge_mae_m,
            current.worst_two_group_max_mean_m,
            current.epoch,
        )
        epoch_record: dict[str, object] = {
            "epoch": epoch_index + 1,
            "updates_completed": update_count,
            "learning_rate_first": learning_rate_at_update(
                epoch_index * len(fit_records),
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
            "selection_group_mean_edge_mae_m": selection_means,
            "selection_group_max_residual_m": selection_maxima,
            "selection_macro_group_mean_edge_mae_m": current.macro_group_mean_edge_mae_m,
            "selection_worst_two_group_max_mean_m": current.worst_two_group_max_mean_m,
            "selected_epoch_so_far": selected_so_far.epoch,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        epoch_ledger.append(epoch_record)
        _save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            completed_epoch=epoch_index + 1,
            update_count=update_count,
            config_sha256=sha256_file(args.config),
            feature_manifest_sha256=expected_hashes["feature_manifest_sha256"],
            label_manifest_sha256=expected_hashes["label_manifest_sha256"],
            role_manifest_sha256=expected_hashes["role_manifest_sha256"],
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
                config_sha256=sha256_file(args.config),
                feature_manifest_sha256=expected_hashes["feature_manifest_sha256"],
                label_manifest_sha256=expected_hashes["label_manifest_sha256"],
                role_manifest_sha256=expected_hashes["role_manifest_sha256"],
                selection_key=current_key,
            )
        elapsed = time.perf_counter() - started
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        output_gib = _directory_bytes(args.output_directory) / 1024**3
        if peak_gib > float(config["resources"]["maximum_peak_gpu_gib"]):
            raise RuntimeError("H1 peak GPU allocation exceeded the registered cap")
        if elapsed > float(config["resources"]["maximum_wall_minutes"]) * 60.0:
            raise RuntimeError("H1 wall time exceeded the registered cap")
        if output_gib > float(config["resources"]["maximum_output_gib"]):
            raise RuntimeError("H1 output size exceeded the registered cap")
        progress: dict[str, object] = {
            "experiment_id": config["experiment_id"],
            "status": "IN_PROGRESS",
            "implementation_revision": args.implementation_revision,
            "config_sha256": sha256_file(args.config),
            **expected_hashes,
            "seed": seed,
            "completed_epochs": epoch_index + 1,
            "maximum_epochs": maximum_epochs,
            "selected_epoch_so_far": selected_so_far.epoch,
            "zero_field_selection_primary_m": zero_primary,
            "zero_field_selection_worst_two_m": zero_worst_two,
            "epoch_ledger": epoch_ledger,
            "prohibited_payload_accessed": False,
            "external_checkpoint_loaded": False,
        }
        _write_json(result_path, progress)
        print(
            json.dumps(
                {
                    "epoch": epoch_index + 1,
                    "train_mean_objective": epoch_record["train_mean_objective"],
                    "selection_primary_m": current.macro_group_mean_edge_mae_m,
                    "selection_secondary_m": current.worst_two_group_max_mean_m,
                    "selected_epoch_so_far": selected_so_far.epoch,
                    "epoch_seconds": epoch_record["epoch_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selected = select_stage_p_checkpoint(candidates)
    if best_key is None or selected.epoch != best_key[2]:
        raise AssertionError("persisted best checkpoint disagrees with frozen H1 selector")
    passes_signal_gate = qualifies_stage_p_signal(
        learned_mean_m=selected.macro_group_mean_edge_mae_m,
        zero_mean_m=zero_primary,
    )
    elapsed = time.perf_counter() - started
    final_result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": "PASS" if passes_signal_gate else "STOP",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        **expected_hashes,
        "seed": seed,
        "fresh_random_initialization": True,
        "external_checkpoint_loaded": False,
        "parameter_count": parameter_count,
        "fit_group_count": 42,
        "selection_group_count": 8,
        "fit_anchor_count_per_epoch": len(fit_records),
        "selection_anchor_count_per_epoch": len(selection_records),
        "completed_epochs": maximum_epochs,
        "optimizer_updates": update_count,
        "selected_epoch": selected.epoch,
        "selected_primary_m": selected.macro_group_mean_edge_mae_m,
        "selected_secondary_m": selected.worst_two_group_max_mean_m,
        "zero_field_selection_primary_m": zero_primary,
        "zero_field_selection_worst_two_m": zero_worst_two,
        "selection_signal_gate_passed": passes_signal_gate,
        "epoch_ledger": epoch_ledger,
        "best_checkpoint_sha256": sha256_file(best_path),
        "latest_checkpoint_sha256": sha256_file(latest_path),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": elapsed,
        "output_bytes": _directory_bytes(args.output_directory),
        "prohibited_payload_accessed": False,
        "r004h2_registration_authorized": passes_signal_gate,
        "claim_boundary": (
            "Selection-side fresh-potential central-signal qualification only. No scale, "
            "proxy, development, official calibration, coverage, task utility, Gate B, "
            "OOD robustness, or novelty claim is established."
        ),
    }
    _write_json(result_path, final_result)
    print(json.dumps(final_result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
