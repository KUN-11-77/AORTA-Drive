"""Train and select the sole registered R004H2 bounded scale head."""

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
from typing import Any, Literal, Sequence, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.heterogeneous import deterministic_group_roles
from aorta_drive.models.heterogeneous_envelope import (
    HeterogeneousEnvelopePredictor,
    bounded_laplace_scale_loss,
)
from aorta_drive.models.quotient_field import QuotientFieldPredictor
from aorta_drive.training.r004ec import (
    JoinedAnchor,
    deterministic_interleaved_epoch,
    learning_rate_at_update,
    load_joined_anchors,
    load_training_anchor,
    sample_dense_features_torch,
)
from aorta_drive.training.r004h import (
    StageSCheckpointCandidate,
    evidence_only_scale_control,
    pooled_scale_control,
    qualifies_scale_caps,
    scale_cap_fractions,
    select_stage_s_checkpoint,
    uniform_scale_control,
)
from aorta_drive.vision.feature_cache import sha256_file


def _checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return cast(dict[str, Any], payload)


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
    model: HeterogeneousEnvelopePredictor,
    records: Sequence[JoinedAnchor],
    device: torch.device,
    *,
    scale_variant: Literal["learned", "uniform"],
    cap_position_tolerance: float,
) -> tuple[dict[str, float], dict[str, float]]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    scale_count = 0
    lower_count = 0.0
    upper_count = 0.0
    scale_min = math.inf
    scale_max = 0.0
    log_scale_sum = 0.0
    model.eval()
    with torch.no_grad():
        for record in records:
            features, mask, labels = _load_tensors(record, device)
            output = model(features, mask)
            scales = (
                output.edge_scales_m
                if scale_variant == "learned"
                else uniform_scale_control(output.edge_scales_m)
            )
            loss = bounded_laplace_scale_loss(output.potential, labels, scales)
            value = float(loss.total.item())
            if not math.isfinite(value):
                raise FloatingPointError("nonfinite Stage-S selection proper loss")
            totals[record.recording_group] += value
            counts[record.recording_group] += 1
            lower, upper = scale_cap_fractions(
                scales,
                cap_position_tolerance=cap_position_tolerance,
            )
            elements = scales.numel()
            scale_count += elements
            lower_count += lower * elements
            upper_count += upper * elements
            scale_min = min(scale_min, float(scales.min().item()))
            scale_max = max(scale_max, float(scales.max().item()))
            log_scale_sum += float(torch.log(scales).sum(dtype=torch.float64).item())
    if len(totals) != 8 or any(count != 16 for count in counts.values()):
        raise ValueError("selection evaluation must contain 8 complete groups")
    group_means = {group: totals[group] / counts[group] for group in sorted(totals)}
    diagnostics = {
        "lower_cap_fraction": lower_count / scale_count,
        "upper_cap_fraction": upper_count / scale_count,
        "minimum_scale_m": scale_min,
        "maximum_scale_m": scale_max,
        "geometric_mean_scale_m": math.exp(log_scale_sum / scale_count),
    }
    return group_means, diagnostics


def _save_checkpoint(
    path: Path,
    *,
    model: HeterogeneousEnvelopePredictor,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    update_count: int,
    config_sha256: str,
    stage_p_checkpoint_sha256: str,
    feature_manifest_sha256: str,
    label_manifest_sha256: str,
    role_manifest_sha256: str,
    selection_key: tuple[float, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "scale_head_state_dict": model.scale_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_epoch": completed_epoch,
            "update_count": update_count,
            "config_sha256": config_sha256,
            "stage_p_checkpoint_sha256": stage_p_checkpoint_sha256,
            "feature_manifest_sha256": feature_manifest_sha256,
            "label_manifest_sha256": label_manifest_sha256,
            "role_manifest_sha256": role_manifest_sha256,
            "selection_key": selection_key,
            "potential_parameter_updates": 0,
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


def _control_replay(
    model: HeterogeneousEnvelopePredictor,
    record: JoinedAnchor,
    device: torch.device,
) -> dict[str, object]:
    features, mask, _ = _load_tensors(record, device)
    model.eval()
    with torch.no_grad():
        first = model(features, mask)
        second = model(features, mask)
        if not torch.equal(first.potential, second.potential):
            raise AssertionError("selected Stage-P potential did not replay exactly")
        if not torch.equal(first.edge_scales_m, second.edge_scales_m):
            raise AssertionError("selected scale head did not replay exactly")
        controls_first = {
            "learned": first.edge_scales_m,
            "pooled": pooled_scale_control(first.edge_scales_m),
            "uniform": uniform_scale_control(first.edge_scales_m),
            "evidence_only": evidence_only_scale_control(mask),
        }
        controls_second = {
            "learned": second.edge_scales_m,
            "pooled": pooled_scale_control(second.edge_scales_m),
            "uniform": uniform_scale_control(second.edge_scales_m),
            "evidence_only": evidence_only_scale_control(mask),
        }
        summary: dict[str, object] = {}
        for name, values in controls_first.items():
            if not torch.equal(values, controls_second[name]):
                raise AssertionError(f"{name} scale control did not replay exactly")
            if values.shape != (1, 8064) or not torch.isfinite(values).all():
                raise AssertionError(f"{name} scale control has an invalid tensor")
            if torch.any(values < 2.0**-10) or torch.any(values > 2.0**4):
                raise AssertionError(f"{name} scale control violated registered bounds")
            summary[name] = {
                "minimum_m": float(values.min().item()),
                "maximum_m": float(values.max().item()),
                "geometric_mean_m": float(
                    torch.exp(torch.log(values).mean(dtype=torch.float64)).item()
                ),
            }
    return {
        "recording_group": record.recording_group,
        "anchor_index": record.anchor_index,
        "controls": summary,
        "exact_replay": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--recording-group-manifest", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--stage-p-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/r004h2_scale_fit.yaml")
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    expected_hashes = {
        "recording_group_manifest_sha256": sha256_file(args.recording_group_manifest),
        "role_manifest_sha256": sha256_file(args.role_manifest),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "label_manifest_sha256": sha256_file(args.label_manifest),
    }
    for name, actual in expected_hashes.items():
        if actual != inputs[name]:
            raise ValueError(f"{name} disagrees with the frozen H2 config")
    stage_p_hash = sha256_file(args.stage_p_checkpoint)
    if stage_p_hash != config["predecessor"]["selected_checkpoint_sha256"]:
        raise ValueError("Stage-P checkpoint hash disagrees with the frozen H2 config")
    stage_p_payload = _checkpoint(args.stage_p_checkpoint)
    if stage_p_payload.get("config_sha256") != config["predecessor"]["h1_config_sha256"]:
        raise ValueError("Stage-P checkpoint config provenance disagrees with H1")

    roles = _read_roles(args.role_manifest)
    regenerated = deterministic_group_roles(
        _source_train_groups(args.recording_group_manifest),
        namespace="AORTA-R004EG-roles-v1",
        fit_count=42,
        selection_count=8,
        proxy_cal_count=16,
    )
    if roles["fit"] != regenerated.fit or roles["selection"] != regenerated.selection:
        raise ValueError("fit/selection roles are not a deterministic replay")
    if roles["proxy_cal"] != regenerated.proxy_cal:
        raise ValueError("proxy-cal roles are not a deterministic replay")

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

    optimization = config["optimization"]
    seed = int(optimization["seed"])
    maximum_epochs = int(optimization["maximum_epochs"])
    warmup_epochs = int(optimization["warmup_epochs"])
    total_updates = maximum_epochs * len(fit_records)
    warmup_updates = warmup_epochs * len(fit_records)
    base_learning_rate = float(optimization["base_learning_rate"])
    betas_raw = optimization["betas"]
    if not isinstance(betas_raw, list) or len(betas_raw) != 2:
        raise ValueError("H2 AdamW betas must contain exactly two values")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("R004H2 requires the registered CUDA device")
    torch.cuda.init()  # type: ignore[no-untyped-call]
    potential = QuotientFieldPredictor().to(device)
    potential.load_state_dict(stage_p_payload["model_state_dict"], strict=True)
    model = HeterogeneousEnvelopePredictor(potential).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != int(config["model"]["scale_head_trainable_parameters"]):
        raise ValueError("trainable parameter count disagrees with the frozen H2 config")
    if any(parameter.requires_grad for parameter in model.potential_predictor.parameters()):
        raise ValueError("Stage-P parameters were not frozen")
    optimizer = torch.optim.AdamW(
        model.scale_head.parameters(),
        lr=base_learning_rate,
        betas=(float(betas_raw[0]), float(betas_raw[1])),
        weight_decay=float(optimization["weight_decay"]),
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    result_path = args.output_directory / "result.json"
    latest_path = args.output_directory / "latest_scale.pt"
    best_path = args.output_directory / "best_scale.pt"
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    cap_tolerance = float(config["cap_collapse"]["cap_position_tolerance"])
    uniform_group_losses, uniform_diagnostics = _selection_metrics(
        model,
        selection_records,
        device,
        scale_variant="uniform",
        cap_position_tolerance=cap_tolerance,
    )
    uniform_primary = sum(uniform_group_losses.values()) / 8
    candidates: list[StageSCheckpointCandidate] = []
    epoch_ledger: list[dict[str, object]] = []
    update_count = 0
    best_key: tuple[float, int] | None = None

    for epoch_index in range(maximum_epochs):
        epoch_started = time.perf_counter()
        model.train()
        train_totals: list[float] = []
        train_scaled: list[float] = []
        train_logs: list[float] = []
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
            output = model(features, mask)
            loss = bounded_laplace_scale_loss(
                output.potential,
                labels,
                output.edge_scales_m,
            )
            torch.autograd.backward(loss.total)
            if any(
                parameter.grad is not None
                for parameter in model.potential_predictor.parameters()
            ):
                raise RuntimeError("a frozen Stage-P parameter received a gradient")
            gradients = [
                parameter.grad
                for parameter in model.scale_head.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if len(gradients) != 2 or not all(
                bool(torch.isfinite(gradient).all().item()) for gradient in gradients
            ):
                raise FloatingPointError("missing or nonfinite scale-head gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.scale_head.parameters(),
                float(optimization["gradient_clip_global_norm"]),
            )
            scalar_values = (
                float(loss.total.item()),
                float(loss.mean_scaled_absolute_residual.item()),
                float(loss.mean_log_scale.item()),
                float(gradient_norm.item()),
                learning_rate,
            )
            if any(not math.isfinite(value) for value in scalar_values):
                raise FloatingPointError("nonfinite H2 training scalar")
            optimizer.step()
            update_count += 1
            train_totals.append(scalar_values[0])
            train_scaled.append(scalar_values[1])
            train_logs.append(scalar_values[2])
            gradient_norm_maximum = max(gradient_norm_maximum, scalar_values[3])

        selection_losses, scale_diagnostics = _selection_metrics(
            model,
            selection_records,
            device,
            scale_variant="learned",
            cap_position_tolerance=cap_tolerance,
        )
        candidate = StageSCheckpointCandidate(
            epoch=epoch_index + 1,
            group_mean_proper_loss=tuple(selection_losses.values()),
        )
        candidates.append(candidate)
        selected_so_far = select_stage_s_checkpoint(candidates)
        current = select_stage_s_checkpoint((candidate,))
        current_key = (current.macro_group_mean_proper_loss, current.epoch)
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
            "train_mean_proper_loss": sum(train_totals) / len(train_totals),
            "train_mean_scaled_absolute_residual": sum(train_scaled) / len(train_scaled),
            "train_mean_log_scale": sum(train_logs) / len(train_logs),
            "preclip_gradient_norm_maximum": gradient_norm_maximum,
            "selection_group_mean_proper_loss": selection_losses,
            "selection_macro_group_mean_proper_loss": current.macro_group_mean_proper_loss,
            "selection_scale_diagnostics": scale_diagnostics,
            "selected_epoch_so_far": selected_so_far.epoch,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        epoch_ledger.append(epoch_record)
        checkpoint_arguments = {
            "model": model,
            "optimizer": optimizer,
            "completed_epoch": epoch_index + 1,
            "update_count": update_count,
            "config_sha256": sha256_file(args.config),
            "stage_p_checkpoint_sha256": stage_p_hash,
            "feature_manifest_sha256": expected_hashes["feature_manifest_sha256"],
            "label_manifest_sha256": expected_hashes["label_manifest_sha256"],
            "role_manifest_sha256": expected_hashes["role_manifest_sha256"],
            "selection_key": current_key,
        }
        _save_checkpoint(latest_path, **checkpoint_arguments)  # type: ignore[arg-type]
        if best_key is None or current_key < best_key:
            best_key = current_key
            _save_checkpoint(best_path, **checkpoint_arguments)  # type: ignore[arg-type]
        elapsed = time.perf_counter() - started
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        output_gib = _directory_bytes(args.output_directory) / 1024**3
        if peak_gib > float(config["resources"]["maximum_peak_gpu_gib"]):
            raise RuntimeError("H2 peak GPU allocation exceeded the registered cap")
        if elapsed > float(config["resources"]["maximum_wall_minutes"]) * 60.0:
            raise RuntimeError("H2 wall time exceeded the registered cap")
        if output_gib > float(config["resources"]["maximum_output_gib"]):
            raise RuntimeError("H2 output size exceeded the registered cap")
        progress: dict[str, object] = {
            "experiment_id": config["experiment_id"],
            "status": "IN_PROGRESS",
            "implementation_revision": args.implementation_revision,
            "config_sha256": sha256_file(args.config),
            "stage_p_checkpoint_sha256": stage_p_hash,
            **expected_hashes,
            "completed_epochs": epoch_index + 1,
            "maximum_epochs": maximum_epochs,
            "selected_epoch_so_far": selected_so_far.epoch,
            "uniform_selection_primary": uniform_primary,
            "uniform_scale_diagnostics": uniform_diagnostics,
            "epoch_ledger": epoch_ledger,
            "prohibited_payload_accessed": False,
            "potential_parameter_updates": 0,
        }
        _write_json(result_path, progress)
        print(
            json.dumps(
                {
                    "epoch": epoch_index + 1,
                    "train_mean_proper_loss": epoch_record["train_mean_proper_loss"],
                    "selection_primary": current.macro_group_mean_proper_loss,
                    "uniform_selection_primary": uniform_primary,
                    "lower_cap_fraction": scale_diagnostics["lower_cap_fraction"],
                    "upper_cap_fraction": scale_diagnostics["upper_cap_fraction"],
                    "selected_epoch_so_far": selected_so_far.epoch,
                    "epoch_seconds": epoch_record["epoch_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selected = select_stage_s_checkpoint(candidates)
    if best_key is None or selected.epoch != best_key[1]:
        raise AssertionError("persisted best scale checkpoint disagrees with selector")
    selected_checkpoint = _checkpoint(best_path)
    model.scale_head.load_state_dict(
        selected_checkpoint["scale_head_state_dict"],
        strict=True,
    )
    selected_losses, selected_diagnostics = _selection_metrics(
        model,
        selection_records,
        device,
        scale_variant="learned",
        cap_position_tolerance=cap_tolerance,
    )
    replayed_primary = sum(selected_losses.values()) / 8
    if replayed_primary != selected.macro_group_mean_proper_loss:
        raise AssertionError("selected scale checkpoint did not replay its primary metric")
    proper_gate = replayed_primary < uniform_primary
    cap_gate = qualifies_scale_caps(
        selected_diagnostics["lower_cap_fraction"],
        selected_diagnostics["upper_cap_fraction"],
        collapse_fraction=float(config["cap_collapse"]["collapse_fraction"]),
    )
    control_replay = _control_replay(model, selection_records[0], device)
    decision = "PASS" if proper_gate and cap_gate else "STOP"
    final_result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": decision,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "stage_p_checkpoint_sha256": stage_p_hash,
        **expected_hashes,
        "seed": seed,
        "stage_p_trainable_parameters": 0,
        "scale_head_trainable_parameters": trainable_parameters,
        "fit_group_count": 42,
        "selection_group_count": 8,
        "fit_anchor_count_per_epoch": len(fit_records),
        "selection_anchor_count_per_epoch": len(selection_records),
        "completed_epochs": maximum_epochs,
        "optimizer_updates": update_count,
        "potential_parameter_updates": 0,
        "selected_epoch": selected.epoch,
        "selected_primary": replayed_primary,
        "uniform_selection_primary": uniform_primary,
        "proper_loss_improvement_gate_passed": proper_gate,
        "selected_scale_diagnostics": selected_diagnostics,
        "scale_cap_gate_passed": cap_gate,
        "control_replay": control_replay,
        "epoch_ledger": epoch_ledger,
        "best_scale_checkpoint_sha256": sha256_file(best_path),
        "latest_scale_checkpoint_sha256": sha256_file(latest_path),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
        "output_bytes": _directory_bytes(args.output_directory),
        "prohibited_payload_accessed": False,
        "r004h3_registration_authorized": decision == "PASS",
        "claim_boundary": (
            "Selection-side proper scale-fit and deterministic-control qualification only. "
            "No proxy calibration, coverage, task utility, Gate B, OOD robustness, or "
            "novelty claim is established."
        ),
    }
    _write_json(result_path, final_result)
    print(json.dumps(final_result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
