"""One-fold, outcome-free R005E2 data-loader/backward/checkpoint rehearsal."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from aorta_drive.models.full_context_scorer import (
    FullContextSetScorer,
    equal_anchor_challenger_margin_mse,
)
from aorta_drive.training.r005e2 import (
    ResourceAnchor,
    ResourceRehearsalDataset,
    visual_records_by_identity,
)
from aorta_drive.vision.feature_cache import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _resolve_panel(roots: list[Path], scene_token: str, expected_hash: str) -> Path:
    matches = [
        path
        for root in roots
        if (path := root / f"{scene_token}.npz").is_file()
        and sha256_file(path) == expected_hash
    ]
    if len(matches) != 1:
        raise ValueError("panel must resolve by exact hash in exactly one frozen root")
    return matches[0]


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in batch.items()
    }


def _forward(model: FullContextSetScorer, batch: dict[str, Tensor]) -> Tensor:
    return cast(
        Tensor,
        model(
            batch["candidate_poses"],
            batch["candidate_mask"],
            batch["previous_poses"],
            batch["visual_features"],
            batch["visual_slot_mask"],
        ),
    )


def _state_exact(left: FullContextSetScorer, right: FullContextSetScorer) -> bool:
    return all(
        torch.equal(left_value.detach().cpu(), right.state_dict()[name].detach().cpu())
        for name, left_value in left.state_dict().items()
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("resource rehearsal requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    args.output_directory.mkdir(parents=True)
    started = time.perf_counter()

    roster = cast(dict[str, Any], json.loads(args.panel_roster.read_text(encoding="utf-8")))
    crossfit = cast(dict[str, Any], json.loads(args.crossfit_manifest.read_text(encoding="utf-8")))
    visual = cast(dict[str, Any], json.loads(args.visual_manifest.read_text(encoding="utf-8")))
    fold = next(
        row
        for row in cast(list[dict[str, Any]], crossfit["outer_folds"])
        if int(row["outer_fold"]) == args.outer_fold
    )
    rehearsal_groups = set(cast(list[str], fold["inner_fit_groups"]))
    if len(rehearsal_groups) != 28:
        raise ValueError("frozen one-fold rehearsal must use 28 inner-fit groups")
    visual_index = visual_records_by_identity(visual, args.visual_root)
    anchors: list[ResourceAnchor] = []
    for row in cast(list[dict[str, Any]], roster["anchors"]):
        if str(row["recording_group"]) not in rehearsal_groups:
            continue
        identity = (str(row["log_name"]), str(row["scene_token"]))
        if identity not in visual_index:
            raise ValueError("panel identity is missing from the frozen visual cache")
        anchors.append(
            ResourceAnchor(
                recording_group=str(row["recording_group"]),
                anchor_index=int(row["anchor_index"]),
                scene_token=str(row["scene_token"]),
                panel_path=_resolve_panel(
                    args.panel_root,
                    str(row["scene_token"]),
                    str(row["panel_artifact_sha256"]),
                ),
                feature_paths=visual_index[identity],
            )
        )
    anchors.sort(key=lambda row: (row.recording_group, row.anchor_index))
    if len(anchors) != 448:
        raise ValueError("one-fold rehearsal must contain exactly 448 anchors")

    generator = torch.Generator().manual_seed(args.seed)
    dataset = ResourceRehearsalDataset(anchors)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model = FullContextSetScorer().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    torch.cuda.reset_peak_memory_stats(device)

    data_wait_ms: list[float] = []
    train_step_ms: list[float] = []
    first_cpu_batch: dict[str, Tensor] | None = None
    zero_challenger_count = 0
    completed_steps = 0
    iterator = iter(loader)
    for _ in range(len(loader)):
        waiting = time.perf_counter()
        cpu_batch = next(iterator)
        data_wait_ms.append((time.perf_counter() - waiting) * 1000.0)
        if first_cpu_batch is None:
            first_cpu_batch = {key: value.clone() for key, value in cpu_batch.items()}
        batch = _to_device(cpu_batch, device)
        zero_challenger_count += int((batch["candidate_mask"].sum(dim=1) == 1).sum().item())
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = _forward(model, batch)
            loss = equal_anchor_challenger_margin_mse(
                prediction,
                batch["target_margin"],
                batch["candidate_mask"],
            ).total
        if not torch.isfinite(loss):
            raise FloatingPointError("resource-rehearsal loss is nonfinite")
        loss.backward()  # type: ignore[no-untyped-call]
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise FloatingPointError("resource-rehearsal gradient is nonfinite")
        optimizer.step()
        torch.cuda.synchronize(device)
        train_step_ms.append((time.perf_counter() - step_started) * 1000.0)
        completed_steps += 1

    if first_cpu_batch is None or completed_steps != 448:
        raise RuntimeError("one-fold rehearsal did not consume all 448 anchors")
    checkpoint_path = args.output_directory / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_steps": completed_steps,
            "seed": args.seed,
        },
        checkpoint_path,
    )
    restored = FullContextSetScorer().to(device)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=3.0e-4, weight_decay=1.0e-4
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored.load_state_dict(state["model"])
    restored_optimizer.load_state_dict(state["optimizer"])
    if int(state["completed_steps"]) != completed_steps or int(state["seed"]) != args.seed:
        raise ValueError("checkpoint scalar state did not recover exactly")
    if not _state_exact(model, restored):
        raise ValueError("checkpoint model tensors did not recover exactly")
    restore_batch = _to_device(first_cpu_batch, device)
    model.eval()
    restored.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference = _forward(model, restore_batch)
        recovered = _forward(restored, restore_batch)
    if not torch.equal(reference, recovered):
        raise ValueError("checkpoint forward recovery is not bit exact")

    total_runtime = time.perf_counter() - started
    result: dict[str, Any] = {
        "schema": "aorta.r005e2.training_resource_rehearsal.v1",
        "status": "PASS",
        "scientific_metrics_emitted": False,
        "scientific_labels_accessed": False,
        "development_calibration_test_navdream_target_accessed": False,
        "previous_trajectory_source": "current_native_shape_surrogate_not_scientific_history",
        "target_source": "exact_zero_resource_surrogate",
        "outer_fold": args.outer_fold,
        "inner_fit_group_count": len(rehearsal_groups),
        "anchor_count": len(anchors),
        "completed_training_steps": completed_steps,
        "zero_challenger_anchor_count": zero_challenger_count,
        "batch_size": 1,
        "workers": args.workers,
        "seed": args.seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "data_wait_ms": {
            "mean": float(np.mean(data_wait_ms)),
            "p95": float(np.quantile(data_wait_ms, 0.95)),
            "maximum": float(np.max(data_wait_ms)),
        },
        "train_step_ms": {
            "mean": float(np.mean(train_step_ms)),
            "p95": float(np.quantile(train_step_ms, 0.95)),
            "maximum": float(np.max(train_step_ms)),
        },
        "total_runtime_seconds": total_runtime,
        "projected_three_arm_three_seed_six_fold_gpu_hours": (
            total_runtime * 3 * 3 * 6 / 3600.0
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_tensor_state_exact": True,
        "checkpoint_forward_recovery_bit_exact": True,
        "finite_loss_and_gradients": True,
        "input_hashes": {
            "panel_roster_sha256": sha256_file(args.panel_roster),
            "crossfit_manifest_sha256": sha256_file(args.crossfit_manifest),
            "visual_manifest_sha256": sha256_file(args.visual_manifest),
            "scorer_source_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/aorta_drive/models/full_context_scorer.py"
            ),
            "runner_sha256": sha256_file(Path(__file__)),
        },
    }
    if not math.isfinite(result["projected_three_arm_three_seed_six_fold_gpu_hours"]):
        raise FloatingPointError("resource projection is nonfinite")
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-roster", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(json.dumps(output, indent=2))
