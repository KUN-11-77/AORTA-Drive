"""Run one frozen R005E3A loss-only objective qualification job."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from aorta_drive.models.full_context_scorer import (
    FullContextSetScorer,
    soft_selection_regret_loss,
)
from aorta_drive.training.r005e2_science import (
    ARM_CORRECT,
    ARM_MISMATCH,
    ARM_REMOVED,
    ScientificDataset,
)
from aorta_drive.vision.feature_cache import sha256_file
from run_r005e2_fold_job import (
    _dataset,
    _forward,
    _loader,
    _new_model,
    _physical_gpu_identity,
    _predict,
    _schedule,
    _tensor_state_sha256,
    _to_device,
    _write_json,
)
from run_r005e2s_fold_job import (
    SCHEMES,
    _dataset as _sensitivity_dataset,
    _sensitivity_donor_map,
)


SEEDS = (20261025, 20261026, 20261027)
PARENT_COMMIT = "0fdee4b464865d76f2efbc62955baa7e8f7ca8f5"
PARENT_DECISION = "STOP_CURRENT_VISUAL_MAIN_CLAIM"
TIE_TOLERANCE = 1.0e-12


def _temperature_from_outer_train(
    *,
    bundle_root: Path,
    bundle_manifest: dict[str, Any],
    groups: Iterable[str],
) -> tuple[float, dict[str, Any]]:
    """Derive the sole registered temperature from outer-train utility gaps."""

    allowed = set(groups)
    indices = np.asarray(
        [
            int(row["bundle_index"])
            for row in cast(list[dict[str, Any]], bundle_manifest["anchors"])
            if str(row["recording_group"]) in allowed
        ],
        dtype=np.int64,
    )
    if indices.shape != (16 * len(allowed),):
        raise ValueError("outer-train temperature roster has invalid cardinality")
    target = np.load(bundle_root / "target_margin.npy", mmap_mode="r")
    mask = np.load(bundle_root / "candidate_mask.npy", mmap_mode="r")
    gaps: list[float] = []
    for index in indices.tolist():
        valid = np.asarray(target[index][mask[index]], dtype=np.float64)
        if valid.size == 0 or not np.isfinite(valid).all():
            raise ValueError("temperature source contains no valid finite candidate")
        current = float(np.max(valid)) - valid
        gaps.extend(float(value) for value in current[current > TIE_TOLERANCE])
    if not gaps:
        raise ValueError("outer-train contains no strictly positive utility gap")
    ordered = np.sort(np.asarray(gaps, dtype=np.float64))
    temperature = float(np.median(ordered))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("derived temperature is not finite and strictly positive")
    digest = hashlib.sha256(ordered.tobytes()).hexdigest()
    return temperature, {
        "rule": "outer_train_strictly_positive_best_minus_candidate_gap_median",
        "even_count_median": "arithmetic_mean_of_two_central_order_statistics",
        "tie_tolerance": TIE_TOLERANCE,
        "gap_count": int(ordered.size),
        "minimum_gap": float(ordered[0]),
        "maximum_gap": float(ordered[-1]),
        "ordered_gap_float64_sha256": digest,
        "source_bundle_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }


def _train_soft_epoch(
    model: FullContextSetScorer,
    loader: torch.utils.data.DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    *,
    temperature: float,
) -> list[int]:
    model.train()
    order: list[int] = []
    for cpu_batch in loader:
        order.append(int(cpu_batch["bundle_index"].item()))
        batch = _to_device(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = _forward(model, batch)
            loss = soft_selection_regret_loss(
                prediction,
                batch["target_margin"],
                batch["candidate_mask"],
                temperature=temperature,
            ).total
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("training loss is nonfinite")
        loss.backward()  # type: ignore[no-untyped-call]
        if any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise FloatingPointError("training gradient is nonfinite")
        optimizer.step()
        scheduler.step()
    return order


def _parent_result(
    parent_run_root: Path, *, outer_fold: int, seed: int
) -> tuple[Path, dict[str, Any]]:
    root = parent_run_root / "primary" / f"fold{outer_fold}_V2_correct_seed{seed}"
    result = cast(dict[str, Any], json.loads((root / "result.json").read_text("utf-8")))
    if (
        result.get("status") != "COMPLETED"
        or result.get("code_commit") != PARENT_COMMIT
        or result.get("arm") != "V2_correct"
        or int(result.get("outer_fold", -1)) != outer_fold
        or int(result.get("seed", -1)) != seed
    ):
        raise ValueError("parent V2 job contract failed")
    return root, result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    if args.outer_fold not in range(6) or args.seed not in SEEDS:
        raise ValueError("outer fold or seed is outside the frozen roster")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("R005E3A requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("R005E3A jobs require CUDA")
    if sha256_file(args.sensitivity_manifest) != args.sensitivity_manifest_sha256:
        raise ValueError("sensitivity manifest differs from its frozen hash")
    parent_audit = cast(
        dict[str, Any],
        json.loads((args.parent_run_root / "audit.json").read_text("utf-8")),
    )
    if (
        parent_audit.get("status") != "PASS"
        or parent_audit.get("support") != "UNSUPPORTED"
        or parent_audit.get("decision") != PARENT_DECISION
    ):
        raise ValueError("parent audit is not the frozen R005E2B STOP")

    args.output_directory.mkdir(parents=True)
    started = time.perf_counter()
    bundle = cast(
        dict[str, Any],
        json.loads((args.bundle_root / "manifest.json").read_text("utf-8")),
    )
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text("utf-8"))
    )
    mismatch = cast(
        dict[str, Any], json.loads(args.mismatch_manifest.read_text("utf-8"))
    )
    sensitivity = cast(
        dict[str, Any], json.loads(args.sensitivity_manifest.read_text("utf-8"))
    )
    fold = next(
        row
        for row in cast(list[dict[str, Any]], crossfit["outer_folds"])
        if int(row["outer_fold"]) == args.outer_fold
    )
    parent_root, parent = _parent_result(
        args.parent_run_root, outer_fold=args.outer_fold, seed=args.seed
    )
    selected_epoch = int(parent["selected_epoch"])
    if selected_epoch <= 0:
        raise ValueError("parent selected epoch is invalid")
    temperature, temperature_ledger = _temperature_from_outer_train(
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        groups=cast(list[str], fold["outer_train_groups"]),
    )

    train = _dataset(
        groups=cast(list[str], fold["outer_train_groups"]),
        stage="refit_train",
        arm=ARM_CORRECT,
        outer_fold=args.outer_fold,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    correct = _dataset(
        groups=cast(list[str], fold["outer_test_groups"]),
        stage="outer_test",
        arm=ARM_CORRECT,
        outer_fold=args.outer_fold,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    primary_mismatch = _dataset(
        groups=cast(list[str], fold["outer_test_groups"]),
        stage="outer_test",
        arm=ARM_MISMATCH,
        outer_fold=args.outer_fold,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    removed = _dataset(
        groups=cast(list[str], fold["outer_test_groups"]),
        stage="outer_test",
        arm=ARM_REMOVED,
        outer_fold=args.outer_fold,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    sensitivity_sets: dict[str, ScientificDataset] = {}
    for scheme in SCHEMES:
        donor_map = _sensitivity_donor_map(
            sensitivity, outer_fold=args.outer_fold, scheme=scheme
        )
        sensitivity_sets[scheme] = _sensitivity_dataset(
            groups=cast(list[str], fold["outer_test_groups"]),
            donor_map=donor_map,
            outer_fold=args.outer_fold,
            bundle_root=args.bundle_root,
            bundle_manifest=bundle,
            visual_root=args.visual_root,
        )

    device = torch.device("cuda")
    model, initial_hash = _new_model(args.seed, device)
    if initial_hash != parent["refit_initial_state_sha256"]:
        raise ValueError("treatment initialization differs from parent V2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    train_loader = _loader(train, shuffle=True, seed=args.seed, workers=args.workers)
    expected_steps = selected_epoch * len(train_loader)
    scheduler = _schedule(optimizer, expected_steps)
    training_order: list[int] = []
    for _ in range(selected_epoch):
        training_order.extend(
            _train_soft_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                temperature=temperature,
            )
        )
    if len(training_order) != expected_steps:
        raise RuntimeError("optimizer step budget differs from the frozen contract")

    condition_sets = {
        "correct": correct,
        "S1_primary_geometry": primary_mismatch,
        "S2_disjoint_geometry": sensitivity_sets["S2_disjoint_geometry"],
        "S3_hash_random": sensitivity_sets["S3_hash_random"],
        "removed": removed,
    }
    payload: dict[str, npt.NDArray[Any]] = {}
    expected_indices: npt.NDArray[np.int64] | None = None
    expected_target: npt.NDArray[np.float32] | None = None
    expected_mask: npt.NDArray[np.bool_] | None = None
    for condition, dataset in condition_sets.items():
        loader = _loader(dataset, shuffle=False, seed=args.seed, workers=args.workers)
        indices, prediction, target, mask = _predict(model, loader, device)
        if expected_indices is None:
            expected_indices, expected_target, expected_mask = indices, target, mask
        else:
            if expected_target is None or expected_mask is None:
                raise RuntimeError("partial expected test payload")
            if not (
                np.array_equal(expected_indices, indices)
                and np.array_equal(expected_target, target)
                and np.array_equal(expected_mask, mask)
            ):
                raise ValueError("test intervention roster/target/mask differs")
        payload[f"{condition}_prediction"] = prediction
    if expected_indices is None or expected_target is None or expected_mask is None:
        raise RuntimeError("no test predictions were produced")
    payload.update(
        {
            "bundle_index": expected_indices,
            "target": expected_target,
            "candidate_mask": expected_mask,
        }
    )
    prediction_path = args.output_directory / "predictions.npz"
    np.savez(prediction_path, **payload)
    checkpoint_path = args.output_directory / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "outer_fold": args.outer_fold,
            "seed": args.seed,
            "selected_epoch": selected_epoch,
            "objective": "soft_selection_regret",
            "temperature": temperature,
        },
        checkpoint_path,
    )
    result = {
        "schema": "aorta.r005e3a.fold_job.v1",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "objective": "soft_selection_regret",
        "temperature": temperature,
        "temperature_ledger": temperature_ledger,
        "selected_epoch": selected_epoch,
        "optimizer_step_count": expected_steps,
        "training_bundle_index_order_sha256": hashlib.sha256(
            np.asarray(training_order, dtype=np.int64).tobytes()
        ).hexdigest(),
        "initial_model_state_sha256": initial_hash,
        "parent_refit_initial_state_sha256": parent["refit_initial_state_sha256"],
        "parent_selected_epoch": parent["selected_epoch"],
        "parent_job_result_sha256": sha256_file(parent_root / "result.json"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "test_anchor_count": int(expected_indices.size),
        "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "final_model_state_sha256": _tensor_state_sha256(model),
        "code_commit": args.code_commit,
        "runner_sha256": sha256_file(Path(__file__)),
        "physical_gpu": _physical_gpu_identity(args.physical_gpu),
        "input_sha256": {
            "bundle_manifest": sha256_file(args.bundle_root / "manifest.json"),
            "crossfit_manifest": sha256_file(args.crossfit_manifest),
            "mismatch_manifest": sha256_file(args.mismatch_manifest),
            "sensitivity_manifest": sha256_file(args.sensitivity_manifest),
            "parent_audit": sha256_file(args.parent_run_root / "audit.json"),
        },
        "runtime_seconds": time.perf_counter() - started,
        "development_calibration_test_navdream_target_accessed": False,
        "scientific_metrics_emitted": False,
    }
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--mismatch-manifest", type=Path, required=True)
    parser.add_argument("--sensitivity-manifest", type=Path, required=True)
    parser.add_argument("--sensitivity-manifest-sha256", required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(json.dumps(output, sort_keys=True))
