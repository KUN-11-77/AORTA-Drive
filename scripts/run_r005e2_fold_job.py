"""Run one frozen outer-fold/arm/seed R005E2 scientific job."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from aorta_drive.models.full_context_scorer import (  # type: ignore[import-untyped]
    FullContextSetScorer,
    equal_anchor_challenger_margin_mse,
)
from aorta_drive.training.r005e2_science import (  # type: ignore[import-untyped]
    ARMS,
    ARM_CORRECT,
    ARM_MISMATCH,
    ARM_REMOVED,
    ScientificAnchor,
    ScientificDataset,
    donor_map_for_stage,
    identity,
)
from aorta_drive.vision.feature_cache import sha256_file  # type: ignore[import-untyped]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _tensor_state_sha256(model: FullContextSetScorer) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _physical_gpu_identity(index: int) -> dict[str, str | int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(fields) != 3 or int(fields[0]) != index:
        raise ValueError("physical GPU identity query failed")
    return {"physical_index": index, "uuid": fields[1], "name": fields[2]}


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in batch.items()
        if key != "bundle_index"
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


def _loader(
    dataset: ScientificDataset,
    *,
    shuffle: bool,
    seed: int,
    workers: int,
) -> torch.utils.data.DataLoader[dict[str, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )


def _schedule(
    optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total optimizer steps must be positive")
    warmup = max(1, int(math.ceil(0.05 * total_steps)))

    def factor(step: int) -> float:
        completed = step + 1
        if completed <= warmup:
            return completed / warmup
        progress = (completed - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _train_epoch(
    model: FullContextSetScorer,
    loader: torch.utils.data.DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> None:
    model.train()
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = _forward(model, batch)
            loss = equal_anchor_challenger_margin_mse(
                prediction, batch["target_margin"], batch["candidate_mask"]
            ).total
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("training loss is nonfinite")
        loss.backward()
        if any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise FloatingPointError("training gradient is nonfinite")
        optimizer.step()
        scheduler.step()


def _predict(
    model: FullContextSetScorer,
    loader: torch.utils.data.DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
]:
    model.eval()
    indices: list[int] = []
    predictions: list[npt.NDArray[np.float32]] = []
    targets: list[npt.NDArray[np.float32]] = []
    masks: list[npt.NDArray[np.bool_]] = []
    with torch.inference_mode():
        for cpu_batch in loader:
            indices.append(int(cpu_batch["bundle_index"].item()))
            batch = _to_device(cpu_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = _forward(model, batch)
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("prediction is nonfinite")
            predictions.append(prediction.float().cpu().numpy()[0])
            targets.append(cpu_batch["target_margin"].numpy()[0])
            masks.append(cpu_batch["candidate_mask"].numpy()[0])
    order = np.argsort(np.asarray(indices, dtype=np.int64))
    return (
        np.asarray(indices, dtype=np.int64)[order],
        np.stack(predictions)[order].astype(np.float32),
        np.stack(targets)[order].astype(np.float32),
        np.stack(masks)[order].astype(np.bool_),
    )


def _group_equal_mse(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    indices: npt.NDArray[np.int64],
    index_to_group: dict[int, str],
) -> float:
    grouped: dict[str, list[float]] = {}
    for row_index, bundle_index in enumerate(indices.tolist()):
        challenger = mask[row_index].copy()
        challenger[0] = False
        count = int(challenger.sum())
        value = (
            float(
                np.square(prediction[row_index] - target[row_index])[challenger].mean()
            )
            if count
            else 0.0
        )
        grouped.setdefault(index_to_group[bundle_index], []).append(value)
    if not grouped or any(len(values) != 16 for values in grouped.values()):
        raise ValueError("validation group cardinality must be exactly sixteen")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _feature_paths_by_identity(
    manifest: dict[str, Any], root: Path
) -> dict[str, tuple[Path, ...]]:
    result: dict[str, tuple[Path, ...]] = {}
    for row in cast(list[dict[str, Any]], manifest["anchors"]):
        result[str(row["identity"])] = tuple(
            root / str(relative) for relative in row["correct_feature_relative_paths"]
        )
    if len(result) != 672:
        raise ValueError("scientific bundle visual index must contain 672 anchors")
    return result


def _dataset(
    *,
    groups: Iterable[str],
    stage: str,
    arm: str,
    outer_fold: int,
    bundle_root: Path,
    bundle_manifest: dict[str, Any],
    visual_root: Path,
    mismatch_manifest: dict[str, Any],
) -> ScientificDataset:
    allowed = set(groups)
    features = _feature_paths_by_identity(bundle_manifest, visual_root)
    donors = donor_map_for_stage(mismatch_manifest, outer_fold=outer_fold, stage=stage)
    anchors: list[ScientificAnchor] = []
    indices: list[int] = []
    for row in cast(list[dict[str, Any]], bundle_manifest["anchors"]):
        if str(row["recording_group"]) not in allowed:
            continue
        key = str(row["identity"])
        donor = donors.get(key, key)
        if arm == ARM_MISMATCH and key not in donors:
            raise ValueError("mismatch recipient is absent from the frozen stage map")
        if donor not in features:
            raise ValueError("donor visual identity is absent from the cache")
        anchors.append(
            ScientificAnchor(
                recording_group=str(row["recording_group"]),
                anchor_index=int(row["anchor_index"]),
                identity=key,
                correct_feature_paths=features[key],
                mismatch_feature_paths=features[donor],
            )
        )
        indices.append(int(row["bundle_index"]))
    expected = 16 * len(allowed)
    if len(anchors) != expected:
        raise ValueError("scientific dataset group cardinality mismatch")
    return ScientificDataset(
        anchors=anchors,
        bundle_root=bundle_root,
        bundle_indices=indices,
        arm=arm,
        removed_slot_mean_path=bundle_root / f"removed_slot_mean_fold_{outer_fold}.npy",
    )


def _new_model(seed: int, device: torch.device) -> tuple[FullContextSetScorer, str]:
    _seed_everything(seed)
    model = FullContextSetScorer().to(device)
    return model, _tensor_state_sha256(model)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    if args.arm not in ARMS:
        raise ValueError("arm is not part of the frozen R005E2 roster")
    if args.seed not in (20261025, 20261026, 20261027):
        raise ValueError("seed is not part of the frozen R005E2 roster")
    if args.outer_fold not in range(6):
        raise ValueError("outer fold must be in [0,5]")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("R005E2 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("R005E2 scientific jobs require CUDA")
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
    fold = next(
        row
        for row in cast(list[dict[str, Any]], crossfit["outer_folds"])
        if int(row["outer_fold"]) == args.outer_fold
    )
    index_to_group = {
        int(row["bundle_index"]): str(row["recording_group"])
        for row in cast(list[dict[str, Any]], bundle["anchors"])
    }
    selection_fit = _dataset(
        groups=fold["inner_fit_groups"],
        stage="selection_fit",
        arm=args.arm,
        outer_fold=args.outer_fold,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    selection_validation = _dataset(
        groups=fold["inner_validation_groups"],
        stage="selection_validation",
        arm=args.arm,
        outer_fold=args.outer_fold,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    device = torch.device("cuda")
    model, selection_initial_hash = _new_model(args.seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    selection_loader = _loader(
        selection_fit, shuffle=True, seed=args.seed, workers=args.workers
    )
    validation_loader = _loader(
        selection_validation, shuffle=False, seed=args.seed, workers=args.workers
    )
    scheduler = _schedule(optimizer, 40 * len(selection_loader))
    best_value = math.inf
    best_epoch = 0
    stale = 0
    epoch_ledger: list[dict[str, float | int]] = []
    for epoch in range(1, 41):
        _train_epoch(model, selection_loader, optimizer, scheduler, device)
        indices, prediction, target, mask = _predict(model, validation_loader, device)
        validation_mse = _group_equal_mse(
            prediction, target, mask, indices, index_to_group
        )
        epoch_ledger.append(
            {"epoch": epoch, "validation_group_equal_mse": validation_mse}
        )
        if validation_mse < best_value:
            best_value = validation_mse
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= 5:
            break
    if best_epoch <= 0:
        raise RuntimeError("early stopping failed to select an epoch")
    del model, optimizer, scheduler, selection_loader, validation_loader
    torch.cuda.empty_cache()

    refit = _dataset(
        groups=fold["outer_train_groups"],
        stage="refit_train",
        arm=args.arm,
        outer_fold=args.outer_fold,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    test = _dataset(
        groups=fold["outer_test_groups"],
        stage="outer_test",
        arm=args.arm,
        outer_fold=args.outer_fold,
        bundle_root=args.bundle_root,
        bundle_manifest=bundle,
        visual_root=args.visual_root,
        mismatch_manifest=mismatch,
    )
    model, refit_initial_hash = _new_model(args.seed, device)
    if refit_initial_hash != selection_initial_hash:
        raise RuntimeError("selection/refit model initialization differs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    refit_loader = _loader(refit, shuffle=True, seed=args.seed, workers=args.workers)
    scheduler = _schedule(optimizer, best_epoch * len(refit_loader))
    for _ in range(best_epoch):
        _train_epoch(model, refit_loader, optimizer, scheduler, device)
    test_loader = _loader(test, shuffle=False, seed=args.seed, workers=args.workers)
    indices, prediction, target, mask = _predict(model, test_loader, device)
    prediction_payload: dict[str, npt.NDArray[Any]] = {
        "bundle_index": indices,
        "prediction": prediction,
        "target": target,
        "candidate_mask": mask,
    }
    if args.arm == ARM_CORRECT:
        for intervention_arm, name in (
            (ARM_MISMATCH, "mismatch_intervention_prediction"),
            (ARM_REMOVED, "removed_intervention_prediction"),
        ):
            intervention = _dataset(
                groups=fold["outer_test_groups"],
                stage="outer_test",
                arm=intervention_arm,
                outer_fold=args.outer_fold,
                bundle_root=args.bundle_root,
                bundle_manifest=bundle,
                visual_root=args.visual_root,
                mismatch_manifest=mismatch,
            )
            loader = _loader(
                intervention, shuffle=False, seed=args.seed, workers=args.workers
            )
            intervention_indices, values, _, _ = _predict(model, loader, device)
            if not np.array_equal(intervention_indices, indices):
                raise ValueError("intervention test identities differ")
            prediction_payload[name] = values
    prediction_path = args.output_directory / "predictions.npz"
    np.savez(prediction_path, **prediction_payload)
    checkpoint_path = args.output_directory / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "outer_fold": args.outer_fold,
            "arm": args.arm,
            "seed": args.seed,
            "selected_epoch": best_epoch,
        },
        checkpoint_path,
    )
    final_state_hash = _tensor_state_sha256(model)
    result = {
        "schema": "aorta.r005e2.fold_job.v1",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold": args.outer_fold,
        "arm": args.arm,
        "seed": args.seed,
        "selected_epoch": best_epoch,
        "selection_epochs_executed": len(epoch_ledger),
        "selection_initial_state_sha256": selection_initial_hash,
        "refit_initial_state_sha256": refit_initial_hash,
        "test_anchor_count": int(indices.size),
        "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "final_model_state_sha256": final_state_hash,
        "code_commit": args.code_commit,
        "runner_sha256": sha256_file(Path(__file__)),
        "physical_gpu": _physical_gpu_identity(args.physical_gpu),
        "input_sha256": {
            "bundle_manifest": sha256_file(args.bundle_root / "manifest.json"),
            "crossfit_manifest": sha256_file(args.crossfit_manifest),
            "mismatch_manifest": sha256_file(args.mismatch_manifest),
        },
        "epoch_ledger": epoch_ledger,
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
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "epoch_ledger"},
            sort_keys=True,
        )
    )
