"""Run frozen-checkpoint S2/S3 donor interventions for one R005E2 outer fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from aorta_drive.models.full_context_scorer import FullContextSetScorer
from aorta_drive.training.r005e2 import load_visual_tensor
from aorta_drive.training.r005e2_science import (
    ARM_MISMATCH,
    ScientificAnchor,
    ScientificDataset,
)
from aorta_drive.vision.feature_cache import sha256_file
from run_r005e2_fold_job import (
    _feature_paths_by_identity,
    _loader,
    _physical_gpu_identity,
    _predict,
    _tensor_state_sha256,
)


SCHEMES = ("S2_disjoint_geometry", "S3_hash_random")
SEEDS = (20261025, 20261026, 20261027)
PARENT_COMMIT = "0fdee4b464865d76f2efbc62955baa7e8f7ca8f5"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sensitivity_donor_map(
    manifest: dict[str, Any], *, outer_fold: int, scheme: str
) -> dict[str, str]:
    if scheme not in SCHEMES:
        raise ValueError("unknown donor sensitivity scheme")
    folds = [
        row
        for row in cast(list[dict[str, Any]], manifest.get("folds"))
        if int(row["outer_fold"]) == outer_fold
    ]
    if len(folds) != 1:
        raise ValueError("sensitivity manifest outer fold is not unique")
    schemes = cast(dict[str, Any], folds[0]["schemes"])
    stage = cast(dict[str, Any], schemes[scheme]["outer_test"])
    result: dict[str, str] = {}
    for row in cast(list[dict[str, Any]], stage["rows"]):
        recipient, donor = str(row["recipient"]), str(row["donor"])
        if recipient in result:
            raise ValueError("duplicate sensitivity recipient")
        result[recipient] = donor
    if len(result) != 112:
        raise ValueError("outer-test sensitivity map must contain 112 recipients")
    return result


def _dataset(
    *,
    groups: list[str],
    donor_map: dict[str, str],
    outer_fold: int,
    bundle_root: Path,
    bundle_manifest: dict[str, Any],
    visual_root: Path,
) -> ScientificDataset:
    allowed = set(groups)
    features = _feature_paths_by_identity(bundle_manifest, visual_root)
    anchors: list[ScientificAnchor] = []
    indices: list[int] = []
    for row in cast(list[dict[str, Any]], bundle_manifest["anchors"]):
        if str(row["recording_group"]) not in allowed:
            continue
        recipient = str(row["identity"])
        donor = donor_map.get(recipient)
        if donor is None or donor not in features:
            raise ValueError("sensitivity donor is absent from the frozen visual cache")
        anchors.append(
            ScientificAnchor(
                recording_group=str(row["recording_group"]),
                anchor_index=int(row["anchor_index"]),
                identity=recipient,
                correct_feature_paths=features[recipient],
                mismatch_feature_paths=features[donor],
            )
        )
        indices.append(int(row["bundle_index"]))
    if len(anchors) != 112:
        raise ValueError("outer-test sensitivity dataset must contain 112 anchors")
    return ScientificDataset(
        anchors=anchors,
        bundle_root=bundle_root,
        bundle_indices=indices,
        arm=ARM_MISMATCH,
        removed_slot_mean_path=bundle_root / f"removed_slot_mean_fold_{outer_fold}.npy",
    )


def _dino_rms_distance(
    *,
    groups: list[str],
    donor_map: dict[str, str],
    bundle_manifest: dict[str, Any],
    visual_root: Path,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    allowed = set(groups)
    features = _feature_paths_by_identity(bundle_manifest, visual_root)
    rows: list[tuple[int, float]] = []
    for row in cast(list[dict[str, Any]], bundle_manifest["anchors"]):
        if str(row["recording_group"]) not in allowed:
            continue
        recipient = str(row["identity"])
        donor = donor_map[recipient]
        correct = load_visual_tensor(features[recipient]).float()
        mismatched = load_visual_tensor(features[donor]).float()
        rms = torch.sqrt(torch.mean(torch.square(correct - mismatched)))
        if not bool(torch.isfinite(rms)):
            raise FloatingPointError("DINO donor distance is nonfinite")
        rows.append((int(row["bundle_index"]), float(rms)))
    rows.sort()
    return (
        np.asarray([row[0] for row in rows], dtype=np.int64),
        np.asarray([row[1] for row in rows], dtype=np.float32),
    )


def _checkpoint_state(checkpoint: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = cast(
        dict[str, Any], torch.load(checkpoint, map_location="cpu", weights_only=False)
    )
    state = cast(dict[str, torch.Tensor], payload["model"])
    return state, payload


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    if args.outer_fold not in range(6):
        raise ValueError("outer fold must be in [0,5]")
    if args.expected_sensitivity_sha256 != sha256_file(args.sensitivity_manifest):
        raise ValueError("sensitivity manifest differs from the frozen hash")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("R005E2S requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("R005E2S requires CUDA")
    parent_audit = cast(
        dict[str, Any],
        json.loads((args.parent_run_root / "audit.json").read_text("utf-8")),
    )
    if (
        parent_audit.get("status") != "PASS"
        or parent_audit.get("support") != "UNSUPPORTED"
        or parent_audit.get("decision") != "STOP_CURRENT_VISUAL_MAIN_CLAIM"
    ):
        raise ValueError("parent STOP decision is absent or differs from the frozen contract")
    args.output_directory.mkdir(parents=True)
    started = time.perf_counter()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    bundle = cast(
        dict[str, Any],
        json.loads((args.bundle_root / "manifest.json").read_text("utf-8")),
    )
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text("utf-8"))
    )
    sensitivity = cast(
        dict[str, Any], json.loads(args.sensitivity_manifest.read_text("utf-8"))
    )
    fold = next(
        row
        for row in cast(list[dict[str, Any]], crossfit["outer_folds"])
        if int(row["outer_fold"]) == args.outer_fold
    )
    groups = cast(list[str], fold["outer_test_groups"])
    datasets: dict[str, ScientificDataset] = {}
    distances: dict[str, npt.NDArray[np.float32]] = {}
    distance_indices: npt.NDArray[np.int64] | None = None
    for scheme in SCHEMES:
        donors = _sensitivity_donor_map(
            sensitivity, outer_fold=args.outer_fold, scheme=scheme
        )
        datasets[scheme] = _dataset(
            groups=groups,
            donor_map=donors,
            outer_fold=args.outer_fold,
            bundle_root=args.bundle_root,
            bundle_manifest=bundle,
            visual_root=args.visual_root,
        )
        current_indices, current_distances = _dino_rms_distance(
            groups=groups,
            donor_map=donors,
            bundle_manifest=bundle,
            visual_root=args.visual_root,
        )
        if distance_indices is not None and not np.array_equal(
            distance_indices, current_indices
        ):
            raise ValueError("sensitivity schemes have different recipient rosters")
        distance_indices = current_indices
        distances[scheme] = current_distances
    if distance_indices is None:
        raise RuntimeError("no sensitivity distances were created")

    device = torch.device("cuda")
    predictions: dict[str, list[npt.NDArray[np.float32]]] = {
        scheme: [] for scheme in SCHEMES
    }
    expected_indices: npt.NDArray[np.int64] | None = None
    expected_target: npt.NDArray[np.float32] | None = None
    expected_mask: npt.NDArray[np.bool_] | None = None
    checkpoint_ledger: list[dict[str, Any]] = []
    for seed in SEEDS:
        parent_root = (
            args.parent_run_root
            / "primary"
            / f"fold{args.outer_fold}_V2_correct_seed{seed}"
        )
        parent_result = cast(
            dict[str, Any], json.loads((parent_root / "result.json").read_text("utf-8"))
        )
        if (
            parent_result.get("status") != "COMPLETED"
            or parent_result.get("code_commit") != PARENT_COMMIT
            or int(parent_result.get("outer_fold", -1)) != args.outer_fold
            or int(parent_result.get("seed", -1)) != seed
            or parent_result.get("arm") != "V2_correct"
        ):
            raise ValueError("parent checkpoint result contract failed")
        checkpoint = parent_root / "checkpoint.pt"
        if sha256_file(checkpoint) != parent_result["checkpoint_sha256"]:
            raise ValueError("parent checkpoint file hash mismatch")
        state, checkpoint_payload = _checkpoint_state(checkpoint)
        if (
            int(checkpoint_payload["outer_fold"]) != args.outer_fold
            or int(checkpoint_payload["seed"]) != seed
            or checkpoint_payload["arm"] != "V2_correct"
            or _state_sha256(state) != parent_result["final_model_state_sha256"]
        ):
            raise ValueError("parent checkpoint state contract failed")
        model = FullContextSetScorer().to(device)
        model.load_state_dict(state, strict=True)
        if _tensor_state_sha256(model) != parent_result["final_model_state_sha256"]:
            raise ValueError("loaded model differs from the parent state")
        for scheme in SCHEMES:
            loader = _loader(
                datasets[scheme], shuffle=False, seed=seed, workers=args.workers
            )
            indices, values, target, mask = _predict(model, loader, device)
            if expected_indices is None:
                expected_indices, expected_target, expected_mask = indices, target, mask
            elif not (
                np.array_equal(expected_indices, indices)
                and np.array_equal(expected_target, target)
                and np.array_equal(expected_mask, mask)
            ):
                raise ValueError("scheme/seed target or membership mismatch")
            predictions[scheme].append(values)
        checkpoint_ledger.append(
            {
                "seed": seed,
                "checkpoint_sha256": parent_result["checkpoint_sha256"],
                "model_state_sha256": parent_result["final_model_state_sha256"],
                "selected_epoch": parent_result["selected_epoch"],
            }
        )
        del model
        torch.cuda.empty_cache()
    if expected_indices is None or expected_target is None or expected_mask is None:
        raise RuntimeError("no predictions were produced")
    if not np.array_equal(expected_indices, distance_indices):
        raise ValueError("DINO distance roster differs from prediction roster")

    prediction_path = args.output_directory / "predictions.npz"
    np.savez(
        prediction_path,
        seeds=np.asarray(SEEDS, dtype=np.int64),
        bundle_index=expected_indices,
        target=expected_target,
        candidate_mask=expected_mask,
        S2_disjoint_geometry_prediction=np.stack(predictions["S2_disjoint_geometry"]),
        S3_hash_random_prediction=np.stack(predictions["S3_hash_random"]),
        S2_disjoint_geometry_dino_rms=distances["S2_disjoint_geometry"],
        S3_hash_random_dino_rms=distances["S3_hash_random"],
    )
    result = {
        "schema": "aorta.r005e2s.fold_job.v1",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold": args.outer_fold,
        "test_anchor_count": int(expected_indices.size),
        "schemes": list(SCHEMES),
        "seeds": list(SEEDS),
        "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_ledger": checkpoint_ledger,
        "code_commit": args.code_commit,
        "parent_decision": "STOP_CURRENT_VISUAL_MAIN_CLAIM",
        "parent_decision_is_immutable": True,
        "runner_sha256": sha256_file(Path(__file__)),
        "physical_gpu": _physical_gpu_identity(args.physical_gpu),
        "input_sha256": {
            "bundle_manifest": sha256_file(args.bundle_root / "manifest.json"),
            "crossfit_manifest": sha256_file(args.crossfit_manifest),
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
    parser.add_argument("--sensitivity-manifest", type=Path, required=True)
    parser.add_argument("--expected-sensitivity-sha256", required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "checkpoint_ledger"},
            sort_keys=True,
        )
    )
