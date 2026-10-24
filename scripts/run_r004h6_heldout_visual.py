"""One-read held-out development visual generalization experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
import yaml  # type: ignore[import-untyped]
from PIL import Image

from aorta_drive.evaluation.appearance_mechanism import measure_field_mechanism, oriented_grid_edges
from aorta_drive.evaluation.heldout_visual import deterministic_group_derangement, summarize_heldout_ledgers
from aorta_drive.models.quotient_field import QuotientFieldPredictor
from aorta_drive.training.r004ec import JoinedAnchor, load_joined_anchors, load_teacher_arrays, sample_dense_features_torch
from aorta_drive.vision.appearance import H6_APPEARANCE_CONDITIONS, apply_h6_controlled_appearance
from aorta_drive.vision.feature_cache import (
    dino_python_tree_sha256,
    load_dense_bfloat16_bits,
    sha256_file,
    validate_projection_arrays,
)

RAW_SHAPE = (1080, 1920)
RESIZED_SHAPE = (288, 512)
EXPECTED_DINO_PARAMETERS = 840_633_600
EXPECTED_DINO_FILES = 151
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes its declared root")
    return path


def _groups(path: Path, *, column: str, value: str, expected: int) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = tuple(str(row["recording_group"]) for row in rows if row[column] == value)
    if len(groups) != expected or len(set(groups)) != expected:
        raise ValueError(f"expected {expected} unique {value} recording groups")
    return groups


def _feature_records(path: Path, groups: frozenset[str], expected: int) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("feature manifest lacks records")
    selected: dict[str, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, dict) or raw.get("recording_group") not in groups:
            continue
        item = cast(dict[str, Any], raw)
        token = str(item["scene_token"])
        slots = item.get("slots")
        if token in selected or not isinstance(slots, list) or len(slots) != 12:
            raise ValueError("invalid development feature record")
        item["slots"] = sorted(slots, key=lambda slot: int(slot["slot_index"]))
        if [int(slot["slot_index"]) for slot in item["slots"]] != list(range(12)):
            raise ValueError("slot indices must be exactly 0..11")
        selected[token] = item
    if len(selected) != expected:
        raise ValueError("feature manifest does not contain the frozen panel")
    return selected


def _original_batch_cardinalities(path: Path, batch_size: int) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("feature manifest lacks records")
    hashes = sorted({str(slot["image_sha256"]) for row in records for slot in row["slots"]})
    cardinalities: dict[str, int] = {}
    for start in range(0, len(hashes), batch_size):
        batch = hashes[start : start + batch_size]
        for digest in batch:
            cardinalities[digest] = len(batch)
    return cardinalities


def _load_npy(path: Path) -> npt.NDArray[np.generic]:
    with path.open("rb") as stream:
        return cast(npt.NDArray[np.generic], np.lib.format.read_array(stream, allow_pickle=False))  # type: ignore[no-untyped-call]


def _load_rgb(path: Path, expected: str) -> Float32Array:
    payload = path.read_bytes()
    import hashlib

    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"raw image hash mismatch: {path}")
    with Image.open(BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    if rgb.shape != (*RAW_SHAPE, 3) or not np.all(np.isfinite(rgb)):
        raise ValueError("invalid raw image")
    return np.ascontiguousarray(rgb)


def _preprocess(raw_rgb: Float32Array, condition: str) -> torch.Tensor:
    srgb = raw_rgb / np.float32(255.0)
    if condition != "clean":
        srgb = apply_h6_controlled_appearance(srgb, condition)
    tensor = torch.from_numpy(srgb).permute(2, 0, 1).unsqueeze(0)
    tensor = functional.interpolate(tensor, size=RESIZED_SHAPE, mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _extract(
    raw_images: Sequence[Float32Array], conditions: Sequence[str], backbone: Any,
    device: torch.device, cardinalities: Sequence[int], batch_size: int,
) -> dict[str, Float32Array]:
    items: dict[tuple[str, int], list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for condition in conditions:
        for slot, image in enumerate(raw_images):
            required = int(cardinalities[slot])
            if required <= 0 or required > batch_size:
                raise ValueError("invalid original cache batch cardinality")
            items[(condition, required)].append((slot, _preprocess(image, condition)))
    outputs: dict[str, list[Float32Array | None]] = {condition: [None] * len(raw_images) for condition in conditions}
    for (condition, required), grouped in items.items():
        for start in range(0, len(grouped), required):
            chunk = grouped[start : start + required]
            tensors = [tensor for _slot, tensor in chunk]
            tensors.extend([tensors[-1]] * (required - len(tensors)))
            batch = torch.cat(tensors).to(device=device, dtype=torch.bfloat16)
            with torch.inference_mode():
                layers = backbone.get_intermediate_layers(batch, n=4, reshape=True, return_class_token=False, norm=True)
                features = torch.stack(tuple(layers)).mean(dim=0)
            if features.dtype != torch.bfloat16 or tuple(features.shape[1:]) != (1280, 18, 32):
                raise RuntimeError("DINO feature contract changed")
            dense_cpu = features[: len(chunk)].float().cpu().numpy()
            for (slot, _tensor), dense in zip(chunk, dense_cpu, strict=True):
                outputs[condition][slot] = np.asarray(dense, dtype=np.float32)
    return {name: np.stack(cast(list[Float32Array], values)) for name, values in outputs.items()}


def _predict(model: QuotientFieldPredictor, dense: Float32Array, patch_xy: Float64Array, mask: BoolArray, device: torch.device) -> Float32Array:
    dense_tensor = torch.from_numpy(dense).to(device=device)
    xy_tensor = torch.from_numpy(patch_xy).to(device=device)
    mask_tensor = torch.from_numpy(mask).to(device=device)
    sampled = sample_dense_features_torch(dense_tensor, xy_tensor, mask_tensor)
    with torch.inference_mode():
        prediction = model(sampled.unsqueeze(0), mask_tensor.unsqueeze(0))[0]
    if not bool(torch.isfinite(prediction).all().item()) or abs(float(prediction.mean(dtype=torch.float64))) > 1.0e-5:
        raise FloatingPointError("invalid Stage-P prediction")
    return np.asarray(prediction.cpu().numpy(), dtype=np.float32)


def _fit_teacher_prior(records: Sequence[JoinedAnchor]) -> Float64Array:
    if len(records) != 672:
        raise ValueError("fit prior requires exactly 672 fit anchors")
    return np.asarray(
        np.mean(np.stack([load_teacher_arrays(record).label_m for record in records]), axis=0),
        dtype=np.float64,
    )


def _perturbation(raw: Sequence[Float32Array], clean: Float32Array, changed: Float32Array, condition: str) -> dict[str, float]:
    clean64 = clean.astype(np.float64, copy=False)
    changed64 = changed.astype(np.float64, copy=False)
    numerator = float(np.sum(clean64 * changed64))
    denominator = float(np.linalg.norm(clean64) * np.linalg.norm(changed64))
    pixel = np.mean([
        np.mean(np.abs(apply_h6_controlled_appearance(image / np.float32(255.0), condition) - image / np.float32(255.0)))
        for image in raw
    ]) if condition != "clean" else 0.0
    return {
        "pixel_srgb_mae": float(pixel),
        "dense_feature_rms": float(np.sqrt(np.mean(np.square(changed64 - clean64)))),
        "dense_feature_one_minus_cosine": float(1.0 - numerator / denominator) if denominator > 0 else 0.0,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-p-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke-one-anchor", action="store_true")
    args = parser.parse_args()
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)
    fields_root = args.output_directory / "fields"
    fields_root.mkdir()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    configured_conditions = tuple(row["name"] for row in config["appearance_conditions"]["confirmatory"] + config["appearance_conditions"]["secondary_severity_ladder"])
    if configured_conditions != H6_APPEARANCE_CONDITIONS:
        raise ValueError("code and registered H6 conditions disagree")
    input_hashes = {
        "source_split_manifest_sha256": sha256_file(args.source_split_manifest),
        "role_manifest_sha256": sha256_file(args.role_manifest),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "label_manifest_sha256": sha256_file(args.label_manifest),
        "stage_p_checkpoint_sha256": sha256_file(args.stage_p_checkpoint),
        "backbone_checkpoint_sha256": sha256_file(args.dino_checkpoint),
    }
    for key, actual in input_hashes.items():
        if actual != frozen[key]:
            raise ValueError(f"{key} disagrees with the frozen R004H6 config")
    dino_source_sha256, source_files = dino_python_tree_sha256(args.dino_repo)
    if dino_source_sha256 != frozen["backbone_source_sha256"] or source_files != EXPECTED_DINO_FILES:
        raise ValueError("DINO source tree changed")

    fit_groups = _groups(args.role_manifest, column="role", value="fit", expected=42)
    development_groups = _groups(args.source_split_manifest, column="split", value="development", expected=24)
    if set(fit_groups) & set(development_groups):
        raise ValueError("fit/development overlap")
    joined = load_joined_anchors(args.feature_manifest, args.label_manifest, feature_root=args.feature_root, label_root=args.label_root, allowed_splits=("train", "development"))
    fit_records = tuple(row for row in joined if row.recording_group in set(fit_groups))
    development_records = tuple(row for row in joined if row.recording_group in set(development_groups))
    if len(development_records) != 384 or any(row.split != "development" for row in development_records):
        raise ValueError("development panel is incomplete")
    order = {group: index for index, group in enumerate(development_groups)}
    development_records = tuple(sorted(development_records, key=lambda row: (order[row.recording_group], row.anchor_index)))
    execution_records = development_records[:1] if args.smoke_one_anchor else development_records
    by_group_anchor = {(row.recording_group, row.anchor_index): row for row in development_records}
    donor_map = deterministic_group_derangement(development_groups)
    visual = _feature_records(args.feature_manifest, frozenset(development_groups), 384)
    original_cardinalities = _original_batch_cardinalities(args.feature_manifest, args.batch_size)
    fit_prior = _fit_teacher_prior(fit_records)

    torch.manual_seed(20261024)
    torch.cuda.manual_seed_all(20261024)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    backbone = torch.hub.load(  # type: ignore[no-untyped-call]
        str(args.dino_repo), "dinov3_vith16plus", source="local", weights=str(args.dino_checkpoint)
    )
    backbone.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    if sum(p.numel() for p in backbone.parameters()) != EXPECTED_DINO_PARAMETERS:
        raise ValueError("DINO parameter count changed")
    model = QuotientFieldPredictor().to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.stage_p_checkpoint, map_location="cpu", weights_only=False)
    for key in ("feature_manifest_sha256", "label_manifest_sha256", "role_manifest_sha256"):
        if checkpoint.get(key) != input_hashes[key]:
            raise ValueError(f"Stage-P checkpoint {key} provenance mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    neutral = np.empty((*RAW_SHAPE, 3), dtype=np.float32)
    neutral[...] = np.asarray((0.485, 0.456, 0.406), dtype=np.float32) * np.float32(255.0)
    removed_dense = _extract([neutral] * 12, ("clean",), backbone, device, [args.batch_size] * 12, args.batch_size)["clean"]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    ledgers: list[dict[str, Any]] = []
    clean_replay_mismatch_count = 0
    for number, record in enumerate(execution_records, start=1):
        recipient = visual[record.scene_token]
        slots = cast(list[dict[str, Any]], recipient["slots"])
        raw = [_load_rgb(_inside(args.sensor_root, str(slot["image_relative_path"])), str(slot["image_sha256"])) for slot in slots]
        cardinalities = [original_cardinalities[str(slot["image_sha256"])] for slot in slots]
        dense = _extract(raw, H6_APPEARANCE_CONDITIONS, backbone, device, cardinalities, args.batch_size)
        for slot_index, cached_path in enumerate(record.dense_feature_paths):
            replay_bits = torch.from_numpy(dense["clean"][slot_index]).to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
            if not np.array_equal(load_dense_bfloat16_bits(cached_path), replay_bits):
                clean_replay_mismatch_count += 1
                raise RuntimeError("clean DINO replay is not bit exact")
        donor_record = by_group_anchor[(donor_map[record.recording_group], record.anchor_index)]
        donor_slots = cast(list[dict[str, Any]], visual[donor_record.scene_token]["slots"])
        donor_raw = [_load_rgb(_inside(args.sensor_root, str(slot["image_relative_path"])), str(slot["image_sha256"])) for slot in donor_slots]
        donor_cardinalities = [original_cardinalities[str(slot["image_sha256"])] for slot in donor_slots]
        mismatched_dense = _extract(donor_raw, ("clean",), backbone, device, donor_cardinalities, args.batch_size)["clean"]

        patch_xy, mask = validate_projection_arrays(_load_npy(record.projection_xy_path), _load_npy(record.projection_mask_path))
        teacher = load_teacher_arrays(record).label_m
        predictions = {condition: _predict(model, value, patch_xy, mask, device) for condition, value in dense.items()}
        clean_prediction = predictions["clean"]
        mismatch_prediction = _predict(model, mismatched_dense, patch_xy, mask, device)
        removed_prediction = _predict(model, removed_dense, patch_xy, mask, device)
        condition_metrics = {name: asdict(measure_field_mechanism(value, teacher, clean_prediction)) for name, value in predictions.items()}
        prior_metrics = asdict(measure_field_mechanism(fit_prior, teacher, clean_prediction))
        mismatch_metrics = asdict(measure_field_mechanism(mismatch_prediction, teacher, clean_prediction))
        removed_metrics = asdict(measure_field_mechanism(removed_prediction, teacher, clean_prediction))
        perturbations = {name: _perturbation(raw, dense["clean"], value, name) for name, value in dense.items()}
        field_path = fields_root / f"{record.scene_token}.npz"
        np.savez_compressed(field_path, teacher=teacher.astype(np.float32), fit_teacher_prior=fit_prior.astype(np.float32), mismatched_clean=mismatch_prediction, image_removed=removed_prediction, **{name: value for name, value in predictions.items()})
        ledgers.append({
            "recording_group": record.recording_group,
            "anchor_index": record.anchor_index,
            "scene_token": record.scene_token,
            "donor_recording_group": donor_record.recording_group,
            "donor_scene_token": donor_record.scene_token,
            "conditions": condition_metrics,
            "perturbation_ledger": perturbations,
            "zero_task_mae_m": float(np.mean(np.abs(oriented_grid_edges(teacher)))),
            "fit_teacher_prior": prior_metrics,
            "mismatched_clean": mismatch_metrics,
            "image_removed": removed_metrics,
            "field_artifact": field_path.relative_to(args.output_directory).as_posix(),
            "field_artifact_sha256": sha256_file(field_path),
        })
        if number % 16 == 0 or number == len(execution_records):
            elapsed = time.perf_counter() - started
            peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
            output_gib = sum(path.stat().st_size for path in args.output_directory.rglob("*") if path.is_file()) / 1024**3
            if elapsed > float(config["resources"]["maximum_wall_minutes"]) * 60 or peak_gib > float(config["resources"]["maximum_peak_gpu_gib"]) or output_gib > float(config["resources"]["maximum_output_gib"]):
                raise RuntimeError("R004H6 exceeded a registered resource cap")
            progress = {"experiment_id": config["experiment_id"], "status": "IN_PROGRESS", "completed_anchor_count": number, "total_anchor_count": len(execution_records), "elapsed_seconds": elapsed, "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)), "clean_replay_mismatch_count": clean_replay_mismatch_count, "calibration_test_or_target_accessed": False}
            _write_json(args.output_directory / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)

    if args.smoke_one_anchor:
        smoke = {"experiment_id": "R004H6-one-anchor-integration-smoke-v1", "status": "PASS", "implementation_revision": args.implementation_revision, "config_sha256": sha256_file(args.config), "scientific_metrics_reported_or_used": False, "calibration_test_or_target_accessed": False}
        _write_json(args.output_directory / "smoke_result.json", smoke)
        print(json.dumps(smoke, sort_keys=True))
        return
    summary = summarize_heldout_ledgers(ledgers, all_conditions=H6_APPEARANCE_CONDITIONS)
    result = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": summary["decision"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        **input_hashes,
        "dino_source_sha256": dino_source_sha256,
        "development_group_count": 24,
        "development_anchor_count": 384,
        "donor_derangement": donor_map,
        "clean_replay_mismatch_count": clean_replay_mismatch_count,
        "macro_group_summary": summary,
        "anchor_ledgers": ledgers,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "calibration_test_or_target_accessed": False,
        "claim_boundary": "Current frozen model on historically exposed but current-model-held-out development panel only; no untouched test, population, weather, target, driving, safety, or novelty claim.",
    }
    _write_json(args.output_directory / "result.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "anchor_ledgers"}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
