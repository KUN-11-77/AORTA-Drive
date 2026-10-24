"""Run the frozen fit-only paired appearance mechanism diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from aorta_drive.evaluation.appearance_mechanism import (
    measure_field_mechanism,
    mechanism_condition_supported,
    oriented_grid_edges,
)
from aorta_drive.models.quotient_field import QuotientFieldPredictor
from aorta_drive.training.r004ec import (
    JoinedAnchor,
    load_joined_anchors,
    load_teacher_arrays,
    sample_dense_features_torch,
)
from aorta_drive.vision.appearance import (
    APPEARANCE_CONDITIONS,
    apply_controlled_appearance,
)
from aorta_drive.vision.feature_cache import (
    bfloat16_bits_to_float32,
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
GenericArray = npt.NDArray[np.generic]


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes its declared root")
    return path


def _fit_groups(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = tuple(row["recording_group"] for row in rows if row["role"] == "fit")
    if len(groups) != 42 or len(set(groups)) != 42:
        raise ValueError("R004H5 requires exactly 42 fit recording groups")
    return groups


def _feature_records(path: Path, fit_groups: frozenset[str]) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("feature manifest lacks records")
    selected: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict) or item.get("recording_group") not in fit_groups:
            continue
        token = str(item["scene_token"])
        if token in selected:
            raise ValueError("duplicate fit scene token")
        slots = item.get("slots")
        if not isinstance(slots, list) or len(slots) != 12:
            raise ValueError("fit feature record must contain 12 slots")
        item["slots"] = sorted(slots, key=lambda value: int(value["slot_index"]))
        if [int(value["slot_index"]) for value in item["slots"]] != list(range(12)):
            raise ValueError("slot indices must be exactly 0..11")
        selected[token] = cast(dict[str, Any], item)
    if len(selected) != 672:
        raise ValueError("feature manifest does not contain 672 fit anchors")
    return selected


def _original_batch_cardinalities(
    path: Path, *, batch_size: int
) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or batch_size <= 0:
        raise ValueError("invalid feature manifest or batch size")
    hashes = sorted(
        {
            str(slot["image_sha256"])
            for record in records
            if isinstance(record, dict) and isinstance(record.get("slots"), list)
            for slot in record["slots"]
        }
    )
    if len(hashes) != int(payload.get("unique_image_count", -1)):
        raise ValueError("unique image hashes disagree with the frozen feature manifest")
    cardinalities: dict[str, int] = {}
    for start in range(0, len(hashes), batch_size):
        batch = hashes[start : start + batch_size]
        for image_hash in batch:
            cardinalities[image_hash] = len(batch)
    return cardinalities


def _load_npy(path: Path) -> GenericArray:
    with path.open("rb") as stream:
        return cast(
            GenericArray,
            np.lib.format.read_array(stream, allow_pickle=False),  # type: ignore[no-untyped-call]
        )


def _load_rgb_verified(path: Path, expected_sha256: str) -> Float32Array:
    payload = path.read_bytes()
    import hashlib

    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"raw image hash mismatch: {path}")
    with Image.open(BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    if rgb.shape != (*RAW_SHAPE, 3) or not np.all(np.isfinite(rgb)):
        raise ValueError(f"invalid frozen raw image: {path}")
    return np.ascontiguousarray(rgb)


def _preprocess(raw_rgb: Float32Array, condition: str) -> torch.Tensor:
    if condition == "clean":
        tensor = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
    else:
        srgb = apply_controlled_appearance(
            raw_rgb / np.float32(255.0), condition
        )
        tensor = torch.from_numpy(srgb).permute(2, 0, 1).unsqueeze(0)
    tensor = functional.interpolate(
        tensor,
        size=RESIZED_SHAPE,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _extract_variants(
    *,
    raw_images: Sequence[Float32Array],
    backbone: Any,
    device: torch.device,
    batch_size: int,
    original_batch_cardinalities: Sequence[int],
) -> dict[str, Float32Array]:
    if len(raw_images) != len(original_batch_cardinalities):
        raise ValueError("raw images and original batch cardinalities must align")
    items: dict[tuple[str, int], list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for condition in APPEARANCE_CONDITIONS:
        for slot, image in enumerate(raw_images):
            required = int(original_batch_cardinalities[slot])
            if required <= 0 or required > batch_size:
                raise ValueError("invalid original cache batch cardinality")
            items[(condition, required)].append((slot, _preprocess(image, condition)))
    outputs: dict[str, list[Float32Array | None]] = {
        condition: [None] * 12 for condition in APPEARANCE_CONDITIONS
    }
    for (condition, required), grouped in items.items():
        for start in range(0, len(grouped), required):
            chunk = grouped[start : start + required]
            actual_count = len(chunk)
            padded = [tensor for _slot, tensor in chunk]
            padded.extend([padded[-1]] * (required - actual_count))
            batch = torch.cat(padded, dim=0).to(
                device=device, dtype=torch.bfloat16
            )
            with torch.inference_mode():
                layers = backbone.get_intermediate_layers(
                    batch,
                    n=4,
                    reshape=True,
                    return_class_token=False,
                    norm=True,
                )
                features = torch.stack(tuple(layers), dim=0).mean(dim=0)
            expected = (required, 1280, 18, 32)
            if features.dtype != torch.bfloat16 or tuple(features.shape) != expected:
                raise RuntimeError("DINO feature contract changed")
            if not bool(torch.isfinite(features).all().item()):
                raise FloatingPointError("DINO produced nonfinite features")
            features_cpu = features[:actual_count].float().cpu().numpy()
            for (slot, _tensor), dense in zip(chunk, features_cpu, strict=True):
                outputs[condition][slot] = np.ascontiguousarray(
                    dense, dtype=np.float32
                )
    stacked: dict[str, Float32Array] = {}
    for condition, slots in outputs.items():
        if any(value is None for value in slots):
            raise AssertionError("incomplete DINO condition output")
        stacked[condition] = np.stack(cast(list[Float32Array], slots), axis=0)
    return stacked


def _predict(
    model: QuotientFieldPredictor,
    dense: Float32Array,
    patch_xy: Float64Array,
    mask: BoolArray,
    device: torch.device,
) -> tuple[Float32Array, torch.Tensor, torch.Tensor]:
    dense_tensor = torch.from_numpy(dense).to(device=device, dtype=torch.float32)
    xy_tensor = torch.from_numpy(patch_xy).to(device=device)
    mask_tensor = torch.from_numpy(mask).to(device=device)
    sampled = sample_dense_features_torch(dense_tensor, xy_tensor, mask_tensor)
    with torch.inference_mode():
        prediction = model(sampled.unsqueeze(0), mask_tensor.unsqueeze(0))[0]
    if not bool(torch.isfinite(prediction).all().item()):
        raise FloatingPointError("Stage-P produced a nonfinite field")
    if abs(float(prediction.mean(dtype=torch.float64).item())) > 1.0e-5:
        raise RuntimeError("Stage-P output violated the zero-gauge contract")
    return np.asarray(prediction.cpu().numpy(), dtype=np.float32), sampled, mask_tensor


def _predict_materialized(
    model: QuotientFieldPredictor,
    features: torch.Tensor,
    mask: torch.Tensor,
) -> Float32Array:
    with torch.inference_mode():
        prediction = model(features.unsqueeze(0), mask.unsqueeze(0))[0]
    if not bool(torch.isfinite(prediction).all().item()):
        raise FloatingPointError("probe produced a nonfinite field")
    return np.asarray(prediction.cpu().numpy(), dtype=np.float32)


def _precompute_fit_priors(
    records: Sequence[JoinedAnchor],
) -> tuple[Float32Array, dict[str, Float64Array]]:
    feature_sum = np.zeros(1280, dtype=np.float64)
    feature_count = 0
    seen_features: set[Path] = set()
    total_teacher = np.zeros((64, 64), dtype=np.float64)
    group_teacher: dict[str, Float64Array] = defaultdict(
        lambda: np.zeros((64, 64), dtype=np.float64)
    )
    for record in records:
        teacher = load_teacher_arrays(record).label_m
        total_teacher += teacher
        group_teacher[record.recording_group] += teacher
        for path in record.dense_feature_paths:
            if path in seen_features:
                continue
            seen_features.add(path)
            dense = bfloat16_bits_to_float32(load_dense_bfloat16_bits(path))
            feature_sum += np.sum(dense, axis=(1, 2), dtype=np.float64)
            feature_count += dense.shape[1] * dense.shape[2]
    if len(seen_features) != 8064 or feature_count != 8064 * 18 * 32:
        raise ValueError("fit feature prior did not use the exact unique image set")
    uniform_feature = np.asarray(feature_sum / feature_count, dtype=np.float32)
    leave_one_group_out = {
        group: (total_teacher - subtotal) / (len(records) - 16)
        for group, subtotal in group_teacher.items()
    }
    return uniform_feature, leave_one_group_out


def _macro_group_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["recording_group"])].append(record)
    if len(grouped) != 42 or any(len(values) != 16 for values in grouped.values()):
        raise ValueError("summary requires 42 complete recording groups")
    condition_summaries: dict[str, dict[str, float | bool]] = {}
    rng = np.random.default_rng(20261024)
    for condition in APPEARANCE_CONDITIONS:
        per_group: list[tuple[float, float, float, float, float]] = []
        for group in sorted(grouped):
            values = grouped[group]
            task = np.mean([row["conditions"][condition]["task_mae_m"] for row in values])
            zero = np.mean([row["zero_task_mae_m"] for row in values])
            appearance = np.mean(
                [row["conditions"][condition]["appearance_mae_m"] for row in values]
            )
            energy = np.mean(
                [row["conditions"]["clean"]["edge_energy_mae_m"] for row in values]
            )
            cosine = np.mean(
                [row["conditions"][condition]["teacher_cosine"] for row in values]
            )
            per_group.append((task, zero, appearance, energy, cosine))
        array = np.asarray(per_group, dtype=np.float64)
        task_margin = array[:, 1] - array[:, 0]
        stability_margin = array[:, 3] - array[:, 2]
        draws = rng.integers(0, len(array), size=(10_000, len(array)))
        task_boot = np.mean(task_margin[draws], axis=1)
        stability_boot = np.mean(stability_margin[draws], axis=1)
        cosine_boot = np.mean(array[:, 4][draws], axis=1)
        macro = np.mean(array, axis=0)
        condition_summaries[condition] = {
            "task_mae_m": float(macro[0]),
            "zero_task_mae_m": float(macro[1]),
            "appearance_mae_m": float(macro[2]),
            "clean_edge_energy_mae_m": float(macro[3]),
            "teacher_cosine": float(macro[4]),
            "task_margin_ci95_low_m": float(np.quantile(task_boot, 0.025)),
            "task_margin_ci95_high_m": float(np.quantile(task_boot, 0.975)),
            "stability_margin_ci95_low_m": float(
                np.quantile(stability_boot, 0.025)
            ),
            "stability_margin_ci95_high_m": float(
                np.quantile(stability_boot, 0.975)
            ),
            "teacher_cosine_ci95_low": float(np.quantile(cosine_boot, 0.025)),
            "teacher_cosine_ci95_high": float(np.quantile(cosine_boot, 0.975)),
            "supported": mechanism_condition_supported(
                visual_task_mae_m=float(macro[0]),
                zero_task_mae_m=float(macro[1]),
                appearance_mae_m=float(macro[2]),
                clean_edge_energy_mae_m=float(macro[3]),
                teacher_cosine=float(macro[4]),
            ),
        }
    return {
        "conditions": condition_summaries,
        "all_nonclean_conditions_supported": all(
            bool(condition_summaries[name]["supported"])
            for name in APPEARANCE_CONDITIONS
            if name != "clean"
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
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
    parser.add_argument("--smoke-scene-token")
    args = parser.parse_args()
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)
    fields_root = args.output_directory / "fields"
    fields_root.mkdir()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    registered_conditions = tuple(
        item["name"] for item in config["appearance_conditions"]
    )
    if registered_conditions != APPEARANCE_CONDITIONS:
        raise ValueError("code and registered appearance conditions disagree")
    if args.batch_size != int(config["frozen_inference"]["dino_batch_size"]):
        raise ValueError("DINO batch size disagrees with the frozen R004H5 config")
    input_hashes = {
        "role_manifest_sha256": sha256_file(args.role_manifest),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "label_manifest_sha256": sha256_file(args.label_manifest),
        "stage_p_checkpoint_sha256": sha256_file(args.stage_p_checkpoint),
        "backbone_checkpoint_sha256": sha256_file(args.dino_checkpoint),
    }
    for key, actual in input_hashes.items():
        if actual != frozen[key]:
            raise ValueError(f"{key} disagrees with the frozen R004H5 config")
    dino_source_sha256, source_files = dino_python_tree_sha256(args.dino_repo)
    if (
        dino_source_sha256 != frozen["backbone_source_sha256"]
        or source_files != EXPECTED_DINO_FILES
    ):
        raise ValueError("DINO source tree disagrees with the frozen contract")

    fit_groups = _fit_groups(args.role_manifest)
    fit_set = frozenset(fit_groups)
    joined = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
        allowed_splits=("train",),
    )
    records = tuple(record for record in joined if record.recording_group in fit_set)
    if len(records) != 672 or any(record.split != "train" for record in records):
        raise ValueError("R004H5 must use exactly the 672 fit anchors")
    order = {group: index for index, group in enumerate(fit_groups)}
    records = tuple(sorted(records, key=lambda row: (order[row.recording_group], row.anchor_index)))
    execution_records: tuple[JoinedAnchor, ...]
    if args.smoke_scene_token is not None:
        matches = tuple(
            record for record in records if record.scene_token == args.smoke_scene_token
        )
        if len(matches) != 1:
            raise ValueError("smoke scene token must identify exactly one fit anchor")
        execution_records = matches
    else:
        execution_records = records[:1] if args.smoke_one_anchor else records
    visual_by_token = _feature_records(args.feature_manifest, fit_set)
    original_cardinalities = _original_batch_cardinalities(
        args.feature_manifest, batch_size=args.batch_size
    )
    started = time.perf_counter()
    uniform_feature, loo_teacher = _precompute_fit_priors(records)

    torch.manual_seed(20261024)
    torch.cuda.manual_seed_all(20261024)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda" or args.batch_size <= 0:
        raise ValueError("R004H5 requires the registered CUDA device and positive batch size")
    backbone = torch.hub.load(  # type: ignore[no-untyped-call]
        str(args.dino_repo),
        "dinov3_vith16plus",
        source="local",
        weights=str(args.dino_checkpoint),
    )
    backbone.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    if sum(parameter.numel() for parameter in backbone.parameters()) != EXPECTED_DINO_PARAMETERS:
        raise ValueError("DINO parameter count changed")
    if int(backbone.embed_dim) != 1280 or int(backbone.patch_size) != 16:
        raise ValueError("DINO embedding or patch-size contract changed")
    model = QuotientFieldPredictor().to(device).eval().requires_grad_(False)
    checkpoint = torch.load(args.stage_p_checkpoint, map_location="cpu", weights_only=False)
    for key in ("feature_manifest_sha256", "label_manifest_sha256", "role_manifest_sha256"):
        if checkpoint.get(key) != input_hashes[key]:
            raise ValueError(f"Stage-P checkpoint {key} provenance mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 4_000_225:
        raise ValueError("Stage-P parameter count changed")

    torch.cuda.reset_peak_memory_stats(device)
    ledgers: list[dict[str, Any]] = []
    clean_replay_mismatch_count = 0
    for number, record in enumerate(execution_records, start=1):
        visual = visual_by_token[record.scene_token]
        slots = cast(list[dict[str, Any]], visual["slots"])
        raw_images: list[Float32Array] = []
        for slot in slots:
            path = _inside(args.sensor_root, str(slot["image_relative_path"]))
            raw_images.append(_load_rgb_verified(path, str(slot["image_sha256"])))
        dense_by_condition = _extract_variants(
            raw_images=raw_images,
            backbone=backbone,
            device=device,
            batch_size=args.batch_size,
            original_batch_cardinalities=[
                original_cardinalities[str(slot["image_sha256"])] for slot in slots
            ],
        )
        for slot_index, cached_path in enumerate(record.dense_feature_paths):
            cached_bits = load_dense_bfloat16_bits(cached_path)
            replay_bits = (
                torch.from_numpy(dense_by_condition["clean"][slot_index])
                .to(torch.bfloat16)
                .contiguous()
                .view(torch.uint16)
                .numpy()
            )
            if not np.array_equal(cached_bits, replay_bits):
                clean_replay_mismatch_count += 1
                raise RuntimeError("clean DINO replay is not bit exact")

        patch_xy, mask = validate_projection_arrays(
            _load_npy(record.projection_xy_path), _load_npy(record.projection_mask_path)
        )
        teacher = load_teacher_arrays(record).label_m
        predictions: dict[str, Float32Array] = {}
        clean_sampled: torch.Tensor | None = None
        mask_tensor: torch.Tensor | None = None
        for condition in APPEARANCE_CONDITIONS:
            prediction, sampled, condition_mask = _predict(
                model,
                dense_by_condition[condition],
                patch_xy,
                mask,
                device,
            )
            predictions[condition] = prediction
            if condition == "clean":
                clean_sampled = sampled
            mask_tensor = condition_mask
        if mask_tensor is None or clean_sampled is None:
            raise AssertionError("appearance conditions are empty")
        clean_prediction = predictions["clean"]
        condition_metrics = {
            condition: asdict(
                measure_field_mechanism(prediction, teacher, clean_prediction)
            )
            for condition, prediction in predictions.items()
        }
        zero_task_mae = float(np.mean(np.abs(oriented_grid_edges(teacher))))
        prior_metrics = asdict(
            measure_field_mechanism(
                loo_teacher[record.recording_group], teacher, clean_prediction
            )
        )

        probe_metrics: dict[str, dict[str, float]] = {}
        for condition, dense in dense_by_condition.items():
            pooled = torch.from_numpy(np.mean(dense, axis=(2, 3))).to(device)
            pooled_nodes = pooled.unsqueeze(0).expand(4096, -1, -1).contiguous()
            pooled_nodes = torch.where(mask_tensor.unsqueeze(-1), pooled_nodes, 0.0)
            probe = _predict_materialized(model, pooled_nodes, mask_tensor)
            probe_metrics[f"spatial_pool_{condition}"] = asdict(
                measure_field_mechanism(probe, teacher, clean_prediction)
            )
        uniform = torch.from_numpy(uniform_feature).to(device)
        uniform_nodes = uniform.view(1, 1, -1).expand(4096, 12, -1).contiguous()
        uniform_nodes = torch.where(mask_tensor.unsqueeze(-1), uniform_nodes, 0.0)
        uniform_prediction = _predict_materialized(model, uniform_nodes, mask_tensor)
        probe_metrics["uniform_fit_mean_DINO"] = asdict(
            measure_field_mechanism(uniform_prediction, teacher, clean_prediction)
        )
        for camera in range(3):
            dropped = mask_tensor.clone()
            dropped[:, camera::3] = False
            probe = _predict_materialized(model, clean_sampled, dropped)
            probe_metrics[f"drop_camera_{camera}"] = asdict(
                measure_field_mechanism(probe, teacher, clean_prediction)
            )
        for history in range(4):
            dropped = mask_tensor.clone()
            dropped[:, history * 3 : (history + 1) * 3] = False
            probe = _predict_materialized(model, clean_sampled, dropped)
            probe_metrics[f"drop_history_{history}"] = asdict(
                measure_field_mechanism(probe, teacher, clean_prediction)
            )

        field_path = fields_root / f"{record.scene_token}.npz"
        np.savez_compressed(
            field_path,
            teacher=teacher.astype(np.float32),
            **{name: value.astype(np.float32) for name, value in predictions.items()},
        )
        ledger: dict[str, Any] = {
            "recording_group": record.recording_group,
            "anchor_index": record.anchor_index,
            "scene_token": record.scene_token,
            "conditions": condition_metrics,
            "zero_task_mae_m": zero_task_mae,
            "leave_one_group_out_teacher_prior": prior_metrics,
            "probes": probe_metrics,
            "field_artifact": field_path.relative_to(args.output_directory).as_posix(),
            "field_artifact_sha256": sha256_file(field_path),
        }
        ledgers.append(ledger)
        if number % 16 == 0 or number == len(execution_records):
            elapsed = time.perf_counter() - started
            peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
            output_gib = sum(
                path.stat().st_size
                for path in args.output_directory.rglob("*")
                if path.is_file()
            ) / 1024**3
            if elapsed > float(config["resources"]["maximum_wall_minutes"]) * 60:
                raise RuntimeError("R004H5 exceeded the registered wall-time cap")
            if peak_gib > float(config["resources"]["maximum_peak_gpu_gib"]):
                raise RuntimeError("R004H5 exceeded the registered GPU cap")
            if output_gib > float(config["resources"]["maximum_output_gib"]):
                raise RuntimeError("R004H5 exceeded the registered output cap")
            progress: dict[str, Any] = {
                "experiment_id": config["experiment_id"],
                "status": "IN_PROGRESS",
                "implementation_revision": args.implementation_revision,
                "config_sha256": sha256_file(args.config),
                "completed_anchor_count": number,
                "fit_anchor_count": len(execution_records),
                "elapsed_seconds": elapsed,
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
                "clean_replay_mismatch_count": clean_replay_mismatch_count,
                "prohibited_payload_accessed": False,
            }
            _write_json(args.output_directory / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)

    if args.smoke_one_anchor or args.smoke_scene_token is not None:
        smoke_result: dict[str, Any] = {
            "experiment_id": "R004H5-one-anchor-integration-smoke-v1",
            "status": "PASS",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_revision": args.implementation_revision,
            "config_sha256": sha256_file(args.config),
            "script_sha256": sha256_file(Path(__file__)),
            **input_hashes,
            "fit_anchor_count": 1,
            "appearance_condition_count": len(APPEARANCE_CONDITIONS),
            "clean_replay_mismatch_count": clean_replay_mismatch_count,
            "all_registered_inference_and_probe_branches_exercised": True,
            "scientific_metrics_reported_or_used_for_tuning": False,
            "prohibited_payload_accessed": False,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "claim_boundary": (
                "One fit-anchor integration smoke only. Scientific metric values are not "
                "reported or used; no mechanism or performance claim is supported."
            ),
        }
        _write_json(args.output_directory / "smoke_result.json", smoke_result)
        print(json.dumps(smoke_result, ensure_ascii=False, indent=2, sort_keys=True))
        return

    summary = _macro_group_summary(ledgers)
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": (
            "MECHANISM_SUPPORTED_ON_FIT_CONTROLLED_APPEARANCE"
            if summary["all_nonclean_conditions_supported"]
            else "MECHANISM_UNSUPPORTED_ON_FIT_CONTROLLED_APPEARANCE"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        **input_hashes,
        "dino_source_sha256": dino_source_sha256,
        "fit_group_count": 42,
        "fit_anchor_count": 672,
        "appearance_conditions": list(APPEARANCE_CONDITIONS),
        "clean_replay_mismatch_count": clean_replay_mismatch_count,
        "macro_group_summary": summary,
        "anchor_ledgers": ledgers,
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "prohibited_payload_accessed": False,
        "claim_boundary": (
            "Fit-role controlled photometric mechanism diagnostic only. It does not establish "
            "held-out source generalization, named weather, target OOD, conformal coverage, "
            "driving benefit, safety improvement, Gate B, or novelty."
        ),
    }
    _write_json(args.output_directory / "result.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "anchor_ledgers"}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

