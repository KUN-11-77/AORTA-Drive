"""Build replayable dense DINO and projection artifacts for frozen R004 anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from aorta_drive.protocol.training_join import (
    TrainingAnchor,
    load_training_anchors,
    select_preflight_anchors,
)
from aorta_drive.vision.feature_cache import (
    dino_python_tree_sha256,
    sha256_file,
    validate_projection_arrays,
    write_npy_exact,
)
from aorta_drive.vision.projection import (
    project_points_to_patch_grid,
    transform_current_to_history,
    yaw_from_wxyz,
)

CAMERAS = ("cam_l0", "cam_f0", "cam_r0")
CAMERA_LABELS = ("left", "front", "right")
TIME_SET_SHA256 = "33e6df12d2d0b3b9a4c2fa93789addcb437bc15619c5c559049000dce21fa1a9"
CHECKPOINT_SHA256 = "7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488"
DINO_SOURCE_SHA256 = "98e2b6ff6adc14d31995081657c03128a1153c954f0ea2ce50b3a61cc3fce3b3"
DINO_SOURCE_FILE_COUNT = 151
EXPECTED_PARAMETERS = 840_633_600
EXPECTED_DIMENSION = 1280
PATCH_SHAPE = (18, 32)
RAW_SHAPE = (1080, 1920)
RESIZED_SHAPE = (288, 512)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _grid_points() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    coordinates = np.arange(-31.5, 32.0, 1.0, dtype=np.float64)
    xx, yy = np.meshgrid(coordinates, coordinates)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    xyz = np.column_stack((xy, np.zeros(len(xy), dtype=np.float64)))
    return xy, xyz


def _camera_record(frame: dict[str, Any], camera_name: str) -> dict[str, Any]:
    cameras = {str(name).lower(): value for name, value in frame["cams"].items()}
    if camera_name not in cameras or not isinstance(cameras[camera_name], dict):
        raise ValueError(f"missing or invalid {camera_name} metadata")
    return cast(dict[str, Any], cameras[camera_name])


def _pose(frame: dict[str, Any]) -> tuple[npt.NDArray[np.float64], float]:
    translation = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("invalid ego translation")
    return translation[:2], yaw_from_wxyz(frame["ego2global_rotation"])


def _artifact_id(anchor: TrainingAnchor) -> str:
    return hashlib.sha256(
        f"{anchor.log_name}/{anchor.scene_token}".encode()
    ).hexdigest()[:20]


def _preprocess_image(path: Path, torch: Any, functional: Any) -> Any:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    if tuple(rgb.shape[:2]) != RAW_SHAPE:
        raise RuntimeError(f"unexpected raw image shape for {path}: {rgb.shape[:2]}")
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
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


def _plan_artifacts(
    anchors: tuple[TrainingAnchor, ...],
    *,
    log_root: Path,
    sensor_root: Path,
    output_root: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    points_xy, _points_xyz = _grid_points()
    by_log: dict[str, list[TrainingAnchor]] = defaultdict(list)
    for anchor in anchors:
        by_log[anchor.log_name].append(anchor)
    records: list[dict[str, object]] = []
    images_by_path: dict[str, dict[str, object]] = {}
    images_by_hash: dict[str, dict[str, object]] = {}

    for log_number, log_name in enumerate(sorted(by_log), start=1):
        log_path = log_root / f"{log_name}.pkl"
        if not log_path.is_file():
            raise FileNotFoundError(log_path)
        with log_path.open("rb") as stream:
            frames = pickle.load(stream)  # noqa: S301 - official frozen NAVSIM input
        token_to_index = {str(frame["token"]): index for index, frame in enumerate(frames)}
        for anchor in sorted(by_log[log_name], key=lambda item: item.anchor_index):
            if anchor.scene_token not in token_to_index:
                raise RuntimeError(f"missing scene token in source log: {anchor.scene_token}")
            current_index = token_to_index[anchor.scene_token]
            history_indices = list(range(current_index - 3, current_index + 1))
            if history_indices[0] < 0 or current_index + 10 >= len(frames):
                raise RuntimeError(f"short source time window: {anchor.scene_token}")
            current_translation, current_yaw = _pose(frames[current_index])
            coordinates = np.zeros((4096, 12, 2), dtype=np.float64)
            support = np.zeros((4096, 12), dtype=np.bool_)
            slots: list[dict[str, object]] = []
            for history_order, frame_index in enumerate(history_indices):
                frame = frames[frame_index]
                history_translation, history_yaw = _pose(frame)
                history_xy = transform_current_to_history(
                    points_xy,
                    current_translation,
                    current_yaw,
                    history_translation,
                    history_yaw,
                )
                history_xyz = np.column_stack(
                    (history_xy, np.zeros(len(history_xy), dtype=np.float64))
                )
                for camera_order, (camera_name, camera_label) in enumerate(
                    zip(CAMERAS, CAMERA_LABELS, strict=True)
                ):
                    slot_index = history_order * len(CAMERAS) + camera_order
                    camera = _camera_record(frame, camera_name)
                    relative_image_path = str(camera["data_path"])
                    image_path = sensor_root / relative_image_path
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    if relative_image_path not in images_by_path:
                        image_sha256 = sha256_file(image_path)
                        feature_relative_path = (
                            Path("dense_features")
                            / image_sha256[:2]
                            / f"{image_sha256}.npy"
                        ).as_posix()
                        new_image_record: dict[str, object] = {
                            "image_relative_path": relative_image_path,
                            "image_sha256": image_sha256,
                            "feature_relative_path": feature_relative_path,
                        }
                        images_by_path[relative_image_path] = new_image_record
                        prior = images_by_hash.get(image_sha256)
                        if prior is not None and prior["image_relative_path"] != relative_image_path:
                            new_image_record["content_alias_of"] = prior[
                                "image_relative_path"
                            ]
                        else:
                            images_by_hash[image_sha256] = new_image_record
                    image_record = images_by_path[relative_image_path]
                    rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
                    translation = np.asarray(
                        camera["sensor2lidar_translation"], dtype=np.float64
                    )
                    intrinsics = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
                    distortion = np.asarray(camera["distortion"], dtype=np.float64)
                    projection = project_points_to_patch_grid(
                        history_xyz,
                        rotation,
                        translation,
                        intrinsics,
                        distortion,
                        raw_shape=RAW_SHAPE,
                        resized_shape=RESIZED_SHAPE,
                        patch_size=16,
                        maximum_normalized_radius=1.0,
                    )
                    valid = projection.valid_for_bilinear_patch_sampling
                    coordinates[valid, slot_index] = projection.patch_xy[valid]
                    support[:, slot_index] = valid
                    slots.append(
                        {
                            "slot_index": slot_index,
                            "history_offset": history_order - 3,
                            "camera": camera_label,
                            **image_record,
                        }
                    )
            coordinates, support = validate_projection_arrays(coordinates, support)
            identifier = _artifact_id(anchor)
            xy_relative = (Path("projections") / f"{identifier}.patch_xy.npy").as_posix()
            mask_relative = (Path("projections") / f"{identifier}.mask.npy").as_posix()
            xy_sha256 = write_npy_exact(output_root / xy_relative, coordinates)
            mask_sha256 = write_npy_exact(output_root / mask_relative, support)
            records.append(
                {
                    "recording_group": anchor.recording_group,
                    "anchor_index": anchor.anchor_index,
                    "split": anchor.split,
                    "log_name": anchor.log_name,
                    "scene_token": anchor.scene_token,
                    "projection_xy_relative_path": xy_relative,
                    "projection_xy_sha256": xy_sha256,
                    "projection_mask_relative_path": mask_relative,
                    "projection_mask_sha256": mask_sha256,
                    "supported_node_slot_count": int(np.sum(support)),
                    "all_missing_node_count": int(np.sum(~np.any(support, axis=1))),
                    "slots": slots,
                }
            )
        print(f"planned log {log_number}/{len(by_log)}: {log_name}", flush=True)
    return records, images_by_hash


def build(args: argparse.Namespace) -> dict[str, object]:
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    anchors = select_preflight_anchors(
        load_training_anchors(args.time_set),
        maximum_anchors=args.max_anchors,
        maximum_recording_groups=args.max_recording_groups,
    )
    if sha256_file(args.time_set) != TIME_SET_SHA256:
        raise RuntimeError("frozen R004 time-set SHA-256 mismatch")
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("frozen DINO checkpoint SHA-256 mismatch")
    source_sha256, source_file_count = dino_python_tree_sha256(args.dino_repo)
    if source_sha256 != DINO_SOURCE_SHA256 or source_file_count != DINO_SOURCE_FILE_COUNT:
        raise RuntimeError(
            "frozen DINO Python source tree mismatch: "
            f"sha={source_sha256}, files={source_file_count}"
        )

    args.output_directory.mkdir(parents=True)
    started = time.perf_counter()
    records, images = _plan_artifacts(
        anchors,
        log_root=args.log_root,
        sensor_root=args.sensor_root,
        output_root=args.output_directory,
    )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = torch.hub.load(  # type: ignore[no-untyped-call]
        str(args.dino_repo),
        "dinov3_vith16plus",
        source="local",
        weights=str(args.checkpoint),
    )
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"unexpected DINO parameter count: {parameter_count}")
    if int(model.embed_dim) != EXPECTED_DIMENSION or int(model.patch_size) != 16:
        raise RuntimeError("unexpected DINO embedding or patch-size contract")

    image_items = sorted(images.items())
    feature_hashes: dict[str, str] = {}
    torch.cuda.reset_peak_memory_stats(device)
    for batch_start in range(0, len(image_items), args.batch_size):
        batch_items = image_items[batch_start : batch_start + args.batch_size]
        tensors = [
            _preprocess_image(
                args.sensor_root / str(record["image_relative_path"]), torch, functional
            )
            for _image_sha, record in batch_items
        ]
        batch = torch.cat(tensors, dim=0).to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            layers = model.get_intermediate_layers(
                batch,
                n=4,
                reshape=True,
                return_class_token=False,
                norm=True,
            )
            features = torch.stack(tuple(layers), dim=0).mean(dim=0)
        expected_shape = (len(batch_items), EXPECTED_DIMENSION, *PATCH_SHAPE)
        if features.dtype != torch.bfloat16 or tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected dense feature contract: {features.dtype} {tuple(features.shape)}"
            )
        if not bool(torch.isfinite(features).all().item()):
            raise RuntimeError("DINO produced a nonfinite dense feature")
        bits = features.contiguous().view(torch.uint16).cpu().numpy()
        for index, (image_sha256, record) in enumerate(batch_items):
            feature_path = args.output_directory / str(record["feature_relative_path"])
            feature_hashes[image_sha256] = write_npy_exact(feature_path, bits[index])
        completed = min(batch_start + len(batch_items), len(image_items))
        print(f"extracted {completed}/{len(image_items)} unique image features", flush=True)

    for record in records:
        for slot in cast(list[dict[str, object]], record["slots"]):
            slot["feature_sha256"] = feature_hashes[str(slot["image_sha256"])]
    artifact_bytes = sum(path.stat().st_size for path in args.output_directory.rglob("*.npy"))
    stable_payload = {
        "time_set_sha256": TIME_SET_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dino_source_sha256": DINO_SOURCE_SHA256,
        "records": records,
    }
    result: dict[str, object] = {
        "experiment_id": "R004E-B-frozen-visual-cache",
        "implementation_revision": args.implementation_revision,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "allowed_splits": sorted({anchor.split for anchor in anchors}),
        "calibration_test_or_target_accessed": False,
        "recording_group_count": len({anchor.recording_group for anchor in anchors}),
        "anchor_count": len(anchors),
        "slot_count": 12,
        "image_reference_count": len(anchors) * 12,
        "unique_image_count": len(image_items),
        "artifact_bytes": artifact_bytes,
        "duration_seconds": time.perf_counter() - started,
        "worker_contract": {
            "device": str(device),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "cuda_device_name": torch.cuda.get_device_name(device),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "feature_contract": {
            "model": "dinov3_vith16plus",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "dino_python_source_tree_sha256": source_sha256,
            "dino_python_source_file_count": source_file_count,
            "parameter_count": parameter_count,
            "preprocessing": (
                "RGB_288x512_bicubic_align_corners_false_antialias_"
                "imagenet_normalization"
            ),
            "feature": "normalized_last_4_layer_mean",
            "storage_dtype": "uint16_bfloat16_bits",
            "shape": [EXPECTED_DIMENSION, *PATCH_SHAPE],
        },
        "projection_contract": {
            "coordinate_dtype": "float64",
            "coordinate_shape": [4096, 12, 2],
            "mask_dtype": "bool",
            "mask_shape": [4096, 12],
            "invalid_coordinate_storage": "exact_zero",
            "slot_order": "history_oldest_to_current_then_left_front_right",
        },
        "input_hashes": {
            "time_set_sha256": TIME_SET_SHA256,
            "builder_script_sha256": sha256_file(Path(__file__)),
        },
        "logical_manifest_sha256": _canonical_sha256(stable_payload),
        "records": records,
    }
    manifest_path = args.output_directory / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--max-anchors", type=int)
    parser.add_argument("--max-recording-groups", type=int)
    args = parser.parse_args()
    result = build(args)
    summary = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
