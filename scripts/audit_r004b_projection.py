"""Audit the R004B physical-node to DINO-token projection on source metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from aorta_drive.vision.projection import (
    project_points_to_patch_grid,
    transform_current_to_history,
    yaw_from_wxyz,
)

CAMERAS = ("cam_l0", "cam_f0", "cam_r0")
ALLOWED_SPLITS = {"train", "development"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if camera_name not in cameras:
        raise KeyError(f"missing {camera_name} in frame camera metadata")
    record = cameras[camera_name]
    if not isinstance(record, dict):
        raise TypeError(f"invalid {camera_name} metadata")
    return cast(dict[str, Any], record)


def _pose(frame: dict[str, Any]) -> tuple[npt.NDArray[np.float64], float]:
    translation = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("invalid ego translation")
    yaw = yaw_from_wxyz(frame["ego2global_rotation"])
    return translation[:2], yaw


def _image_shape(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.height), int(image.width)


def _calibration_fingerprint(camera: dict[str, Any]) -> str:
    value = {
        "sensor2lidar_rotation": np.asarray(
            camera["sensor2lidar_rotation"], dtype=np.float64
        ).tolist(),
        "sensor2lidar_translation": np.asarray(
            camera["sensor2lidar_translation"], dtype=np.float64
        ).tolist(),
        "cam_intrinsic": np.asarray(camera["cam_intrinsic"], dtype=np.float64).tolist(),
        "distortion": np.asarray(camera["distortion"], dtype=np.float64).tolist(),
    }
    return _canonical_sha256(value)


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    """Run a metadata-only projection audit without labels or model outputs."""

    requested_splits = {part.strip() for part in args.splits.split(",") if part.strip()}
    if not requested_splits or not requested_splits <= ALLOWED_SPLITS:
        raise ValueError("R004B audit may use only train and development splits")
    with args.time_set.open(encoding="utf-8", newline="") as stream:
        anchors = [row for row in csv.DictReader(stream) if row["split"] in requested_splits]
    anchors.sort(key=lambda row: (row["recording_group"], int(row["anchor_index"])))
    if args.max_anchors is not None:
        if args.max_anchors <= 0:
            raise ValueError("max_anchors must be positive")
        anchors = anchors[: args.max_anchors]
    if not anchors:
        raise ValueError("no anchors selected")

    points_xy, points_xyz = _grid_points()
    radius = np.linalg.norm(points_xy, axis=1)
    radial_bands = ((0.0, 8.0), (8.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 46.0))
    by_log: dict[str, list[dict[str, str]]] = defaultdict(list)
    for anchor in anchors:
        by_log[anchor["log_name"]].append(anchor)

    missing_files: list[str] = []
    invalid_windows: list[str] = []
    image_shapes: dict[str, Counter[str]] = {camera: Counter() for camera in CAMERAS}
    calibration_fingerprints: dict[str, Counter[str]] = {camera: Counter() for camera in CAMERAS}
    maximum_distortion = 0.0
    maximum_rotation_orthogonality_error = 0.0
    maximum_rotation_determinant_error = 0.0
    view_rates: dict[str, list[float]] = defaultdict(list)
    all_history_rates: list[float] = []
    current_only_rates: list[float] = []
    observation_histogram: Counter[int] = Counter()
    radial_visible = {f"{low:g}-{high:g}": [0, 0] for low, high in radial_bands}
    per_anchor: list[dict[str, object]] = []

    for log_number, log_name in enumerate(sorted(by_log), start=1):
        log_path = args.log_root / f"{log_name}.pkl"
        if not log_path.is_file():
            missing_files.append(f"log:{log_name}")
            continue
        with log_path.open("rb") as stream:
            frames = pickle.load(stream)  # noqa: S301 - official frozen NAVSIM input
        token_to_index = {str(frame["token"]): index for index, frame in enumerate(frames)}
        for anchor in by_log[log_name]:
            token = anchor["scene_token"]
            if token not in token_to_index:
                invalid_windows.append(f"missing-token:{token}")
                continue
            current_index = token_to_index[token]
            history_indices = list(range(current_index - 3, current_index + 1))
            if history_indices[0] < 0 or current_index + 10 >= len(frames):
                invalid_windows.append(f"short-window:{token}")
                continue
            current_translation, current_yaw = _pose(frames[current_index])
            observation_count = np.zeros(len(points_xyz), dtype=np.int16)
            current_visible = np.zeros(len(points_xyz), dtype=bool)

            for history_offset, frame_index in enumerate(history_indices):
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
                for camera_name in CAMERAS:
                    camera = _camera_record(frame, camera_name)
                    image_path = args.sensor_root / str(camera["data_path"])
                    if not image_path.is_file():
                        missing_files.append(f"image:{camera['data_path']}")
                        continue
                    raw_shape = _image_shape(image_path)
                    image_shapes[camera_name][f"{raw_shape[0]}x{raw_shape[1]}"] += 1
                    calibration_fingerprints[camera_name][_calibration_fingerprint(camera)] += 1
                    rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
                    maximum_rotation_orthogonality_error = max(
                        maximum_rotation_orthogonality_error,
                        float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
                    )
                    maximum_rotation_determinant_error = max(
                        maximum_rotation_determinant_error,
                        abs(float(np.linalg.det(rotation)) - 1.0),
                    )
                    distortion = np.asarray(camera["distortion"], dtype=np.float64)
                    maximum_distortion = max(maximum_distortion, float(np.max(np.abs(distortion))))
                    projection = project_points_to_patch_grid(
                        history_xyz,
                        rotation,
                        camera["sensor2lidar_translation"],
                        camera["cam_intrinsic"],
                        distortion,
                        raw_shape=raw_shape,
                        resized_shape=(args.resize_height, args.resize_width),
                        patch_size=args.patch_size,
                        distortion_tolerance=args.distortion_tolerance,
                    )
                    valid = projection.valid_for_bilinear_patch_sampling
                    observation_count += valid.astype(np.int16)
                    if history_offset == 3:
                        current_visible |= valid
                    view_rates[f"h{history_offset}:{camera_name}"].append(float(np.mean(valid)))

            all_visible = observation_count > 0
            current_only_rates.append(float(np.mean(current_visible)))
            all_history_rates.append(float(np.mean(all_visible)))
            observation_histogram.update(int(value) for value in observation_count)
            for low, high in radial_bands:
                mask = (radius >= low) & (radius < high)
                key = f"{low:g}-{high:g}"
                radial_visible[key][0] += int(np.sum(all_visible & mask))
                radial_visible[key][1] += int(np.sum(mask))
            per_anchor.append(
                {
                    "scene_token": token,
                    "recording_group": anchor["recording_group"],
                    "split": anchor["split"],
                    "current_frame_visible_fraction": float(np.mean(current_visible)),
                    "four_frame_visible_fraction": float(np.mean(all_visible)),
                    "missing_node_count": int(np.sum(~all_visible)),
                    "observation_count_mean": float(np.mean(observation_count)),
                }
            )
        print(f"audited log {log_number}/{len(by_log)}: {log_name}", flush=True)

    if missing_files or invalid_windows:
        raise RuntimeError(
            f"projection audit incomplete: missing_files={len(missing_files)} "
            f"invalid_windows={len(invalid_windows)}"
        )
    expected = len(anchors)
    if len(per_anchor) != expected:
        raise AssertionError(f"audited {len(per_anchor)} anchors, expected {expected}")
    result: dict[str, object] = {
        "experiment_id": "R004B-projection-audit",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "metadata-only train/development projection audit; no labels or model outputs",
        "splits": sorted(requested_splits),
        "anchor_count": expected,
        "recording_group_count": len({row["recording_group"] for row in anchors}),
        "log_count": len(by_log),
        "grid": {
            "support_m": [-32.0, 32.0, -32.0, 32.0],
            "resolution_m": 1.0,
            "node_count": len(points_xyz),
            "reference_surface_z_m": 0.0,
        },
        "preprocessing": {
            "cameras": list(CAMERAS),
            "history_frames": 4,
            "resize_height": args.resize_height,
            "resize_width": args.resize_width,
            "resize_coordinate_convention": "half_pixel",
            "patch_size": args.patch_size,
            "patch_grid": [
                args.resize_height // args.patch_size,
                args.resize_width // args.patch_size,
            ],
            "validity": "positive_depth_and_raw_bounds_and_patch_center_convex_hull",
            "occlusion_claim": False,
        },
        "image_shapes": {key: dict(value) for key, value in image_shapes.items()},
        "calibration_fingerprint_counts": {
            key: len(value) for key, value in calibration_fingerprints.items()
        },
        "maximum_abs_distortion": maximum_distortion,
        "maximum_rotation_orthogonality_error": maximum_rotation_orthogonality_error,
        "maximum_rotation_determinant_error": maximum_rotation_determinant_error,
        "current_frame_visible_fraction": _quantiles(current_only_rates),
        "four_frame_visible_fraction": _quantiles(all_history_rates),
        "per_view_frame_visible_fraction": {
            key: _quantiles(value) for key, value in sorted(view_rates.items())
        },
        "observation_count_histogram": {
            str(key): value for key, value in sorted(observation_histogram.items())
        },
        "radial_four_frame_visible_fraction": {
            key: visible / total for key, (visible, total) in radial_visible.items()
        },
        "input_hashes": {
            "time_set_sha256": _sha256(args.time_set),
            "projection_source_sha256": _sha256(
                Path(__file__).parents[1] / "src" / "aorta_drive" / "vision" / "projection.py"
            ),
        },
        "per_anchor": per_anchor,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="train,development")
    parser.add_argument("--resize-height", type=int, default=288)
    parser.add_argument("--resize-width", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--distortion-tolerance", type=float, default=1e-12)
    parser.add_argument("--max-anchors", type=int)
    args = parser.parse_args()
    result = audit(args)
    summary = {key: value for key, value in result.items() if key != "per_anchor"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
