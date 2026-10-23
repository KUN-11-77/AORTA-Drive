"""Audit the R004B physical-node to DINO-token projection on source metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import cv2
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


def _image_perimeter(
    shape: tuple[int, int], samples_per_edge: int = 1024
) -> npt.NDArray[np.float64]:
    height, width = shape
    horizontal = np.linspace(0.0, width - 1.0, samples_per_edge)
    vertical = np.linspace(0.0, height - 1.0, samples_per_edge)
    return np.concatenate(
        (
            np.column_stack((horizontal, np.zeros_like(horizontal))),
            np.column_stack((horizontal, np.full_like(horizontal, height - 1.0))),
            np.column_stack((np.zeros_like(vertical), vertical)),
            np.column_stack((np.full_like(vertical, width - 1.0), vertical)),
        )
    )


def _first_radial_turning_radius(distortion: npt.NDArray[np.float64]) -> float:
    """First positive radius where d[r * radial(r)]/dr is zero."""

    k1, k2, _p1, _p2, k3 = distortion
    roots_s = np.roots([7.0 * k3, 5.0 * k2, 3.0 * k1, 1.0])
    positive_s = sorted(
        float(root.real)
        for root in roots_s
        if abs(float(root.imag)) <= 1e-10 and float(root.real) > 0.0
    )
    if not positive_s:
        return float("inf")
    return math.sqrt(positive_s[0])


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
    distortion_vectors: dict[str, set[tuple[float, ...]]] = {
        camera: set() for camera in CAMERAS
    }
    oracle_checks_by_calibration: dict[str, dict[str, float | int]] = {}
    maximum_distortion = 0.0
    maximum_rotation_orthogonality_error = 0.0
    maximum_rotation_determinant_error = 0.0
    view_rates: dict[str, list[float]] = defaultdict(list)
    pinhole_valid_disagreement_rates: list[float] = []
    pinhole_false_valid_rates: list[float] = []
    pinhole_missed_valid_rates: list[float] = []
    distortion_mean_pixel_shifts: list[float] = []
    distortion_max_pixel_shifts: list[float] = []
    distortion_mean_patch_shifts: list[float] = []
    raw_image_support_rates: list[float] = []
    patch_hull_exclusion_rates: list[float] = []
    patch_hull_conditional_exclusion_rates: list[float] = []
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
                    fingerprint = _calibration_fingerprint(camera)
                    calibration_fingerprints[camera_name][fingerprint] += 1
                    rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
                    translation = np.asarray(
                        camera["sensor2lidar_translation"], dtype=np.float64
                    )
                    intrinsics = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
                    maximum_rotation_orthogonality_error = max(
                        maximum_rotation_orthogonality_error,
                        float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
                    )
                    maximum_rotation_determinant_error = max(
                        maximum_rotation_determinant_error,
                        abs(float(np.linalg.det(rotation)) - 1.0),
                    )
                    distortion = np.asarray(camera["distortion"], dtype=np.float64)
                    distortion_vectors[camera_name].add(
                        tuple(float(value) for value in distortion)
                    )
                    maximum_distortion = max(
                        maximum_distortion, float(np.max(np.abs(distortion)))
                    )
                    projection = project_points_to_patch_grid(
                        history_xyz,
                        rotation,
                        translation,
                        intrinsics,
                        distortion,
                        raw_shape=raw_shape,
                        resized_shape=(args.resize_height, args.resize_width),
                        patch_size=args.patch_size,
                        maximum_normalized_radius=args.maximum_normalized_radius,
                    )
                    pinhole_projection = project_points_to_patch_grid(
                        history_xyz,
                        rotation,
                        translation,
                        intrinsics,
                        np.zeros(5, dtype=np.float64),
                        raw_shape=raw_shape,
                        resized_shape=(args.resize_height, args.resize_width),
                        patch_size=args.patch_size,
                        maximum_normalized_radius=args.maximum_normalized_radius,
                    )
                    if fingerprint not in oracle_checks_by_calibration:
                        camera_points = (history_xyz - translation) @ rotation
                        oracle_uv, _ = cv2.projectPoints(
                            camera_points,
                            np.zeros(3, dtype=np.float64),
                            np.zeros(3, dtype=np.float64),
                            intrinsics,
                            distortion,
                        )
                        oracle_uv = oracle_uv.reshape(-1, 2)
                        finite_both = (
                            np.all(np.isfinite(projection.raw_uv), axis=1)
                            & np.all(np.isfinite(oracle_uv), axis=1)
                        )
                        with np.errstate(divide="ignore", invalid="ignore"):
                            normalized = camera_points[:, :2] / camera_points[:, 2:3]
                        bounded = (
                            projection.positive_depth
                            & finite_both
                            & np.all(np.isfinite(normalized), axis=1)
                            & (
                                np.linalg.norm(normalized, axis=1)
                                <= args.maximum_normalized_radius
                            )
                        )
                        oracle_inside = (
                            bounded
                            & (oracle_uv[:, 0] >= 0.0)
                            & (oracle_uv[:, 0] <= raw_shape[1] - 1.0)
                            & (oracle_uv[:, 1] >= 0.0)
                            & (oracle_uv[:, 1] <= raw_shape[0] - 1.0)
                        )
                        mask_mismatches = int(
                            np.sum(projection.inside_raw_image != oracle_inside)
                        )
                        perimeter = _image_perimeter(raw_shape)
                        ideal_perimeter = cv2.undistortPoints(
                            perimeter[:, None, :], intrinsics, distortion
                        ).reshape(-1, 2)
                        maximum_boundary_radius = float(
                            np.max(np.linalg.norm(ideal_perimeter, axis=1))
                        )
                        radial_turning_radius = _first_radial_turning_radius(distortion)
                        if maximum_boundary_radius >= args.maximum_normalized_radius:
                            raise RuntimeError(
                                "calibrated ray bound excludes an observed image boundary: "
                                f"boundary={maximum_boundary_radius:.12g}, "
                                f"bound={args.maximum_normalized_radius:.12g}"
                            )
                        if args.maximum_normalized_radius >= radial_turning_radius:
                            raise RuntimeError(
                                "calibrated ray bound reaches radial polynomial fold-back: "
                                f"bound={args.maximum_normalized_radius:.12g}, "
                                f"turning={radial_turning_radius:.12g}"
                            )
                        if not np.any(bounded):
                            raise RuntimeError(
                                f"no bounded positive-depth OpenCV oracle rays for {fingerprint}"
                            )
                        oracle_checks_by_calibration[fingerprint] = {
                            "bounded_ray_count": int(np.sum(bounded)),
                            "inside_mask_mismatch_count": mask_mismatches,
                            "maximum_ideal_image_boundary_radius": maximum_boundary_radius,
                            "radial_only_first_turning_radius": radial_turning_radius,
                            "maximum_absolute_error_px": float(
                                np.max(
                                    np.abs(
                                        projection.raw_uv[bounded] - oracle_uv[bounded]
                                    )
                                )
                            ),
                        }
                    valid = projection.valid_for_bilinear_patch_sampling
                    raw_support = projection.inside_raw_image
                    excluded_by_patch_hull = raw_support & ~valid
                    raw_image_support_rates.append(float(np.mean(raw_support)))
                    patch_hull_exclusion_rates.append(float(np.mean(excluded_by_patch_hull)))
                    patch_hull_conditional_exclusion_rates.append(
                        float(np.sum(excluded_by_patch_hull) / max(np.sum(raw_support), 1))
                    )
                    pinhole_valid = pinhole_projection.valid_for_bilinear_patch_sampling
                    pinhole_valid_disagreement_rates.append(float(np.mean(valid != pinhole_valid)))
                    pinhole_false_valid_rates.append(float(np.mean(~valid & pinhole_valid)))
                    pinhole_missed_valid_rates.append(float(np.mean(valid & ~pinhole_valid)))
                    if np.any(valid):
                        pixel_shift = np.linalg.norm(
                            projection.raw_uv[valid] - pinhole_projection.raw_uv[valid], axis=1
                        )
                        patch_shift = np.linalg.norm(
                            projection.patch_xy[valid] - pinhole_projection.patch_xy[valid], axis=1
                        )
                        distortion_mean_pixel_shifts.append(float(np.mean(pixel_shift)))
                        distortion_max_pixel_shifts.append(float(np.max(pixel_shift)))
                        distortion_mean_patch_shifts.append(float(np.mean(patch_shift)))
                    view_rates[f"h{history_offset}:{camera_name}"].append(
                        float(np.mean(valid))
                    )
                    observation_count += valid.astype(np.int16)
                    if history_offset == 3:
                        current_visible |= valid

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
                    "current_frame_supported_fraction": float(np.mean(current_visible)),
                    "four_frame_supported_fraction": float(np.mean(all_visible)),
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
    maximum_oracle_error = max(
        float(check["maximum_absolute_error_px"])
        for check in oracle_checks_by_calibration.values()
    )
    oracle_mask_mismatches = sum(
        int(check["inside_mask_mismatch_count"])
        for check in oracle_checks_by_calibration.values()
    )
    if oracle_mask_mismatches:
        raise RuntimeError(
            "projection disagrees with cv2.projectPoints image membership: "
            f"mismatches={oracle_mask_mismatches}"
        )
    if maximum_oracle_error > args.oracle_max_error_px:
        raise RuntimeError(
            "projection disagrees with cv2.projectPoints: "
            f"max_abs_error_px={maximum_oracle_error:.12g} > "
            f"{args.oracle_max_error_px:.12g}"
        )
    result: dict[str, object] = {
        "experiment_id": "R004B-projection-audit",
        "implementation_revision": args.implementation_revision,
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
            "maximum_normalized_radius": args.maximum_normalized_radius,
            "validity": (
                "positive_depth_and_calibrated_ray_domain_and_raw_bounds_"
                "and_patch_center_convex_hull"
            ),
            "distortion_model": "opencv_brown_conrady_5_forward",
            "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
            "occlusion_claim": False,
        },
        "image_shapes": {key: dict(value) for key, value in image_shapes.items()},
        "calibration_fingerprint_counts": {
            key: len(value) for key, value in calibration_fingerprints.items()
        },
        "distortion_vectors": {
            key: [list(vector) for vector in sorted(value)]
            for key, value in distortion_vectors.items()
        },
        "maximum_abs_distortion": maximum_distortion,
        "pinhole_counterfactual": {
            "purpose": "quantify the error from silently dropping NAVSIM distortion",
            "valid_mask_disagreement_fraction_per_view": _quantiles(
                pinhole_valid_disagreement_rates
            ),
            "pinhole_false_valid_fraction_per_view": _quantiles(
                pinhole_false_valid_rates
            ),
            "pinhole_missed_valid_fraction_per_view": _quantiles(
                pinhole_missed_valid_rates
            ),
            "mean_pixel_shift_on_valid_nodes_per_view": _quantiles(
                distortion_mean_pixel_shifts
            ),
            "maximum_pixel_shift_on_valid_nodes_per_view": _quantiles(
                distortion_max_pixel_shifts
            ),
            "mean_patch_shift_on_valid_nodes_per_view": _quantiles(
                distortion_mean_patch_shifts
            ),
        },
        "patch_token_support": {
            "raw_image_support_fraction_per_view": _quantiles(raw_image_support_rates),
            "patch_hull_exclusion_fraction_per_view": _quantiles(
                patch_hull_exclusion_rates
            ),
            "conditional_exclusion_among_raw_supported_per_view": _quantiles(
                patch_hull_conditional_exclusion_rates
            ),
            "policy": "excluded nodes are missing evidence; no padding or clamping",
        },
        "opencv_projection_oracle": {
            "opencv_version": cv2.__version__,
            "unique_calibrations_compared": len(oracle_checks_by_calibration),
            "bounded_normalized_radius": args.maximum_normalized_radius,
            "maximum_ideal_image_boundary_radius": max(
                float(check["maximum_ideal_image_boundary_radius"])
                for check in oracle_checks_by_calibration.values()
            ),
            "minimum_radial_only_first_turning_radius": min(
                float(check["radial_only_first_turning_radius"])
                for check in oracle_checks_by_calibration.values()
            ),
            "inside_mask_mismatch_count": oracle_mask_mismatches,
            "maximum_absolute_error_px": maximum_oracle_error,
            "acceptance_threshold_px": args.oracle_max_error_px,
            "status": "PASS",
        },
        "maximum_rotation_orthogonality_error": maximum_rotation_orthogonality_error,
        "maximum_rotation_determinant_error": maximum_rotation_determinant_error,
        "current_frame_supported_fraction": _quantiles(current_only_rates),
        "four_frame_supported_fraction": _quantiles(all_history_rates),
        "per_view_frame_supported_fraction": {
            key: _quantiles(value) for key, value in sorted(view_rates.items())
        },
        "observation_count_histogram": {
            str(key): value for key, value in sorted(observation_histogram.items())
        },
        "radial_four_frame_supported_fraction": {
            key: visible / total for key, (visible, total) in radial_visible.items()
        },
        "input_hashes": {
            "time_set_sha256": _sha256(args.time_set),
            "audit_script_sha256": _sha256(Path(__file__)),
            "projection_source_sha256": _sha256(
                Path(__file__).parents[1] / "src" / "aorta_drive" / "vision" / "projection.py"
            ),
        },
        "method_sources": {
            "opencv_calib3d": "https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html",
            "nuplan_camera_schema_commit": "e9241677997dd86bfc0bcd44817ab04fe631405b",
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
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--splits", default="train,development")
    parser.add_argument("--resize-height", type=int, default=288)
    parser.add_argument("--resize-width", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--maximum-normalized-radius", type=float, default=1.0)
    parser.add_argument("--oracle-max-error-px", type=float, default=1e-8)
    parser.add_argument("--max-anchors", type=int)
    args = parser.parse_args()
    result = audit(args)
    summary = {key: value for key, value in result.items() if key != "per_anchor"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
