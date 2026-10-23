"""Deterministic projection from ego-ground nodes to camera patch tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PatchProjection:
    """Projected coordinates and conservative evidence-validity masks."""

    raw_uv: npt.NDArray[np.float64]
    resized_uv: npt.NDArray[np.float64]
    patch_xy: npt.NDArray[np.float64]
    positive_depth: npt.NDArray[np.bool_]
    inside_calibrated_ray_domain: npt.NDArray[np.bool_]
    inside_raw_image: npt.NDArray[np.bool_]
    valid_for_bilinear_patch_sampling: npt.NDArray[np.bool_]


def yaw_from_wxyz(quaternion: npt.ArrayLike) -> float:
    """Return planar yaw from a finite ``[w, x, y, z]`` quaternion."""

    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("quaternion must be a finite [w, x, y, z] vector")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = values / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def transform_current_to_history(
    points_current_xy: npt.ArrayLike,
    current_translation_xy: npt.ArrayLike,
    current_yaw: float,
    history_translation_xy: npt.ArrayLike,
    history_yaw: float,
) -> npt.NDArray[np.float64]:
    """Express current-ego planar points in a historical ego frame."""

    points = np.asarray(points_current_xy, dtype=np.float64)
    current_translation = np.asarray(current_translation_xy, dtype=np.float64)
    history_translation = np.asarray(history_translation_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points_current_xy must be a finite N x 2 matrix")
    if current_translation.shape != (2,) or history_translation.shape != (2,):
        raise ValueError("ego translations must be length-two vectors")
    if not np.all(np.isfinite(current_translation)) or not np.all(np.isfinite(history_translation)):
        raise ValueError("ego translations must be finite")
    if not math.isfinite(current_yaw) or not math.isfinite(history_yaw):
        raise ValueError("ego yaws must be finite")

    current_cos = math.cos(current_yaw)
    current_sin = math.sin(current_yaw)
    global_points = np.empty_like(points)
    global_points[:, 0] = (
        current_cos * points[:, 0] - current_sin * points[:, 1] + current_translation[0]
    )
    global_points[:, 1] = (
        current_sin * points[:, 0] + current_cos * points[:, 1] + current_translation[1]
    )
    shifted = global_points - history_translation
    history_cos = math.cos(history_yaw)
    history_sin = math.sin(history_yaw)
    result = np.empty_like(shifted)
    result[:, 0] = history_cos * shifted[:, 0] + history_sin * shifted[:, 1]
    result[:, 1] = -history_sin * shifted[:, 0] + history_cos * shifted[:, 1]
    return result


def _validate_shape(name: str, shape: tuple[int, int], patch_size: int | None = None) -> None:
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    if patch_size is not None and (height % patch_size != 0 or width % patch_size != 0):
        raise ValueError(f"{name} dimensions must be divisible by patch_size")


def distort_normalized_points_opencv(
    normalized_xy: npt.ArrayLike,
    distortion: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Apply OpenCV's five-parameter Brown--Conrady distortion model.

    ``normalized_xy`` contains ideal pinhole coordinates ``(X/Z, Y/Z)`` and
    ``distortion`` must have the OpenCV order ``[k1, k2, p1, p2, k3]``.  This
    is the forward mapping used by ``cv2.projectPoints``; it is not an
    undistortion iteration.
    """

    normalized = np.asarray(normalized_xy, dtype=np.float64)
    coefficients = np.asarray(distortion, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != 2:
        raise ValueError("normalized_xy must be an N x 2 matrix")
    if coefficients.shape != (5,):
        raise ValueError("distortion must contain exactly [k1, k2, p1, p2, k3]")
    if not np.all(np.isfinite(normalized)) or not np.all(np.isfinite(coefficients)):
        raise ValueError("normalized coordinates and distortion must be finite")

    k1, k2, p1, p2, k3 = coefficients
    x = normalized[:, 0]
    y = normalized[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xy = x * y
    distorted = np.empty_like(normalized)
    distorted[:, 0] = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x)
    distorted[:, 1] = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy
    return distorted


def project_points_to_patch_grid(
    points_xyz: npt.ArrayLike,
    sensor2lidar_rotation: npt.ArrayLike,
    sensor2lidar_translation: npt.ArrayLike,
    intrinsics: npt.ArrayLike,
    distortion: npt.ArrayLike,
    *,
    raw_shape: tuple[int, int],
    resized_shape: tuple[int, int],
    patch_size: int,
    minimum_depth: float = 1e-3,
    maximum_normalized_radius: float = 1.0,
) -> PatchProjection:
    """Project lidar-frame points to a rectangular DINO patch-token grid.

    The resize transform uses the half-pixel convention
    ``u' = (u + 0.5) * W'/W - 0.5``.  A projection is evidence-valid only
    inside the convex hull of patch centers, so bilinear sampling never pads,
    clamps, or extrapolates beyond actual patch tokens.  Raw NAVSIM images are
    sampled in their observed image coordinates, so the ideal camera rays are
    mapped with OpenCV's five-parameter Brown--Conrady forward distortion.  A
    bounded ideal-ray domain prevents nonphysical polynomial fold-back outside
    the calibrated field of view.
    """

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if not math.isfinite(minimum_depth) or minimum_depth <= 0.0:
        raise ValueError("minimum_depth must be finite and positive")
    if not math.isfinite(maximum_normalized_radius) or maximum_normalized_radius <= 0.0:
        raise ValueError("maximum_normalized_radius must be finite and positive")
    _validate_shape("raw_shape", raw_shape)
    _validate_shape("resized_shape", resized_shape, patch_size)

    points = np.asarray(points_xyz, dtype=np.float64)
    rotation = np.asarray(sensor2lidar_rotation, dtype=np.float64)
    translation = np.asarray(sensor2lidar_translation, dtype=np.float64)
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    coefficients = np.asarray(distortion, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_xyz must be a finite N x 3 matrix")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("sensor2lidar_rotation must be a finite 3 x 3 matrix")
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("sensor2lidar_translation must be a finite length-three vector")
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("intrinsics must be a finite 3 x 3 matrix")
    if not np.allclose(
        intrinsic,
        np.array(
            [
                [intrinsic[0, 0], 0.0, intrinsic[0, 2]],
                [0.0, intrinsic[1, 1], intrinsic[1, 2]],
                [0.0, 0.0, 1.0],
            ]
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("intrinsics must follow the standard OpenCV pinhole form")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("intrinsic focal lengths must be positive")

    camera_points = (points - translation) @ rotation
    depth = camera_points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = camera_points[:, :2] / camera_points[:, 2:3]
    finite_normalized = np.all(np.isfinite(normalized), axis=1)
    inside_ray_domain = finite_normalized & (
        np.linalg.norm(np.where(finite_normalized[:, None], normalized, 0.0), axis=1)
        <= maximum_normalized_radius
    )
    safe_normalized = np.where(finite_normalized[:, None], normalized, 0.0)
    distorted = distort_normalized_points_opencv(safe_normalized, coefficients)
    raw_uv = np.empty_like(distorted)
    raw_uv[:, 0] = intrinsic[0, 0] * distorted[:, 0] + intrinsic[0, 2]
    raw_uv[:, 1] = intrinsic[1, 1] * distorted[:, 1] + intrinsic[1, 2]
    raw_uv[~finite_normalized] = np.nan
    finite_uv = np.all(np.isfinite(raw_uv), axis=1)
    positive_depth = np.isfinite(depth) & (depth > minimum_depth)
    raw_height, raw_width = raw_shape
    inside_raw = (
        positive_depth
        & inside_ray_domain
        & finite_uv
        & (raw_uv[:, 0] >= 0.0)
        & (raw_uv[:, 0] <= raw_width - 1.0)
        & (raw_uv[:, 1] >= 0.0)
        & (raw_uv[:, 1] <= raw_height - 1.0)
    )

    resized_height, resized_width = resized_shape
    resized_uv = np.empty_like(raw_uv)
    resized_uv[:, 0] = (raw_uv[:, 0] + 0.5) * resized_width / raw_width - 0.5
    resized_uv[:, 1] = (raw_uv[:, 1] + 0.5) * resized_height / raw_height - 0.5
    patch_xy = (resized_uv + 0.5) / patch_size - 0.5
    patch_height = resized_height // patch_size
    patch_width = resized_width // patch_size
    valid_patch = (
        inside_raw
        & np.all(np.isfinite(patch_xy), axis=1)
        & (patch_xy[:, 0] >= 0.0)
        & (patch_xy[:, 0] <= patch_width - 1.0)
        & (patch_xy[:, 1] >= 0.0)
        & (patch_xy[:, 1] <= patch_height - 1.0)
    )
    return PatchProjection(
        raw_uv=raw_uv,
        resized_uv=resized_uv,
        patch_xy=patch_xy,
        positive_depth=positive_depth,
        inside_calibrated_ray_domain=inside_ray_domain,
        inside_raw_image=inside_raw,
        valid_for_bilinear_patch_sampling=valid_patch,
    )
