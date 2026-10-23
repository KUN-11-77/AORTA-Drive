"""Geometry-aware visual evidence interfaces."""

from aorta_drive.vision.feature_cache import (
    bfloat16_bits_to_float32,
    bilinear_sample_dense_numpy,
    dino_python_tree_sha256,
    load_dense_bfloat16_bits,
    sha256_file,
    validate_projection_arrays,
    write_npy_exact,
)
from aorta_drive.vision.projection import (
    PatchProjection,
    distort_normalized_points_opencv,
    project_points_to_patch_grid,
    transform_current_to_history,
    yaw_from_wxyz,
)

__all__ = [
    "PatchProjection",
    "bfloat16_bits_to_float32",
    "bilinear_sample_dense_numpy",
    "dino_python_tree_sha256",
    "distort_normalized_points_opencv",
    "load_dense_bfloat16_bits",
    "project_points_to_patch_grid",
    "sha256_file",
    "transform_current_to_history",
    "validate_projection_arrays",
    "write_npy_exact",
    "yaw_from_wxyz",
]
