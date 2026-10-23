"""Geometry-aware visual evidence interfaces."""

from aorta_drive.vision.projection import (
    PatchProjection,
    distort_normalized_points_opencv,
    project_points_to_patch_grid,
    transform_current_to_history,
    yaw_from_wxyz,
)

__all__ = [
    "PatchProjection",
    "distort_normalized_points_opencv",
    "project_points_to_patch_grid",
    "transform_current_to_history",
    "yaw_from_wxyz",
]
