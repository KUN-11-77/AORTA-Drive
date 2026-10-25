"""Causal history alignment for the R005E2 contextual-margin probe."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from aorta_drive.certification.geometry import Pose2D
from aorta_drive.evaluation.navsim_execution import _states_to_local_poses


@dataclass(frozen=True)
class PreviousContext:
    """Previous realized plan on the time support shared with the current plan."""

    poses: npt.NDArray[np.float64]
    overlap_start: int
    observed_gap_s: float
    maximum_transform_error_m: float


def array_sha256(value: npt.ArrayLike) -> str:
    """Hash an array with the exact convention used by R005E1B."""

    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def cache_time_s(cache: Any) -> float:
    time_point = cache.ego_state.time_point
    if hasattr(time_point, "time_s"):
        return float(time_point.time_s)
    return float(time_point.time_us) / 1_000_000.0


def align_previous_simulated_context(
    previous_states: npt.ArrayLike,
    *,
    previous_time_s: float,
    current_time_s: float,
    current_pose: Pose2D,
    simulation_interval_s: float,
    x_index: int,
    y_index: int,
    heading_index: int,
) -> PreviousContext:
    """Express the previous rollout's causal overlap in the current ego frame.

    R005E1B compares ``current_states[:-k]`` with ``previous_states[k:]``, where
    ``k`` is the rounded observation gap in simulator steps.  The network must
    therefore receive the same ``previous_states[k:]`` support.  Keeping the
    expired prefix or stretching the remaining samples to a new horizon would
    change the conditioning variable used by the target.
    """

    states = np.asarray(previous_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2 or not np.isfinite(states).all():
        raise ValueError("previous simulated states must be finite [time,state]")
    if not math.isfinite(simulation_interval_s) or simulation_interval_s <= 0.0:
        raise ValueError("simulation interval must be finite and positive")
    observed_gap = current_time_s - previous_time_s
    if not math.isfinite(observed_gap) or not 0.0 < observed_gap < 0.55:
        raise ValueError("observation gap is outside the official context contract")
    overlap_start = round(observed_gap / simulation_interval_s)
    if overlap_start <= 0 or overlap_start >= states.shape[0]:
        raise ValueError("previous/current context has no valid temporal overlap")
    local, maximum_error = _states_to_local_poses(
        states[None, ...],
        current_pose=current_pose,
        x_index=x_index,
        y_index=y_index,
        heading_index=heading_index,
    )
    poses = np.ascontiguousarray(local[0, overlap_start:], dtype=np.float64)
    if poses.shape != (states.shape[0] - overlap_start, 3):
        raise ValueError("aligned previous-context shape is invalid")
    return PreviousContext(
        poses=poses,
        overlap_start=overlap_start,
        observed_gap_s=observed_gap,
        maximum_transform_error_m=maximum_error,
    )
