"""Frozen multi-control proxy-calibration helpers for R004H3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.typing as npt

from aorta_drive.certification.calibration import split_conformal_threshold


CONTROL_NAMES = ("learned", "pooled", "uniform", "evidence_only")


@dataclass(frozen=True)
class ControlCalibration:
    """One control's independent finite-sample proxy threshold."""

    session_scores: tuple[float, ...]
    threshold: float
    rank: int


def canonical_grid_edges(grid_size: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return rightward-then-downward endpoints for a row-major square grid."""

    if grid_size <= 1:
        raise ValueError("grid_size must exceed one")
    nodes = np.arange(grid_size * grid_size, dtype=np.int64).reshape(grid_size, grid_size)
    sources = np.concatenate((nodes[:, :-1].reshape(-1), nodes[:-1, :].reshape(-1)))
    targets = np.concatenate((nodes[:, 1:].reshape(-1), nodes[1:, :].reshape(-1)))
    return sources, targets


def calibrate_control_session_scores(
    scores_by_control: Mapping[str, tuple[float, ...]],
    *,
    alpha: float,
    expected_sessions: int,
) -> dict[str, ControlCalibration]:
    """Calibrate every registered control independently without filtering sessions."""

    if set(scores_by_control) != set(CONTROL_NAMES):
        raise ValueError("scores must contain exactly the four registered controls")
    if expected_sessions <= 0:
        raise ValueError("expected_sessions must be positive")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    calibrated: dict[str, ControlCalibration] = {}
    for name in CONTROL_NAMES:
        calibrated[name] = calibrate_one_control_session_scores(
            scores_by_control[name],
            alpha=alpha,
            expected_sessions=expected_sessions,
            control_name=name,
        )
    return calibrated


def calibrate_one_control_session_scores(
    scores: tuple[float, ...],
    *,
    alpha: float,
    expected_sessions: int,
    control_name: str,
) -> ControlCalibration:
    """Calibrate one complete control or reject it without session filtering."""

    if not control_name:
        raise ValueError("control_name must be non-empty")
    if expected_sessions <= 0:
        raise ValueError("expected_sessions must be positive")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (expected_sessions,):
        raise ValueError(f"{control_name} must contain every registered session")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{control_name} contains an invalid session score")
    rank = math.ceil((expected_sessions + 1) * (1.0 - alpha))
    return ControlCalibration(
        session_scores=tuple(float(value) for value in values),
        threshold=split_conformal_threshold(values, alpha),
        rank=rank,
    )
