"""Source-session simultaneous calibration helpers."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def minimum_calibration_sessions(alpha: float) -> int:
    """Return the minimum sample count for a finite split-conformal quantile."""

    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return math.ceil(1.0 / alpha) - 1


def split_conformal_threshold(scores: npt.ArrayLike, alpha: float) -> float:
    """Compute the finite-sample split-conformal order-statistic threshold."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite vector")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = math.ceil((values.size + 1) * (1.0 - alpha))
    if rank > values.size:
        raise ValueError("not enough calibration sessions for a finite threshold")
    return float(np.partition(values, rank - 1)[rank - 1])


def session_max_score(
    predicted_gradients: npt.ArrayLike,
    label_gradients: npt.ArrayLike,
    edge_scales: npt.ArrayLike,
) -> float:
    """Return the maximum standardized residual over all times and edges."""

    predicted = np.asarray(predicted_gradients, dtype=np.float64)
    labels = np.asarray(label_gradients, dtype=np.float64)
    scales = np.asarray(edge_scales, dtype=np.float64)
    if predicted.ndim != 2 or labels.shape != predicted.shape:
        raise ValueError("gradient arrays must be aligned time-by-edge matrices")
    if scales.shape != (predicted.shape[1],):
        raise ValueError("edge_scales must contain one value per edge")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(labels)):
        raise ValueError("gradients must be finite")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("edge scales must be finite and strictly positive")
    return float(np.max(np.abs(predicted - labels) / scales[None, :]))
