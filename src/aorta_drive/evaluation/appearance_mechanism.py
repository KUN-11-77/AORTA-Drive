"""Gauge-invariant measurements for paired appearance field diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def oriented_grid_edges(field: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return rightward then downward edges of one finite square field."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("field must be a square rank-2 array")
    if values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("field must be finite with side length at least two")
    horizontal = values[:, 1:] - values[:, :-1]
    vertical = values[1:, :] - values[:-1, :]
    return np.concatenate((horizontal.ravel(), vertical.ravel()))


@dataclass(frozen=True)
class FieldMechanismMetrics:
    task_mae_m: float
    task_rms_m: float
    task_max_m: float
    teacher_cosine: float
    edge_energy_mae_m: float
    edge_energy_rms_m: float
    appearance_mae_m: float
    appearance_rms_m: float
    appearance_max_m: float


def measure_field_mechanism(
    prediction: npt.ArrayLike,
    teacher: npt.ArrayLike,
    clean_prediction: npt.ArrayLike,
) -> FieldMechanismMetrics:
    """Measure task retention and paired change entirely in graph-edge space."""

    predicted_edges = oriented_grid_edges(prediction)
    teacher_edges = oriented_grid_edges(teacher)
    clean_edges = oriented_grid_edges(clean_prediction)
    residual = predicted_edges - teacher_edges
    appearance = predicted_edges - clean_edges
    predicted_norm = float(np.linalg.norm(predicted_edges))
    teacher_norm = float(np.linalg.norm(teacher_edges))
    cosine = (
        float(np.dot(predicted_edges, teacher_edges) / (predicted_norm * teacher_norm))
        if predicted_norm > 0.0 and teacher_norm > 0.0
        else 0.0
    )
    return FieldMechanismMetrics(
        task_mae_m=float(np.mean(np.abs(residual))),
        task_rms_m=float(np.sqrt(np.mean(np.square(residual)))),
        task_max_m=float(np.max(np.abs(residual))),
        teacher_cosine=cosine,
        edge_energy_mae_m=float(np.mean(np.abs(predicted_edges))),
        edge_energy_rms_m=float(np.sqrt(np.mean(np.square(predicted_edges)))),
        appearance_mae_m=float(np.mean(np.abs(appearance))),
        appearance_rms_m=float(np.sqrt(np.mean(np.square(appearance)))),
        appearance_max_m=float(np.max(np.abs(appearance))),
    )


def mechanism_condition_supported(
    *,
    visual_task_mae_m: float,
    zero_task_mae_m: float,
    appearance_mae_m: float,
    clean_edge_energy_mae_m: float,
    teacher_cosine: float,
) -> bool:
    """Apply the frozen non-collapse mechanism gate for one appearance condition."""

    values = (
        visual_task_mae_m,
        zero_task_mae_m,
        appearance_mae_m,
        clean_edge_energy_mae_m,
        teacher_cosine,
    )
    return bool(
        all(np.isfinite(value) for value in values)
        and 0.0 <= visual_task_mae_m < zero_task_mae_m
        and 0.0 <= appearance_mae_m < clean_edge_energy_mae_m
        and clean_edge_energy_mae_m > 0.0
        and teacher_cosine > 0.0
    )

