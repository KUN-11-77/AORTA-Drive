"""Heterogeneous edge-envelope primitives with outward-safe binary64 arithmetic."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class GroupRoles:
    """Deterministic fit, selection, and historically exposed proxy-cal roles."""

    fit: tuple[str, ...]
    selection: tuple[str, ...]
    proxy_cal: tuple[str, ...]


def deterministic_group_roles(
    group_ids: Sequence[str],
    *,
    namespace: str,
    fit_count: int,
    selection_count: int,
    proxy_cal_count: int,
) -> GroupRoles:
    """Assign unique group IDs by a registered SHA-256 ordering."""

    groups = tuple(group_ids)
    if not namespace:
        raise ValueError("namespace must be non-empty")
    if any(not group for group in groups):
        raise ValueError("group IDs must be non-empty")
    if len(set(groups)) != len(groups):
        raise ValueError("group IDs must be unique")
    counts = (fit_count, selection_count, proxy_cal_count)
    if any(count < 0 for count in counts) or sum(counts) != len(groups):
        raise ValueError("role counts must be non-negative and exhaust all groups")

    def key(group: str) -> tuple[bytes, str]:
        payload = f"{namespace}\0{group}".encode()
        return hashlib.sha256(payload).digest(), group

    ordered = tuple(sorted(groups, key=key))
    fit_end = fit_count
    selection_end = fit_end + selection_count
    return GroupRoles(
        fit=ordered[:fit_end],
        selection=ordered[fit_end:selection_end],
        proxy_cal=ordered[selection_end:],
    )


def _aligned_node_fields(
    potentials: npt.ArrayLike,
    labels: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    u = np.asarray(potentials, dtype=np.float64)
    d = np.asarray(labels, dtype=np.float64)
    if u.ndim != 2 or d.shape != u.shape or u.shape[1] == 0:
        raise ValueError("node fields must be aligned non-empty time-by-node matrices")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(d)):
        raise ValueError("node fields must be finite")
    return u, d


def _validated_edges(
    sources: npt.ArrayLike,
    targets: npt.ArrayLike,
    node_count: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    source = np.asarray(sources, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    if source.ndim != 1 or target.shape != source.shape or source.size == 0:
        raise ValueError("edge endpoints must be aligned non-empty vectors")
    if np.any(source < 0) or np.any(target < 0):
        raise ValueError("edge endpoints must be non-negative")
    if np.any(source >= node_count) or np.any(target >= node_count):
        raise ValueError("edge endpoint is out of range")
    if np.any(source == target):
        raise ValueError("self edges are not supported")
    return source, target


def _subtract_interval(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    with np.errstate(over="ignore", invalid="ignore"):
        nearest = left - right
    if not np.all(np.isfinite(nearest)):
        raise ValueError("subtraction overflowed")
    lower = np.nextafter(nearest, -np.inf)
    upper = np.nextafter(nearest, np.inf)
    exact_zero = left == right
    lower = np.where(exact_zero, 0.0, lower)
    upper = np.where(exact_zero, 0.0, upper)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("outward subtraction is non-finite")
    return lower, upper


def _subtract_up(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    with np.errstate(over="ignore", invalid="ignore"):
        nearest = left - right
    if not np.all(np.isfinite(nearest)):
        raise ValueError("subtraction overflowed")
    upper = np.nextafter(nearest, np.inf)
    upper = np.where(left == right, 0.0, upper)
    if not np.all(np.isfinite(upper)):
        raise ValueError("outward subtraction is non-finite")
    return upper


def outward_abs_edge_residuals(
    potentials: npt.ArrayLike,
    labels: npt.ArrayLike,
    sources: npt.ArrayLike,
    targets: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Upper-bound four-term absolute residuals from represented node values.

    Ordinary rounded edge gradients are deliberately not accepted as inputs because
    cancellation can erase residuals before an outward comparison is attempted.
    """

    u, d = _aligned_node_fields(potentials, labels)
    source, target = _validated_edges(sources, targets, u.shape[1])
    gu_lower, gu_upper = _subtract_interval(u[:, target], u[:, source])
    gd_lower, gd_upper = _subtract_interval(d[:, target], d[:, source])
    positive = _subtract_up(gu_upper, gd_lower)
    negative = _subtract_up(gd_upper, gu_lower)
    residuals = np.maximum(np.maximum(positive, negative), 0.0)
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residual enclosure is non-finite")
    return residuals.astype(np.float64, copy=False)


def outward_standardized_scores(
    residual_upper: npt.ArrayLike,
    scales: npt.ArrayLike,
    *,
    scale_min: float,
    scale_max: float,
) -> npt.NDArray[np.float64]:
    """Divide non-negative residual upper bounds by valid scales toward +infinity."""

    residuals = np.asarray(residual_upper, dtype=np.float64)
    scale_values = np.asarray(scales, dtype=np.float64)
    if residuals.ndim != 2 or scale_values.shape != residuals.shape:
        raise ValueError("residuals and scales must be aligned time-by-edge matrices")
    if not math.isfinite(scale_min) or not math.isfinite(scale_max):
        raise ValueError("scale bounds must be finite")
    if not 0.0 < scale_min < scale_max:
        raise ValueError("scale bounds must satisfy 0 < min < max")
    if not np.all(np.isfinite(residuals)) or np.any(residuals < 0.0):
        raise ValueError("residual upper bounds must be finite and non-negative")
    if not np.all(np.isfinite(scale_values)):
        raise ValueError("scales must be finite")
    if np.any(scale_values < scale_min) or np.any(scale_values > scale_max):
        raise ValueError("scales are outside the registered range")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        nearest = residuals / scale_values
    if not np.all(np.isfinite(nearest)):
        raise ValueError("standardized score overflowed")
    scores = np.nextafter(nearest, np.inf)
    scores = np.where(residuals == 0.0, 0.0, scores)
    if not np.all(np.isfinite(scores)):
        raise ValueError("outward standardized score is non-finite")
    return scores.astype(np.float64, copy=False)


def outward_session_max_score(
    potentials: npt.ArrayLike,
    labels: npt.ArrayLike,
    sources: npt.ArrayLike,
    targets: npt.ArrayLike,
    scales: npt.ArrayLike,
    *,
    scale_min: float,
    scale_max: float,
) -> float:
    """Return the outward-safe maximum standardized score for one session."""

    residuals = outward_abs_edge_residuals(potentials, labels, sources, targets)
    scores = outward_standardized_scores(
        residuals,
        scales,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    return float(np.max(scores))


def outward_edge_budgets(
    threshold: float,
    scales: npt.ArrayLike,
    *,
    scale_min: float,
    scale_max: float,
) -> npt.NDArray[np.float64]:
    """Multiply a finite threshold and represented positive scales toward +infinity."""

    scale_values = np.asarray(scales, dtype=np.float64)
    if scale_values.size == 0:
        raise ValueError("scales must be non-empty")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    if not math.isfinite(scale_min) or not math.isfinite(scale_max):
        raise ValueError("scale bounds must be finite")
    if not 0.0 < scale_min < scale_max:
        raise ValueError("scale bounds must satisfy 0 < min < max")
    if not np.all(np.isfinite(scale_values)):
        raise ValueError("scales must be finite")
    if np.any(scale_values < scale_min) or np.any(scale_values > scale_max):
        raise ValueError("scales are outside the registered range")
    if threshold == 0.0:
        return np.zeros_like(scale_values, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        nearest = threshold * scale_values
    if not np.all(np.isfinite(nearest)):
        raise ValueError("edge budget multiplication overflowed")
    budgets = np.nextafter(nearest, np.inf)
    if not np.all(np.isfinite(budgets)):
        raise ValueError("outward edge budget is non-finite")
    return budgets.astype(np.float64, copy=False)
