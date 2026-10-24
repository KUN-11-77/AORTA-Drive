"""Interval-exact diagnostics for teacher-critical candidate query nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CriticalSupport:
    possible_minimizer_indices: tuple[int, ...]
    supported_possible_minimizer_indices: tuple[int, ...]
    unsupported_possible_minimizer_indices: tuple[int, ...]

    @property
    def all_possible_minimizers_supported(self) -> bool:
        return not self.unsupported_possible_minimizer_indices


def outward_node_intervals(
    values: npt.ArrayLike, errors: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Enclose value +/- non-negative error with outward binary64 rounding."""

    center = np.asarray(values, dtype=np.float64).reshape(-1)
    radius = np.asarray(errors, dtype=np.float64).reshape(-1)
    if center.shape != radius.shape or not center.size:
        raise ValueError("node values and errors must be aligned non-empty vectors")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(radius)):
        raise ValueError("node values and errors must be finite")
    if np.any(radius < 0):
        raise ValueError("node errors must be non-negative")
    lower = np.nextafter(center - radius, -np.inf)
    upper = np.nextafter(center + radius, np.inf)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("outward interval overflowed")
    return lower, upper


def possible_minimizer_indices(
    node_indices: npt.ArrayLike,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
) -> npt.NDArray[np.int64]:
    """Return nodes that minimize the candidate for some independent interval realization.

    For a finite candidate set F and node intervals [l_q, u_q], q is a possible
    minimizer iff l_q <= min_p u_p.  Necessity follows from d_q >= l_q and
    min_p d_p <= min_p u_p.  Sufficiency is witnessed by d_q=l_q and setting
    every other node to its upper endpoint.
    """

    indices = np.asarray(node_indices, dtype=np.int64).reshape(-1)
    lower_array = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper_array = np.asarray(upper, dtype=np.float64).reshape(-1)
    if not indices.size:
        raise ValueError("candidate node set must be non-empty")
    if lower_array.shape != upper_array.shape or not lower_array.size:
        raise ValueError("lower and upper bounds must be aligned non-empty vectors")
    if np.any(indices < 0) or np.any(indices >= lower_array.size):
        raise ValueError("candidate node index is outside the interval field")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("candidate node indices must be unique")
    if (
        not np.all(np.isfinite(lower_array))
        or not np.all(np.isfinite(upper_array))
        or np.any(lower_array > upper_array)
    ):
        raise ValueError("invalid node intervals")
    cutoff = float(np.min(upper_array[indices]))
    return cast(
        npt.NDArray[np.int64],
        indices[lower_array[indices] <= cutoff].astype(np.int64, copy=False),
    )


def critical_support(
    node_indices: npt.ArrayLike,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    has_direct_evidence: npt.ArrayLike,
) -> CriticalSupport:
    """Partition possible minimizers by the frozen direct-evidence mask."""

    evidence = np.asarray(has_direct_evidence, dtype=np.bool_).reshape(-1)
    lower_array = np.asarray(lower, dtype=np.float64).reshape(-1)
    if evidence.shape != lower_array.shape:
        raise ValueError("evidence mask and interval field must be aligned")
    possible = possible_minimizer_indices(node_indices, lower, upper)
    supported = possible[evidence[possible]]
    unsupported = possible[~evidence[possible]]
    return CriticalSupport(
        possible_minimizer_indices=tuple(int(value) for value in possible),
        supported_possible_minimizer_indices=tuple(int(value) for value in supported),
        unsupported_possible_minimizer_indices=tuple(int(value) for value in unsupported),
    )
