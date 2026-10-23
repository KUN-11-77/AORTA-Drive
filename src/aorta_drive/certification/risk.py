"""Authorization and relative-margin risk event helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def authorization_mask(certified_bounds: npt.ArrayLike, gamma_task: float) -> npt.NDArray[np.bool_]:
    """Return which challenger pairs satisfy the frozen authorization threshold."""

    bounds = np.asarray(certified_bounds, dtype=np.float64)
    if bounds.ndim != 1 or not np.all(np.isfinite(bounds)):
        raise ValueError("certified_bounds must be a finite vector")
    if not np.isfinite(gamma_task) or gamma_task < 0:
        raise ValueError("gamma_task must be finite and non-negative")
    return bounds >= gamma_task


def relative_margin_failure(
    authorized: npt.ArrayLike,
    true_relative_margins: npt.ArrayLike,
    gamma_task: float,
) -> bool:
    """Evaluate the session event that any authorized pair misses ``gamma_task``."""

    mask = np.asarray(authorized, dtype=np.bool_)
    margins = np.asarray(true_relative_margins, dtype=np.float64)
    if mask.ndim != 1 or margins.shape != mask.shape:
        raise ValueError("authorized and true_relative_margins must be aligned vectors")
    if not np.all(np.isfinite(margins)):
        raise ValueError("true_relative_margins must be finite")
    if not np.isfinite(gamma_task) or gamma_task < 0:
        raise ValueError("gamma_task must be finite and non-negative")
    return bool(np.any(mask & (margins < gamma_task)))
