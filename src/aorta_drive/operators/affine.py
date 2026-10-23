"""Reference implementation of the affine task-operator factorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class AffineTaskOperator:
    """Normalized representation of a feasibility-margin panel.

    The represented margin vector is ``scale * relative_shape + offset``.
    A degenerate panel has near-zero weighted variance and therefore does not
    identify a meaningful relative shape.
    """

    relative_shape: FloatArray
    scale: float
    offset: float
    weights: FloatArray
    degenerate: bool

    def compose(self) -> FloatArray:
        """Reconstruct the feasibility-margin vector."""

        return self.scale * self.relative_shape + self.offset


def _normalized_weights(size: int, weights: npt.ArrayLike | None) -> FloatArray:
    if weights is None:
        return np.full(size, 1.0 / size, dtype=np.float64)

    value = np.asarray(weights, dtype=np.float64)
    if value.shape != (size,):
        raise ValueError(f"weights must have shape ({size},), got {value.shape}")
    if not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError("weights must be finite and strictly positive")
    return cast(FloatArray, value / value.sum())


def factorize_margin_panel(
    margins: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    eps: float = 1e-8,
) -> AffineTaskOperator:
    """Factorize margins into weighted relative shape, scale, and offset.

    Args:
        margins: One-dimensional candidate feasibility margins.
        weights: Optional positive probe weights. They are normalized to sum to one.
        eps: Panels with weighted standard deviation at or below this value are
            marked degenerate.

    Returns:
        An :class:`AffineTaskOperator` satisfying the weighted centering and
        unit-scale convention whenever the panel is non-degenerate.

    Raises:
        ValueError: If inputs are empty, non-finite, incorrectly shaped, or if
            ``eps`` is not positive.
    """

    value = np.asarray(margins, dtype=np.float64)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("margins must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(value)):
        raise ValueError("margins must be finite")
    if eps <= 0:
        raise ValueError("eps must be positive")

    normalized_weights = _normalized_weights(value.size, weights)
    offset = float(np.dot(normalized_weights, value))
    centered = value - offset
    scale = float(np.sqrt(np.dot(normalized_weights, centered**2)))
    degenerate = scale <= eps
    relative_shape = np.zeros_like(value) if degenerate else centered / scale

    return AffineTaskOperator(
        relative_shape=relative_shape,
        scale=scale,
        offset=offset,
        weights=normalized_weights,
        degenerate=degenerate,
    )
