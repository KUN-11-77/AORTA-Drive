"""Outcome-free layer contracts for the R007 DINO evidence audit."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt


Float32Array = npt.NDArray[np.float32]


def fixed_layer_candidates(depth: int, *, last_n: int = 4) -> tuple[int, ...]:
    """Return zero-based, outcome-free depth quartiles plus the final layers.

    Quartile index ``q`` is ``ceil(depth * q) - 1``.  This rule depends only on
    the audited model depth and cannot use task labels or held-out results.
    """

    if depth < 1:
        raise ValueError("model depth must be positive")
    if last_n < 1 or last_n > depth:
        raise ValueError("last_n must lie in [1, depth]")
    quartiles = {
        math.ceil(depth * numerator / 4) - 1 for numerator in (1, 2, 3)
    }
    final = set(range(depth - last_n, depth))
    return tuple(sorted(quartiles | final))


def validate_layer_maps(
    layers: Mapping[int, npt.ArrayLike],
    *,
    expected_indices: Sequence[int],
) -> tuple[int, ...]:
    """Validate a complete finite layer payload with one shared tensor shape."""

    expected = tuple(expected_indices)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected_indices must be unique and non-empty")
    if set(layers) != set(expected):
        missing = sorted(set(expected) - set(layers))
        extra = sorted(set(layers) - set(expected))
        raise ValueError(f"layer payload disagrees: missing={missing}, extra={extra}")
    arrays = [np.asarray(layers[index]) for index in expected]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"layer shapes disagree: {sorted(shapes)}")
    if not arrays[0].size:
        raise ValueError("layer maps must be non-empty")
    if not all(np.issubdtype(array.dtype, np.floating) for array in arrays):
        raise TypeError("layer maps must use floating-point storage")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("layer maps must be finite")
    return tuple(int(dimension) for dimension in arrays[0].shape)


def equal_weight_layer_mean(
    layers: Mapping[int, npt.ArrayLike],
    *,
    indices: Sequence[int],
) -> Float32Array:
    """Average model-normalized layers without changing token or channel width."""

    selected = tuple(indices)
    validate_layer_maps(layers, expected_indices=selected)
    stack = np.stack(
        [np.asarray(layers[index], dtype=np.float32) for index in selected], axis=0
    )
    result = np.mean(stack, axis=0, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("layer mean is non-finite")
    return np.ascontiguousarray(result, dtype=np.float32)
