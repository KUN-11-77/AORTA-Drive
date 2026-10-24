"""Deterministic pointwise appearance operators for paired fit-only diagnostics."""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

APPEARANCE_CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "low_exposure",
    "low_contrast",
    "cool_cast",
)


def apply_controlled_appearance(
    image_srgb: npt.ArrayLike,
    condition: str,
) -> npt.NDArray[np.float32]:
    """Apply one frozen pointwise sRGB stress without moving any pixel.

    The input must use channel-last RGB values in ``[0, 1]``. These operators
    are controlled photometric stresses, not simulations of named weather.
    """

    image = np.asarray(image_srgb, dtype=np.float32)
    if image.ndim < 3 or image.shape[-1] != 3:
        raise ValueError("image_srgb must have channel-last RGB shape")
    if not np.all(np.isfinite(image)):
        raise ValueError("image_srgb must be finite")
    if np.any(image < 0.0) or np.any(image > 1.0):
        raise ValueError("image_srgb values must lie in [0, 1]")
    if condition not in APPEARANCE_CONDITIONS:
        raise ValueError(f"unknown appearance condition: {condition}")

    if condition == "clean":
        output = image.copy()
    elif condition == "low_exposure":
        output = image * np.float32(0.50)
    elif condition == "low_contrast":
        output = np.float32(0.50) + np.float32(0.55) * (
            image - np.float32(0.50)
        )
    else:
        gains = np.asarray((0.85, 0.95, 1.10), dtype=np.float32)
        output = image * gains
    return np.ascontiguousarray(np.clip(output, 0.0, 1.0), dtype=np.float32)

