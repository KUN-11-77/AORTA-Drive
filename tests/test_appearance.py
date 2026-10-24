from __future__ import annotations

import numpy as np
import pytest

from aorta_drive.vision.appearance import (
    APPEARANCE_CONDITIONS,
    H6_APPEARANCE_CONDITIONS,
    apply_controlled_appearance,
    apply_h6_controlled_appearance,
)


def test_registered_appearance_operators_match_frozen_formulas() -> None:
    image = np.asarray([[[0.0, 0.5, 1.0], [0.2, 0.4, 0.6]]], dtype=np.float32)
    expected = {
        "clean": image,
        "low_exposure": image * np.float32(0.5),
        "low_contrast": np.float32(0.5)
        + np.float32(0.55) * (image - np.float32(0.5)),
        "cool_cast": np.clip(
            image * np.asarray((0.85, 0.95, 1.10), dtype=np.float32), 0.0, 1.0
        ),
    }
    for condition in APPEARANCE_CONDITIONS:
        actual = apply_controlled_appearance(image, condition)
        np.testing.assert_array_equal(actual, expected[condition].astype(np.float32))
        assert actual.shape == image.shape
        assert actual.dtype == np.float32
        assert actual.flags.c_contiguous


def test_appearance_operators_are_pointwise_and_do_not_alias_input() -> None:
    image = np.linspace(0.0, 1.0, 18, dtype=np.float32).reshape(2, 3, 3)
    for condition in APPEARANCE_CONDITIONS:
        output = apply_controlled_appearance(image, condition)
        assert not np.shares_memory(output, image)
        assert np.all(np.isfinite(output))
        assert np.all((0.0 <= output) & (output <= 1.0))


@pytest.mark.parametrize(
    ("image", "condition"),
    [
        (np.zeros((2, 2), dtype=np.float32), "clean"),
        (np.full((2, 2, 3), np.nan, dtype=np.float32), "clean"),
        (np.full((2, 2, 3), 1.1, dtype=np.float32), "clean"),
        (np.zeros((2, 2, 3), dtype=np.float32), "unregistered"),
    ],
)
def test_appearance_operators_reject_invalid_inputs(
    image: np.ndarray, condition: str
) -> None:
    with pytest.raises(ValueError):
        apply_controlled_appearance(image, condition)


def test_h6_twice_conditions_are_exact_compositions() -> None:
    image = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 4, 3)
    assert H6_APPEARANCE_CONDITIONS[:4] == APPEARANCE_CONDITIONS
    for base in APPEARANCE_CONDITIONS[1:]:
        expected = apply_controlled_appearance(
            apply_controlled_appearance(image, base), base
        )
        actual = apply_h6_controlled_appearance(image, f"{base}_twice")
        np.testing.assert_array_equal(actual, expected)

