from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from aorta_drive.vision.dino_evidence_layers import (
    equal_weight_layer_mean,
    fixed_layer_candidates,
    validate_layer_maps,
)


def test_fixed_candidates_depend_only_on_model_depth() -> None:
    assert fixed_layer_candidates(32) == (7, 15, 23, 28, 29, 30, 31)
    assert fixed_layer_candidates(12) == (2, 5, 8, 9, 10, 11)


def test_fixed_candidates_reject_invalid_contracts() -> None:
    for depth, last_n in ((0, 1), (4, 0), (4, 5)):
        try:
            fixed_layer_candidates(depth, last_n=last_n)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid layer candidate contract was accepted")


def test_equal_mean_preserves_shape_width_and_expected_value() -> None:
    layers = {
        3: np.full((2, 5, 7), 1.0, dtype=np.float32),
        7: np.full((2, 5, 7), 3.0, dtype=np.float32),
    }
    result = equal_weight_layer_mean(layers, indices=(3, 7))
    assert result.shape == (2, 5, 7)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.full((2, 5, 7), 2.0, np.float32))


def test_layer_validation_rejects_missing_shape_dtype_and_nonfinite() -> None:
    valid = {
        1: np.zeros((2, 3), dtype=np.float32),
        2: np.ones((2, 3), dtype=np.float32),
    }
    assert validate_layer_maps(valid, expected_indices=(1, 2)) == (2, 3)

    bad_payloads: tuple[
        tuple[Mapping[int, npt.ArrayLike], tuple[int, ...]], ...
    ] = (
        ({1: valid[1]}, (1, 2)),
        ({1: valid[1], 2: np.ones((3, 2), np.float32)}, (1, 2)),
        ({1: valid[1], 2: np.ones((2, 3), np.int64)}, (1, 2)),
        ({1: valid[1], 2: np.full((2, 3), np.nan, np.float32)}, (1, 2)),
    )
    for payload, indices in bad_payloads:
        try:
            validate_layer_maps(payload, expected_indices=indices)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid layer payload was accepted")
