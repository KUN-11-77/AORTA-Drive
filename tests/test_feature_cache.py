from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aorta_drive.vision.feature_cache import (
    bfloat16_bits_to_float32,
    bilinear_sample_dense_numpy,
    load_dense_bfloat16_bits,
    validate_projection_arrays,
    write_npy_exact,
)


def test_bfloat16_bits_decode_exact_representable_values() -> None:
    bits = np.array([0x3F80, 0xC000, 0x0000], dtype=np.uint16)
    decoded = bfloat16_bits_to_float32(bits)
    np.testing.assert_array_equal(decoded, np.array([1.0, -2.0, 0.0], dtype=np.float32))


def test_exact_npy_replay_and_dense_validation(tmp_path: Path) -> None:
    bits = np.full((2, 2, 3), 0x3F80, dtype=np.uint16)
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    assert write_npy_exact(first, bits) == write_npy_exact(second, bits)
    assert first.read_bytes() == second.read_bytes()
    loaded = load_dense_bfloat16_bits(first, expected_shape=(2, 2, 3))
    np.testing.assert_array_equal(loaded, bits)


def test_projection_contract_requires_zero_invalid_storage() -> None:
    patch_xy = np.zeros((2, 2, 2), dtype=np.float64)
    mask = np.array([[True, False], [True, True]], dtype=np.bool_)
    patch_xy[0, 0] = (1.5, 0.5)
    patch_xy[1, 0] = (2.0, 1.0)
    patch_xy[1, 1] = (0.0, 1.0)
    validate_projection_arrays(
        patch_xy, mask, node_count=2, slot_count=2, patch_shape=(2, 3)
    )
    patch_xy[0, 1, 0] = 0.25
    with pytest.raises(ValueError, match="exact zero"):
        validate_projection_arrays(
            patch_xy, mask, node_count=2, slot_count=2, patch_shape=(2, 3)
        )


def test_bilinear_oracle_and_masked_zero() -> None:
    dense = np.array(
        [
            [[[0.0, 2.0], [4.0, 6.0]]],
            [[[10.0, 14.0], [18.0, 22.0]]],
        ],
        dtype=np.float32,
    )
    patch_xy = np.array(
        [
            [[0.5, 0.5], [0.25, 0.75]],
            [[1.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    mask = np.array([[True, True], [True, False]], dtype=np.bool_)
    sampled = bilinear_sample_dense_numpy(dense, patch_xy, mask)
    np.testing.assert_allclose(sampled[0, 0], [3.0])
    np.testing.assert_allclose(sampled[0, 1], [17.0])
    np.testing.assert_allclose(sampled[1, 0], [6.0])
    np.testing.assert_array_equal(sampled[1, 1], [0.0])
