from __future__ import annotations

import numpy as np
import pytest
import torch

from aorta_drive.training.r005e2 import candidate_tensors_from_panel_arrays


def test_candidate_tensors_preserve_native_and_ragged_mask() -> None:
    poses = np.arange(7 * 41 * 3, dtype=np.float64).reshape(7, 41, 3)
    eligible = np.asarray([True, False, True, False, False, True], dtype=np.bool_)
    candidate, mask = candidate_tensors_from_panel_arrays(poses, eligible)
    assert candidate.shape == (7, 41, 3)
    assert mask.tolist() == [True, True, False, True, False, False, True]
    assert torch.equal(candidate[0], torch.from_numpy(poses[0].astype(np.float32)))
    assert torch.equal(candidate[2], torch.zeros_like(candidate[2]))


def test_candidate_tensors_reject_bad_contract() -> None:
    with pytest.raises(ValueError):
        candidate_tensors_from_panel_arrays(
            np.zeros((7, 40, 3), dtype=np.float64), np.ones(6, dtype=np.bool_)
        )
