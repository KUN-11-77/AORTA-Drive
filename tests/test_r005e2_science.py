from __future__ import annotations

import numpy as np
import pytest

from aorta_drive.training.r005e2_science import (
    donor_map_for_stage,
    validate_bundle_arrays,
)


def test_donor_map_for_stage_is_exact() -> None:
    manifest = {
        "folds": [
            {
                "outer_fold": 2,
                "stages": {
                    "selection_fit": {
                        "rows": [
                            {"recipient": "g0::00", "donor": "g1::01"},
                            {"recipient": "g0::01", "donor": "g1::00"},
                        ]
                    }
                },
            }
        ]
    }
    assert donor_map_for_stage(manifest, outer_fold=2, stage="selection_fit") == {
        "g0::00": "g1::01",
        "g0::01": "g1::00",
    }


def test_donor_map_rejects_duplicate_recipient() -> None:
    manifest = {
        "folds": [
            {
                "outer_fold": 0,
                "stages": {
                    "outer_test": {
                        "rows": [
                            {"recipient": "g0::00", "donor": "g1::00"},
                            {"recipient": "g0::00", "donor": "g2::00"},
                        ]
                    }
                },
            }
        ]
    }
    with pytest.raises(ValueError, match="duplicate"):
        donor_map_for_stage(manifest, outer_fold=0, stage="outer_test")


def test_validate_bundle_arrays(tmp_path) -> None:
    np.save(tmp_path / "candidate_poses.npy", np.zeros((2, 7, 41, 3), np.float32))
    mask = np.ones((2, 7), np.bool_)
    np.save(tmp_path / "candidate_mask.npy", mask)
    np.save(tmp_path / "previous_poses.npy", np.zeros((2, 41, 3), np.float32))
    np.save(tmp_path / "target_margin.npy", np.zeros((2, 7), np.float32))
    validate_bundle_arrays(tmp_path, expected_count=2)


def test_validate_bundle_rejects_nonzero_native(tmp_path) -> None:
    np.save(tmp_path / "candidate_poses.npy", np.zeros((1, 7, 41, 3), np.float32))
    np.save(tmp_path / "candidate_mask.npy", np.ones((1, 7), np.bool_))
    np.save(tmp_path / "previous_poses.npy", np.zeros((1, 41, 3), np.float32))
    target = np.zeros((1, 7), np.float32)
    target[0, 0] = 0.1
    np.save(tmp_path / "target_margin.npy", target)
    with pytest.raises(ValueError, match="native"):
        validate_bundle_arrays(tmp_path, expected_count=1)
