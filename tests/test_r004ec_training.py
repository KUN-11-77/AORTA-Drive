from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aorta_drive.training.r004ec import (
    JoinedAnchor,
    deterministic_interleaved_epoch,
    load_joined_anchors,
    sample_dense_features_torch,
)
from aorta_drive.vision.feature_cache import bilinear_sample_dense_numpy


def _record(group: str, anchor: int, split: str = "train") -> JoinedAnchor:
    placeholder = Path("placeholder")
    return JoinedAnchor(
        recording_group=group,
        anchor_index=anchor,
        split=split,
        log_name=f"log-{group}",
        scene_token=f"token-{group}-{anchor}",
        dense_feature_paths=(placeholder,) * 12,
        projection_xy_path=placeholder,
        projection_mask_path=placeholder,
        label_path=placeholder,
        label_artifact_sha256="0" * 64,
    )


def test_vectorized_torch_sampler_matches_numpy_oracle() -> None:
    rng = np.random.default_rng(11)
    dense = rng.normal(size=(3, 5, 4, 6)).astype(np.float32)
    xy = np.zeros((17, 3, 2), dtype=np.float64)
    xy[..., 0] = rng.uniform(0.0, 5.0, size=(17, 3))
    xy[..., 1] = rng.uniform(0.0, 3.0, size=(17, 3))
    mask = rng.random((17, 3)) > 0.3
    expected = bilinear_sample_dense_numpy(dense, xy, mask)
    actual = sample_dense_features_torch(
        torch.from_numpy(dense), torch.from_numpy(xy), torch.from_numpy(mask)
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-6)
    np.testing.assert_array_equal(actual[~mask], 0.0)


def test_interleaved_epoch_is_seeded_exact_and_never_adjacent_within_group() -> None:
    records = tuple(_record(group, anchor) for group in ("a", "b", "c") for anchor in range(16))
    first = deterministic_interleaved_epoch(records, seed=23, epoch=0)
    replay = deterministic_interleaved_epoch(records, seed=23, epoch=0)
    second = deterministic_interleaved_epoch(records, seed=23, epoch=1)
    assert first == replay
    assert set(first) == set(records)
    assert first != second
    assert all(
        left.recording_group != right.recording_group
        for left, right in zip(first, first[1:])
    )


def test_interleaved_epoch_rejects_development_and_incomplete_groups() -> None:
    with pytest.raises(ValueError, match="only train"):
        development = tuple(_record("a", index, "development") for index in range(16))
        deterministic_interleaved_epoch(development, seed=1, epoch=0)
    with pytest.raises(ValueError, match="exactly 16"):
        incomplete = tuple(_record("a", index) for index in range(15))
        deterministic_interleaved_epoch(incomplete, seed=1, epoch=0)


def test_manifest_join_rejects_prohibited_visual_split(tmp_path: Path) -> None:
    visual = {
        "status": "PASS",
        "records": [{"split": "calibration", "log_name": "l", "scene_token": "t"}],
    }
    labels = {
        "status": "PASS",
        "records": [{"cache_id": "l/t", "artifact_name": "a.npz", "artifact_sha256": "0" * 64}],
    }
    feature_manifest = tmp_path / "features.json"
    label_manifest = tmp_path / "labels.json"
    feature_manifest.write_text(json.dumps(visual), encoding="utf-8")
    label_manifest.write_text(json.dumps(labels), encoding="utf-8")
    with pytest.raises(ValueError, match="prohibited split"):
        load_joined_anchors(feature_manifest, label_manifest)
