from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aorta_drive.training.r004ec import (
    JoinedAnchor,
    deterministic_interleaved_epoch,
    learning_rate_at_update,
    load_joined_anchors,
    qualifies_single_group_overfit,
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
        "status": "completed",
        "cache_count": 1,
        "cache_pass_count": 1,
        "cache_failure_count": 0,
        "records": [
            {
                "cache_id": "l/t",
                "artifact_name": "a.npz",
                "artifact_sha256": "0" * 64,
            }
        ],
    }
    feature_manifest = tmp_path / "features.json"
    label_manifest = tmp_path / "labels.json"
    feature_manifest.write_text(json.dumps(visual), encoding="utf-8")
    label_manifest.write_text(json.dumps(labels), encoding="utf-8")
    with pytest.raises(ValueError, match="prohibited split"):
        load_joined_anchors(feature_manifest, label_manifest)


def test_manifest_join_accepts_completed_zero_failure_teacher_manifest(
    tmp_path: Path,
) -> None:
    visual_records = []
    label_records = []
    for anchor in range(16):
        log_name = "log-a"
        scene_token = f"token-{anchor}"
        visual_records.append(
            {
                "recording_group": "group-a",
                "anchor_index": anchor,
                "split": "train",
                "log_name": log_name,
                "scene_token": scene_token,
                "projection_xy_relative_path": f"p-{anchor}.xy.npy",
                "projection_mask_relative_path": f"p-{anchor}.mask.npy",
                "slots": [
                    {
                        "slot_index": slot,
                        "feature_relative_path": f"f-{anchor}-{slot}.npy",
                    }
                    for slot in range(12)
                ],
            }
        )
        label_records.append(
            {
                "cache_id": f"{log_name}/{scene_token}",
                "artifact_name": f"label-{anchor}.npz",
                "artifact_sha256": "0" * 64,
            }
        )
    feature_manifest = tmp_path / "features.json"
    label_manifest = tmp_path / "labels.json"
    feature_manifest.write_text(
        json.dumps({"status": "PASS", "records": visual_records}), encoding="utf-8"
    )
    label_manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "cache_count": 16,
                "cache_pass_count": 16,
                "cache_failure_count": 0,
                "records": label_records,
            }
        ),
        encoding="utf-8",
    )
    joined = load_joined_anchors(feature_manifest, label_manifest)
    assert len(joined) == 16
    assert all(record.recording_group == "group-a" for record in joined)
    assert joined[0].label_path == (tmp_path / "labels" / "label-0.npz").resolve()


def test_single_group_overfit_gate_uses_both_preregistered_ratios() -> None:
    assert qualifies_single_group_overfit(
        initial_mean_objective=1.0,
        final_mean_objective=0.9,
        initial_group_max_residual=2.0,
        evaluated_group_max_residuals=(2.1, 1.96),
    )
    assert not qualifies_single_group_overfit(
        initial_mean_objective=1.0,
        final_mean_objective=0.89,
        initial_group_max_residual=2.0,
        evaluated_group_max_residuals=(1.99, 1.97),
    )
    assert not qualifies_single_group_overfit(
        initial_mean_objective=0.0,
        final_mean_objective=0.0,
        initial_group_max_residual=2.0,
        evaluated_group_max_residuals=(1.0,),
    )


def test_learning_rate_schedule_has_frozen_warmup_and_cosine_endpoints() -> None:
    assert learning_rate_at_update(
        0, total_updates=100, warmup_updates=10, base_learning_rate=3e-4
    ) == pytest.approx(3e-5)
    assert learning_rate_at_update(
        9, total_updates=100, warmup_updates=10, base_learning_rate=3e-4
    ) == pytest.approx(3e-4)
    assert learning_rate_at_update(
        10, total_updates=100, warmup_updates=10, base_learning_rate=3e-4
    ) == pytest.approx(3e-4)
    assert learning_rate_at_update(
        100, total_updates=100, warmup_updates=10, base_learning_rate=3e-4
    ) == pytest.approx(0.0)
