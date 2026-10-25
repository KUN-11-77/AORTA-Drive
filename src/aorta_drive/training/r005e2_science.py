"""Frozen scientific data interface for the R005E2 visual-increment gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from aorta_drive.training.r005e2 import load_visual_tensor


ARM_REMOVED = "V0_removed"
ARM_MISMATCH = "V1_mismatch"
ARM_CORRECT = "V2_correct"
ARMS = (ARM_REMOVED, ARM_MISMATCH, ARM_CORRECT)


@dataclass(frozen=True)
class ScientificAnchor:
    recording_group: str
    anchor_index: int
    identity: str
    correct_feature_paths: tuple[Path, ...]
    mismatch_feature_paths: tuple[Path, ...]


def identity(recording_group: str, anchor_index: int) -> str:
    return f"{recording_group}::{anchor_index:02d}"


def donor_map_for_stage(
    manifest: Mapping[str, object], *, outer_fold: int, stage: str
) -> dict[str, str]:
    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise ValueError("mismatch manifest lacks folds")
    matches = [
        row
        for row in folds
        if isinstance(row, dict) and int(row.get("outer_fold", -1)) == outer_fold
    ]
    if len(matches) != 1:
        raise ValueError("mismatch manifest outer fold is not unique")
    stages = matches[0].get("stages")
    if not isinstance(stages, dict) or stage not in stages:
        raise ValueError("mismatch manifest stage is missing")
    stage_payload = stages[stage]
    if not isinstance(stage_payload, dict) or not isinstance(
        stage_payload.get("rows"), list
    ):
        raise ValueError("mismatch stage rows are malformed")
    result: dict[str, str] = {}
    for row in stage_payload["rows"]:
        if not isinstance(row, dict):
            raise ValueError("mismatch row is malformed")
        recipient = str(row["recipient"])
        donor = str(row["donor"])
        if recipient in result:
            raise ValueError("duplicate mismatch recipient")
        result[recipient] = donor
    return result


class ScientificDataset(torch.utils.data.Dataset[dict[str, Tensor]]):
    """Load frozen tensors while varying only the registered visual arm."""

    def __init__(
        self,
        *,
        anchors: Sequence[ScientificAnchor],
        bundle_root: Path,
        bundle_indices: Sequence[int],
        arm: str,
        removed_slot_mean_path: Path,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown R005E2 arm: {arm}")
        if not anchors or len(anchors) != len(bundle_indices):
            raise ValueError("anchors and bundle indices must be nonempty and aligned")
        self._anchors = tuple(anchors)
        self._indices = tuple(int(value) for value in bundle_indices)
        self._arm = arm
        self._candidate = np.load(bundle_root / "candidate_poses.npy", mmap_mode="r")
        self._mask = np.load(bundle_root / "candidate_mask.npy", mmap_mode="r")
        self._previous = np.load(bundle_root / "previous_poses.npy", mmap_mode="r")
        self._target = np.load(bundle_root / "target_margin.npy", mmap_mode="r")
        removed = np.load(removed_slot_mean_path, allow_pickle=False)
        if removed.dtype != np.float32 or removed.shape != (12, 1, 1, 1280):
            raise ValueError("removed slot mean must have float32 shape [12,1,1,1280]")
        if not np.isfinite(removed).all():
            raise ValueError("removed slot mean is nonfinite")
        self._removed = (
            torch.from_numpy(removed).to(torch.bfloat16).expand(12, 18, 32, 1280)
        )

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        anchor = self._anchors[item]
        index = self._indices[item]
        if self._arm == ARM_REMOVED:
            visual = self._removed
        elif self._arm == ARM_MISMATCH:
            visual = load_visual_tensor(anchor.mismatch_feature_paths)
        else:
            visual = load_visual_tensor(anchor.correct_feature_paths)
        return {
            "bundle_index": torch.tensor(index, dtype=torch.int64),
            "candidate_poses": torch.from_numpy(
                np.asarray(self._candidate[index], dtype=np.float32).copy()
            ),
            "candidate_mask": torch.from_numpy(
                np.asarray(self._mask[index], dtype=np.bool_).copy()
            ),
            "previous_poses": torch.from_numpy(
                np.asarray(self._previous[index], dtype=np.float32).copy()
            ),
            "visual_features": visual,
            "visual_slot_mask": torch.ones(12, dtype=torch.bool),
            "target_margin": torch.from_numpy(
                np.asarray(self._target[index], dtype=np.float32).copy()
            ),
        }


def validate_bundle_arrays(root: Path, expected_count: int = 672) -> None:
    contracts: tuple[tuple[str, tuple[int, ...], npt.DTypeLike], ...] = (
        ("candidate_poses.npy", (expected_count, 7, 41, 3), np.float32),
        ("candidate_mask.npy", (expected_count, 7), np.bool_),
        ("previous_poses.npy", (expected_count, 36, 3), np.float32),
        ("target_margin.npy", (expected_count, 7), np.float32),
    )
    for name, shape, dtype in contracts:
        array = np.load(root / name, mmap_mode="r")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"bundle contract mismatch: {name}")
        if dtype != np.bool_ and not np.isfinite(array).all():
            raise ValueError(f"bundle array is nonfinite: {name}")
    mask = np.load(root / "candidate_mask.npy", mmap_mode="r")
    target = np.load(root / "target_margin.npy", mmap_mode="r")
    if not bool(mask[:, 0].all()) or not np.array_equal(
        target[:, 0], np.zeros(expected_count)
    ):
        raise ValueError("native validity or exact-zero target contract failed")
    if not np.array_equal(target[~mask], np.zeros(int((~mask).sum()))):
        raise ValueError("masked target storage must be exact zero")
