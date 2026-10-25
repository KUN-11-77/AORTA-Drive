"""Outcome-free data primitives for the R005E2 training-resource rehearsal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from aorta_drive.vision.feature_cache import (
    bfloat16_bits_to_float32,
    load_dense_bfloat16_bits,
)


@dataclass(frozen=True)
class ResourceAnchor:
    recording_group: str
    anchor_index: int
    scene_token: str
    panel_path: Path
    feature_paths: tuple[Path, ...]


def candidate_tensors_from_panel_arrays(
    realized_local_poses: npt.ArrayLike,
    challenger_eligible: npt.ArrayLike,
) -> tuple[Tensor, Tensor]:
    """Create fixed storage while preserving the frozen ragged legal panel."""

    poses = np.asarray(realized_local_poses)
    eligible = np.asarray(challenger_eligible)
    if poses.dtype != np.float64 or poses.shape != (7, 41, 3):
        raise ValueError("panel poses must have float64 shape [7, 41, 3]")
    if eligible.dtype != np.bool_ or eligible.shape != (6,):
        raise ValueError("challenger eligibility must have bool shape [6]")
    if not np.all(np.isfinite(poses)):
        raise ValueError("panel poses must be finite")
    mask = np.concatenate((np.ones(1, dtype=np.bool_), eligible))
    storage = poses.copy()
    storage[~mask] = 0.0
    return torch.from_numpy(storage.astype(np.float32)), torch.from_numpy(mask)


def load_visual_tensor(feature_paths: Sequence[Path]) -> Tensor:
    """Load exactly twelve frozen dense maps as [slot, height, width, channel]."""

    if len(feature_paths) != 12:
        raise ValueError("R005E2 requires exactly twelve visual slots")
    maps = [
        bfloat16_bits_to_float32(load_dense_bfloat16_bits(path)).transpose(1, 2, 0)
        for path in feature_paths
    ]
    visual = np.ascontiguousarray(np.stack(maps, axis=0), dtype=np.float32)
    return torch.from_numpy(visual).to(dtype=torch.bfloat16)


def visual_records_by_identity(
    payload: Mapping[str, Any], root: Path
) -> dict[tuple[str, str], tuple[Path, ...]]:
    """Index a validated visual manifest without accepting ambiguous identities."""

    if payload.get("schema") != "aorta.r005e2.frozen_visual_cache.v1":
        raise ValueError("unexpected R005E2 visual-cache schema")
    if payload.get("status") != "PASS":
        raise ValueError("R005E2 visual cache is not complete")
    if payload.get("development_calibration_test_navdream_target_accessed") is not False:
        raise ValueError("R005E2 visual cache accessed a sealed role")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("visual manifest must contain records")
    result: dict[tuple[str, str], tuple[Path, ...]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("visual record must be an object")
        slots = record.get("slots")
        if not isinstance(slots, list) or len(slots) != 12:
            raise ValueError("visual record must contain twelve slots")
        ordered = sorted(slots, key=lambda slot: int(slot["slot_index"]))
        if [int(slot["slot_index"]) for slot in ordered] != list(range(12)):
            raise ValueError("visual slot indices are not exact")
        key = (str(record["log_name"]), str(record["scene_token"]))
        if key in result:
            raise ValueError("duplicate visual identity")
        result[key] = tuple(root / str(slot["feature_relative_path"]) for slot in ordered)
    return result


class ResourceRehearsalDataset(torch.utils.data.Dataset[dict[str, Tensor]]):
    """Real I/O and geometry with zero scientific targets for resource qualification."""

    def __init__(self, anchors: Sequence[ResourceAnchor]) -> None:
        if not anchors:
            raise ValueError("resource rehearsal requires at least one anchor")
        self._anchors = tuple(anchors)

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        anchor = self._anchors[index]
        with np.load(anchor.panel_path, allow_pickle=False) as panel:
            candidate, mask = candidate_tensors_from_panel_arrays(
                panel["realized_local_poses"], panel["challenger_eligible"]
            )
        visual = load_visual_tensor(anchor.feature_paths)
        target = torch.zeros(7, dtype=torch.float32)
        return {
            "candidate_poses": candidate,
            "candidate_mask": mask,
            "previous_poses": candidate[0].clone(),
            "visual_features": visual,
            "visual_slot_mask": torch.ones(12, dtype=torch.bool),
            "target_margin": target,
        }
