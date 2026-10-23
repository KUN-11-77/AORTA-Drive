"""Exact feature/label join and runtime materialization for R004E-C."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
from torch import Tensor

from aorta_drive.vision.feature_cache import (
    bfloat16_bits_to_float32,
    load_dense_bfloat16_bits,
    validate_projection_arrays,
)


@dataclass(frozen=True)
class JoinedAnchor:
    """One frozen visual/teacher pair identified by an exact scene key."""

    recording_group: str
    anchor_index: int
    split: str
    log_name: str
    scene_token: str
    dense_feature_paths: tuple[Path, ...]
    projection_xy_path: Path
    projection_mask_path: Path
    label_path: Path
    label_artifact_sha256: str


@dataclass(frozen=True)
class LoadedAnchor:
    """Validated dense maps, projection coordinates, mask, and teacher arrays."""

    record: JoinedAnchor
    dense_features: npt.NDArray[np.float32]
    patch_xy: npt.NDArray[np.float64]
    mask: npt.NDArray[np.bool_]
    label_m: npt.NDArray[np.float64]
    total_error_m: npt.NDArray[np.float64]


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("manifest artifact path escapes its declared root")
    return candidate


def load_joined_anchors(
    feature_manifest: Path,
    label_manifest: Path,
    *,
    feature_root: Path | None = None,
    label_root: Path | None = None,
    allowed_splits: Sequence[str] = ("train", "development"),
) -> tuple[JoinedAnchor, ...]:
    """Load and validate the exact R004E-B/R004E-A manifest join.

    Both manifests must describe only the frozen source train/development time set.
    The exact full join is checked before the requested split subset is returned.
    """

    allowed = frozenset(allowed_splits)
    if not allowed or not allowed <= {"train", "development"}:
        raise ValueError("only train and development splits are permitted")
    feature_root = feature_manifest.parent if feature_root is None else feature_root
    label_root = label_manifest.parent if label_root is None else label_root
    visual = json.loads(feature_manifest.read_text(encoding="utf-8"))
    labels = json.loads(label_manifest.read_text(encoding="utf-8"))
    visual_records = visual.get("records")
    label_records = labels.get("records")
    if not isinstance(visual_records, list) or not isinstance(label_records, list):
        raise ValueError("both manifests must contain record lists")
    if visual.get("status") != "PASS":
        raise ValueError("frozen visual manifest must have PASS status")
    teacher_completed = (
        labels.get("status") == "completed"
        and labels.get("cache_failure_count") == 0
        and labels.get("cache_pass_count") == labels.get("cache_count")
        and labels.get("cache_count") == len(label_records)
    )
    if not teacher_completed:
        raise ValueError("frozen teacher manifest must be completed with zero failures")

    label_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for item in label_records:
        if not isinstance(item, dict) or not isinstance(item.get("cache_id"), str):
            raise ValueError("invalid label record")
        parts = item["cache_id"].split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError("label cache_id must be log_name/scene_token")
        key = (parts[0], parts[1])
        if key in label_by_key:
            raise ValueError("duplicate label key")
        label_by_key[key] = item

    joined: list[JoinedAnchor] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in visual_records:
        if not isinstance(item, dict):
            raise ValueError("invalid visual record")
        split = item.get("split")
        if split not in {"train", "development"}:
            raise ValueError("visual manifest contains a prohibited split")
        log_name = item.get("log_name")
        scene_token = item.get("scene_token")
        if not isinstance(log_name, str) or not isinstance(scene_token, str):
            raise ValueError("visual record lacks its exact scene key")
        key = (log_name, scene_token)
        if key in seen_keys:
            raise ValueError("duplicate visual scene key")
        seen_keys.add(key)
        label = label_by_key.get(key)
        if label is None:
            raise ValueError("visual record has no exact teacher label")
        slots = item.get("slots")
        if not isinstance(slots, list) or len(slots) != 12:
            raise ValueError("every visual record must have exactly 12 slots")
        ordered = sorted(slots, key=lambda slot: int(slot["slot_index"]))
        if [int(slot["slot_index"]) for slot in ordered] != list(range(12)):
            raise ValueError("slot indices must be exactly 0..11")
        if split not in allowed:
            continue
        artifact_name = label.get("artifact_name")
        artifact_sha256 = label.get("artifact_sha256")
        if not isinstance(artifact_name, str) or not isinstance(artifact_sha256, str):
            raise ValueError("label record lacks artifact provenance")
        joined.append(
            JoinedAnchor(
                recording_group=str(item["recording_group"]),
                anchor_index=int(item["anchor_index"]),
                split=str(split),
                log_name=log_name,
                scene_token=scene_token,
                dense_feature_paths=tuple(
                    _inside(feature_root, str(slot["feature_relative_path"]))
                    for slot in ordered
                ),
                projection_xy_path=_inside(
                    feature_root, str(item["projection_xy_relative_path"])
                ),
                projection_mask_path=_inside(
                    feature_root, str(item["projection_mask_relative_path"])
                ),
                label_path=_inside(label_root, artifact_name),
                label_artifact_sha256=artifact_sha256,
            )
        )
    if seen_keys != set(label_by_key):
        raise ValueError("feature and label manifests do not have an exact full key join")
    joined.sort(key=lambda row: (row.recording_group, row.anchor_index))
    if not joined:
        raise ValueError("no anchors selected")
    counts = Counter(row.recording_group for row in joined)
    if any(count != 16 for count in counts.values()):
        raise ValueError("every selected recording group must contain exactly 16 anchors")
    if len({(row.recording_group, row.anchor_index) for row in joined}) != len(joined):
        raise ValueError("duplicate recording-group/anchor index")
    return tuple(joined)


def _load_npy(path: Path) -> npt.NDArray[np.generic]:
    with path.open("rb") as stream:
        array = np.lib.format.read_array(  # type: ignore[no-untyped-call]
            stream, allow_pickle=False
        )
    return cast(npt.NDArray[np.generic], array)


def load_training_anchor(record: JoinedAnchor) -> LoadedAnchor:
    """Reopen and validate every artifact required for one training anchor."""

    dense = np.stack(
        [
            bfloat16_bits_to_float32(load_dense_bfloat16_bits(path))
            for path in record.dense_feature_paths
        ],
        axis=0,
    )
    patch_xy, mask = validate_projection_arrays(
        _load_npy(record.projection_xy_path), _load_npy(record.projection_mask_path)
    )
    with np.load(record.label_path, allow_pickle=False) as archive:
        required = {
            "label_m",
            "transform_error_m",
            "distance_error_m",
            "total_error_m",
        }
        if set(archive.files) != required:
            raise ValueError("teacher artifact arrays do not match the frozen contract")
        arrays = {name: np.asarray(archive[name]) for name in required}
    if any(array.dtype != np.float64 or array.shape != (64, 64) for array in arrays.values()):
        raise ValueError("teacher arrays must have float64 shape [64, 64]")
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("teacher arrays must be finite")
    total = arrays["total_error_m"]
    if (
        np.any(total < 0.0)
        or np.any(total < arrays["transform_error_m"])
        or np.any(total < arrays["distance_error_m"])
    ):
        raise ValueError("teacher error ledger is not outward-safe")
    return LoadedAnchor(
        record=record,
        dense_features=np.ascontiguousarray(dense, dtype=np.float32),
        patch_xy=patch_xy,
        mask=mask,
        label_m=np.ascontiguousarray(arrays["label_m"]),
        total_error_m=np.ascontiguousarray(total),
    )


def sample_dense_features_torch(
    dense_features: Tensor, patch_xy: Tensor, mask: Tensor
) -> Tensor:
    """Vectorized differentiable equivalent of the frozen NumPy sampling oracle."""

    if dense_features.ndim != 4:
        raise ValueError("dense_features must have shape [slots, channels, height, width]")
    slots, _, height, width = dense_features.shape
    if patch_xy.ndim != 3 or patch_xy.shape[1:] != (slots, 2):
        raise ValueError("patch_xy must have shape [nodes, slots, 2]")
    if mask.dtype is not torch.bool or mask.shape != patch_xy.shape[:2]:
        raise ValueError("mask must be boolean with shape [nodes, slots]")
    if not bool(torch.isfinite(dense_features).all().item()):
        raise ValueError("dense features must be finite")
    if not bool(torch.isfinite(patch_xy).all().item()):
        raise ValueError("patch coordinates must be finite")
    coordinates = patch_xy.to(device=dense_features.device, dtype=dense_features.dtype)
    normalized = torch.empty_like(coordinates)
    normalized[..., 0] = 2.0 * coordinates[..., 0] / (width - 1) - 1.0
    normalized[..., 1] = 2.0 * coordinates[..., 1] / (height - 1) - 1.0
    grid = normalized.transpose(0, 1).unsqueeze(2)
    sampled = functional.grid_sample(
        dense_features,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    output = sampled[..., 0].permute(2, 0, 1).contiguous()
    output = torch.where(mask.to(device=output.device).unsqueeze(-1), output, 0.0)
    if not bool(torch.isfinite(output).all().item()):
        raise ValueError("sampled node-slot features must be finite")
    return output


def deterministic_interleaved_epoch(
    records: Sequence[JoinedAnchor], *, seed: int, epoch: int
) -> tuple[JoinedAnchor, ...]:
    """Return each anchor once, interleaving independently shuffled recording groups."""

    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    grouped: dict[str, list[JoinedAnchor]] = defaultdict(list)
    for record in records:
        if record.split != "train":
            raise ValueError("training schedules may contain only train anchors")
        grouped[record.recording_group].append(record)
    if not grouped or any(len(group) != 16 for group in grouped.values()):
        raise ValueError("every training group must contain exactly 16 anchors")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    group_names = sorted(grouped)
    rng.shuffle(group_names)
    for name, group in grouped.items():
        group.sort(key=lambda row: row.anchor_index)
        grouped[name] = [group[int(index)] for index in rng.permutation(len(group))]
    schedule = tuple(grouped[name][position] for position in range(16) for name in group_names)
    if len(set(schedule)) != len(records) or len(schedule) != len(records):
        raise AssertionError("interleaved schedule is not an exact anchor permutation")
    return schedule
