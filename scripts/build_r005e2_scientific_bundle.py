"""Materialize the exact R005E2 tensors and outcome-free removed controls."""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

from aorta_drive.evaluation.h4 import (  # type: ignore[import-untyped]
    sample_cached_native_trajectory,
)
from aorta_drive.training.r005e2 import (  # type: ignore[import-untyped]
    candidate_tensors_from_panel_arrays,
    visual_records_by_identity,
)
from aorta_drive.training.r005e2_science import (  # type: ignore[import-untyped]
    identity,
    validate_bundle_arrays,
)
from aorta_drive.vision.feature_cache import (  # type: ignore[import-untyped]
    bfloat16_bits_to_float32,
    load_dense_bfloat16_bits,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _resolve_panel(roots: list[Path], scene_token: str, expected_hash: str) -> Path:
    matches = [
        path
        for root in roots
        if (path := root / f"{scene_token}.npz").is_file()
        and sha256_file(path) == expected_hash
    ]
    if len(matches) != 1:
        raise ValueError("panel must resolve exactly once by frozen hash")
    return matches[0]


def _load_previous(root: Path, record: dict[str, Any]) -> npt.NDArray[np.float64]:
    path = root / str(record["relative_cache_path"])
    if sha256_file(path) != str(record["cache_sha256"]):
        raise ValueError("previous metric-cache hash mismatch")
    with lzma.open(path, "rb") as stream:
        cache = pickle.load(stream)  # noqa: S301 - frozen official NAVSIM cache
    poses = np.asarray(
        sample_cached_native_trajectory(cache).poses[1:], dtype=np.float64
    )
    if poses.shape != (41, 3) or not np.isfinite(poses).all():
        raise ValueError("previous native trajectory contract failed")
    return poses


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)
    roster = cast(dict[str, Any], json.loads(args.panel_roster.read_text("utf-8")))
    target = cast(dict[str, Any], json.loads(args.target_result.read_text("utf-8")))
    previous = cast(
        dict[str, Any], json.loads(args.previous_manifest.read_text("utf-8"))
    )
    visual = cast(dict[str, Any], json.loads(args.visual_manifest.read_text("utf-8")))
    crossfit = cast(
        dict[str, Any], json.loads(args.crossfit_manifest.read_text("utf-8"))
    )
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text("utf-8")))
    if sha256_file(args.target_result) != str(config["frozen_target"]["result_sha256"]):
        raise ValueError("target result differs from the preregistered SHA")
    if sha256_file(args.crossfit_manifest) != str(
        config["cross_fitting"]["manifest_sha256"]
    ):
        raise ValueError("cross-fit manifest differs from the preregistered SHA")
    if sha256_file(args.panel_roster) != str(
        config["preflight"]["panel_identity_roster"]["sha256"]
    ):
        raise ValueError("panel roster differs from the preregistered SHA")
    feature_contract = cast(dict[str, Any], visual["feature_contract"])
    if (
        str(feature_contract["checkpoint_sha256"])
        != str(config["visual_backbone"]["checkpoint_sha256"])
        or str(feature_contract["dino_python_source_tree_sha256"])
        != str(config["visual_backbone"]["source_sha256"])
        or feature_contract["shape"] != [1280, 18, 32]
    ):
        raise ValueError("visual cache provenance differs from the preregistration")
    if target.get("development_calibration_test_navdream_target_accessed") is not False:
        raise ValueError("target source accessed a sealed role")
    rows = cast(list[dict[str, Any]], roster["anchors"])
    target_by_key = {
        identity(str(row["recording_group"]), int(row["anchor_index"])): row
        for row in cast(list[dict[str, Any]], target["anchor_ledgers"])
    }
    previous_by_key = {
        identity(str(row["recording_group"]), int(row["anchor_index"])): row
        for row in cast(list[dict[str, Any]], previous["records"])
    }
    if len(rows) != 672 or len(target_by_key) != 672 or len(previous_by_key) != 672:
        raise ValueError("bundle inputs must each contain exactly 672 anchors")
    visual_index = visual_records_by_identity(visual, args.visual_root)

    candidate = np.zeros((672, 7, 41, 3), dtype=np.float32)
    mask = np.zeros((672, 7), dtype=np.bool_)
    previous_poses = np.zeros((672, 41, 3), dtype=np.float32)
    target_margin = np.zeros((672, 7), dtype=np.float32)
    index_rows: list[dict[str, Any]] = []
    groups: list[str] = []
    for index, row in enumerate(rows):
        group = str(row["recording_group"])
        anchor_index = int(row["anchor_index"])
        key = identity(group, anchor_index)
        target_row = target_by_key[key]
        previous_row = previous_by_key[key]
        if str(target_row["scene_token"]) != str(row["scene_token"]) or str(
            previous_row["scene_token"]
        ) != str(row["previous_scene_token"]):
            raise ValueError("bundle identity join mismatch")
        panel_path = _resolve_panel(
            args.panel_root,
            str(row["scene_token"]),
            str(row["panel_artifact_sha256"]),
        )
        with np.load(panel_path, allow_pickle=False) as panel:
            candidate_tensor, mask_tensor = candidate_tensors_from_panel_arrays(
                panel["realized_local_poses"], panel["challenger_eligible"]
            )
        legal = [int(value) for value in target_row["legal_registered_indices"]]
        if legal != np.flatnonzero(mask_tensor.numpy()).tolist():
            raise ValueError("target legal domain differs from frozen panel")
        candidate[index] = candidate_tensor.numpy()
        mask[index] = mask_tensor.numpy()
        previous_poses[index] = _load_previous(args.previous_root, previous_row).astype(
            np.float32
        )
        utilities = {
            int(item["registered_index"]): float(item["contextual_utility"])
            for item in target_row["candidates"]
        }
        native = utilities[0]
        for registered_index in legal:
            value = utilities[registered_index] - native
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError("native-relative contextual margin is invalid")
            target_margin[index, registered_index] = np.float32(value)
        visual_key = (str(row["log_name"]), str(row["scene_token"]))
        if visual_key not in visual_index:
            raise ValueError("visual cache identity is missing")
        groups.append(group)
        index_rows.append(
            {
                "bundle_index": index,
                "identity": key,
                "recording_group": group,
                "anchor_index": anchor_index,
                "log_name": str(row["log_name"]),
                "scene_token": str(row["scene_token"]),
                "candidate_count": len(legal),
                "correct_feature_relative_paths": [
                    str(path.relative_to(args.visual_root))
                    for path in visual_index[visual_key]
                ],
            }
        )

    np.save(
        args.output_directory / "candidate_poses.npy", candidate, allow_pickle=False
    )
    np.save(args.output_directory / "candidate_mask.npy", mask, allow_pickle=False)
    np.save(
        args.output_directory / "previous_poses.npy", previous_poses, allow_pickle=False
    )
    np.save(
        args.output_directory / "target_margin.npy", target_margin, allow_pickle=False
    )

    fold_for_group = {
        str(group): int(fold)
        for group, fold in cast(dict[str, Any], crossfit["group_outer_fold"]).items()
    }
    global_sum = np.zeros((12, 1280), dtype=np.float64)
    heldout_sum = np.zeros((6, 12, 1280), dtype=np.float64)
    global_count = np.zeros(12, dtype=np.int64)
    heldout_count = np.zeros((6, 12), dtype=np.int64)
    for row in index_rows:
        fold = fold_for_group[str(row["recording_group"])]
        for slot_index, relative in enumerate(row["correct_feature_relative_paths"]):
            feature = bfloat16_bits_to_float32(
                load_dense_bfloat16_bits(args.visual_root / str(relative))
            )
            if feature.shape != (1280, 18, 32):
                raise ValueError("visual feature shape mismatch")
            slot_sum = feature.astype(np.float64).sum(axis=(1, 2))
            global_sum[slot_index] += slot_sum
            heldout_sum[fold, slot_index] += slot_sum
            global_count[slot_index] += 18 * 32
            heldout_count[fold, slot_index] += 18 * 32
    mean_hashes: dict[str, str] = {}
    for fold in range(6):
        denominator = global_count - heldout_count[fold]
        if not np.array_equal(denominator, np.full(12, 560 * 18 * 32)):
            raise ValueError("outer-train removed-token denominator mismatch")
        mean = ((global_sum - heldout_sum[fold]) / denominator[:, None]).astype(
            np.float32
        )[:, None, None, :]
        path = args.output_directory / f"removed_slot_mean_fold_{fold}.npy"
        np.save(path, mean, allow_pickle=False)
        mean_hashes[str(fold)] = sha256_file(path)

    validate_bundle_arrays(args.output_directory)
    array_hashes = {
        name: sha256_file(args.output_directory / name)
        for name in (
            "candidate_poses.npy",
            "candidate_mask.npy",
            "previous_poses.npy",
            "target_margin.npy",
        )
    }
    manifest = {
        "schema": "aorta.r005e2.scientific_bundle.v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "anchor_count": 672,
        "recording_group_count": len(set(groups)),
        "source_only": True,
        "development_calibration_test_navdream_target_accessed": False,
        "target_estimand": "candidate_Uctx_minus_native_Uctx",
        "removed_control": "outer_train_camera_history_slot_mean_over_anchors_and_patches",
        "array_sha256": array_hashes,
        "removed_slot_mean_sha256": mean_hashes,
        "inputs": {
            "config_sha256": sha256_file(args.config),
            "panel_roster_sha256": sha256_file(args.panel_roster),
            "target_result_sha256": sha256_file(args.target_result),
            "previous_manifest_sha256": sha256_file(args.previous_manifest),
            "visual_manifest_sha256": sha256_file(args.visual_manifest),
            "crossfit_manifest_sha256": sha256_file(args.crossfit_manifest),
        },
        "anchors": index_rows,
        "visual_feature_contract": feature_contract,
    }
    _write_json(args.output_directory / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel-roster", type=Path, required=True)
    parser.add_argument("--target-result", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, action="append", required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: value for key, value in result.items() if key != "anchors"}))
