"""Materialize the exact R005E2 tensors and outcome-free removed controls."""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import Pose2D
from aorta_drive.evaluation.h4 import (
    sample_cached_native_trajectory,
)
from aorta_drive.evaluation.navsim_execution import execute_official_navsim_batch
from aorta_drive.training.r005e2 import (
    candidate_tensors_from_panel_arrays,
    visual_records_by_identity,
)
from aorta_drive.training.r005e2_science import (
    identity,
    validate_bundle_arrays,
)
from aorta_drive.training.r005e2_context import (
    align_previous_simulated_context,
    array_sha256,
    cache_time_s,
)
from aorta_drive.vision.feature_cache import (
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


def _records_for_keys(
    payload: dict[str, Any], required_keys: set[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in cast(list[dict[str, Any]], payload["records"]):
        key = identity(str(record["recording_group"]), int(record["anchor_index"]))
        if key not in required_keys:
            continue
        if key in result:
            raise ValueError("metric manifest has duplicate required identity")
        result[key] = record
    if set(result) != required_keys:
        raise ValueError("metric manifest does not uniquely cover the frozen roster")
    return result


def _load_cache(root: Path, record: dict[str, Any], *, role: str) -> Any:
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / str(record["relative_cache_path"])).resolve(strict=True)
    if resolved_root not in path.parents:
        raise ValueError(f"{role} metric-cache path escapes its frozen root")
    if sha256_file(path) != str(record["cache_sha256"]):
        raise ValueError(f"{role} metric-cache hash mismatch")
    with lzma.open(path, "rb") as stream:
        return pickle.load(stream)  # noqa: S301 - frozen official NAVSIM cache


def _materialize_previous_context(
    task: tuple[
        dict[str, Any],
        dict[str, Any],
        str,
        float,
        str,
        str,
    ],
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    (
        current_record,
        previous_record,
        expected_states_hash,
        expected_gap_s,
        current_root_text,
        previous_root_text,
    ) = task
    current_cache = _load_cache(Path(current_root_text), current_record, role="current")
    previous_cache = _load_cache(
        Path(previous_root_text), previous_record, role="previous"
    )
    previous_native = np.asarray(
        sample_cached_native_trajectory(previous_cache).poses[1:], dtype=np.float64
    )
    if previous_native.shape != (8, 3) or not np.isfinite(previous_native).all():
        raise ValueError("previous native planner output must be finite [8,3]")
    execution = execute_official_navsim_batch(
        previous_cache, previous_native[None, ...]
    )
    if not execution.deterministic_replay_bit_exact:
        raise ValueError("previous native official simulation is nondeterministic")
    previous_states = np.asarray(execution.realized_states[1], dtype=np.float64)
    if previous_states.shape != (41, 11):
        raise ValueError("previous native official states must have shape [41,11]")
    observed_hash = array_sha256(previous_states)
    if observed_hash != expected_states_hash:
        raise ValueError(
            "previous simulated states differ from R005E1B target provenance"
        )
    previous_time = cache_time_s(previous_cache)
    current_time = cache_time_s(current_cache)
    if abs((current_time - previous_time) - expected_gap_s) > 1.0e-6:
        raise ValueError("history timestamps differ from R005E1B target provenance")
    current_rear_axle = current_cache.ego_state.rear_axle
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (  # type: ignore[import-not-found]
        StateIndex,
    )

    context = align_previous_simulated_context(
        previous_states,
        previous_time_s=previous_time,
        current_time_s=current_time,
        current_pose=Pose2D(
            float(current_rear_axle.x),
            float(current_rear_axle.y),
            float(current_rear_axle.heading),
        ),
        simulation_interval_s=0.1,
        x_index=int(StateIndex.X),
        y_index=int(StateIndex.Y),
        heading_index=int(StateIndex.HEADING),
    )
    if context.overlap_start != 5 or context.poses.shape != (36, 3):
        raise ValueError(
            "previous/current overlap differs from the frozen 0.5 s contract"
        )
    return context.poses, {
        "previous_simulated_states_sha256": observed_hash,
        "overlap_start": context.overlap_start,
        "observed_gap_s": context.observed_gap_s,
        "maximum_transform_error_m": context.maximum_transform_error_m,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.history_workers <= 0:
        raise ValueError("history-workers must be positive")
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)
    roster = cast(dict[str, Any], json.loads(args.panel_roster.read_text("utf-8")))
    target = cast(dict[str, Any], json.loads(args.target_result.read_text("utf-8")))
    previous = cast(
        dict[str, Any], json.loads(args.previous_manifest.read_text("utf-8"))
    )
    current = cast(dict[str, Any], json.loads(args.current_manifest.read_text("utf-8")))
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
    required_keys = {
        identity(str(row["recording_group"]), int(row["anchor_index"])) for row in rows
    }
    if len(rows) != 672 or len(required_keys) != 672:
        raise ValueError("panel roster must contain 672 unique anchors")
    target_by_key = {
        identity(str(row["recording_group"]), int(row["anchor_index"])): row
        for row in cast(list[dict[str, Any]], target["anchor_ledgers"])
    }
    previous_by_key = _records_for_keys(previous, required_keys)
    current_by_key = _records_for_keys(current, required_keys)
    if set(target_by_key) != required_keys:
        raise ValueError("target ledger identities differ from the frozen roster")
    visual_index = visual_records_by_identity(visual, args.visual_root)

    history_tasks: list[tuple[dict[str, Any], dict[str, Any], str, float, str, str]] = (
        []
    )
    for row in rows:
        key = identity(str(row["recording_group"]), int(row["anchor_index"]))
        target_row = target_by_key[key]
        current_row = current_by_key[key]
        previous_row = previous_by_key[key]
        if str(current_row["scene_token"]) != str(target_row["scene_token"]) or str(
            previous_row["scene_token"]
        ) != str(target_row["previous_token"]):
            raise ValueError("history cache identities differ from the target ledger")
        history_tasks.append(
            (
                current_row,
                previous_row,
                str(target_row["previous_native_simulated_states_sha256"]),
                float(target_row["observation_interval_s"]),
                str(args.current_root),
                str(args.previous_root),
            )
        )
    with ProcessPoolExecutor(max_workers=args.history_workers) as executor:
        history_rows = list(executor.map(_materialize_previous_context, history_tasks))

    candidate = np.zeros((672, 7, 41, 3), dtype=np.float32)
    mask = np.zeros((672, 7), dtype=np.bool_)
    previous_poses = np.zeros((672, 36, 3), dtype=np.float32)
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
        previous_poses[index] = history_rows[index][0].astype(np.float32)
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
                "history_provenance": history_rows[index][1],
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
        "previous_context_contract": {
            "source": "R005E1B_hash_verified_previous_native_official_simulated_states",
            "coordinate_frame": "current_anchor_ego_rear_axle",
            "temporal_support": "previous_states[round(gap/0.1):]",
            "shape": [36, 3],
            "sample_interval_s": 0.1,
            "shared_current_candidate_horizon_s": 3.5,
            "expired_previous_prefix_retained": False,
            "interpolation_or_padding_used": False,
        },
        "removed_control": "outer_train_camera_history_slot_mean_over_anchors_and_patches",
        "array_sha256": array_hashes,
        "removed_slot_mean_sha256": mean_hashes,
        "inputs": {
            "config_sha256": sha256_file(args.config),
            "panel_roster_sha256": sha256_file(args.panel_roster),
            "target_result_sha256": sha256_file(args.target_result),
            "current_manifest_sha256": sha256_file(args.current_manifest),
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
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, action="append", required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--history-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: value for key, value in result.items() if key != "anchors"}))
