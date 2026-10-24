"""Run the fit-only H7BS4A projective observability precondition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import numpy.typing as npt
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.shadow_decision import build_shadow_panel
from aorta_drive.evaluation.visual_observability import (
    candidate_observability,
    summarize_group_observability,
)
from aorta_drive.vision.feature_cache import sha256_file
from aorta_drive.vision.projection import (
    project_points_to_patch_grid,
    transform_current_to_history,
    yaw_from_wxyz,
)

CAMERAS = ("cam_l0", "cam_f0", "cam_r0")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _groups(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = tuple(row["recording_group"] for row in rows if row["role"] == "fit")
    if len(groups) != 42 or len(set(groups)) != 42:
        raise ValueError("role manifest must contain 42 unique fit groups")
    return groups


def _anchor_sources(path: Path, groups: tuple[str, ...]) -> dict[tuple[str, int], dict[str, str]]:
    selected = set(groups)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["recording_group"] in selected]
    result = {
        (row["recording_group"], int(row["anchor_index"])): {
            "scene_token": row["scene_token"],
            "log_name": row["log_name"],
        }
        for row in rows
    }
    expected = {(group, anchor) for group in groups for anchor in range(16)}
    if set(result) != expected or len(rows) != len(expected):
        raise ValueError("time set does not exactly cover the fit panel")
    return result


def _pose(frame: Mapping[str, Any]) -> tuple[npt.NDArray[np.float64], float]:
    translation = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("invalid ego translation")
    return translation[:2], yaw_from_wxyz(frame["ego2global_rotation"])


def _camera(frame: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    cameras = {str(key).lower(): value for key, value in frame["cams"].items()}
    value = cameras.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing camera metadata: {name}")
    return value


def _projective_evidence(
    *,
    grid: RectangularGrid,
    frames: list[dict[str, Any]],
    current_index: int,
    resized_shape: tuple[int, int],
    raw_shape: tuple[int, int],
    patch_size: int,
    maximum_normalized_radius: float,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    points_xy = grid.centers()
    current_translation, current_yaw = _pose(frames[current_index])
    masks: list[npt.NDArray[np.bool_]] = []
    for frame_index in range(current_index - 3, current_index + 1):
        frame = frames[frame_index]
        history_translation, history_yaw = _pose(frame)
        history_xy = transform_current_to_history(
            points_xy,
            current_translation,
            current_yaw,
            history_translation,
            history_yaw,
        )
        history_xyz = np.column_stack((history_xy, np.zeros(len(history_xy), dtype=np.float64)))
        for name in CAMERAS:
            camera = _camera(frame, name)
            projection = project_points_to_patch_grid(
                history_xyz,
                camera["sensor2lidar_rotation"],
                camera["sensor2lidar_translation"],
                camera["cam_intrinsic"],
                camera["distortion"],
                raw_shape=raw_shape,
                resized_shape=resized_shape,
                patch_size=patch_size,
                maximum_normalized_radius=maximum_normalized_radius,
            )
            masks.append(projection.valid_for_bilinear_patch_sampling)
    slot_mask = np.stack(masks, axis=1)
    return np.any(slot_mask, axis=1), np.sum(slot_mask, axis=1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(Mapping[str, Any], config["H7BS4A"])
    repository = Path(__file__).resolve().parents[1]
    frozen = cast(Mapping[str, str], stage["frozen_input_sha256"])
    for input_name, path in {
        "time_set": args.time_set,
        "group_role_manifest": args.group_role_manifest,
        "h7bs1_fit_result": args.fit_result,
        "h7bs3_result": args.h7bs3_result,
        "r004b_result": args.r004b_result,
    }.items():
        if sha256_file(path) != frozen[input_name]:
            raise ValueError(f"frozen input hash mismatch: {input_name}")
    implementation = cast(Mapping[str, str], stage["implementation_sha256"])
    for source_name, path in {
        "observability_source": repository / "src/aorta_drive/evaluation/visual_observability.py",
        "projection_source": repository / "src/aorta_drive/vision/projection.py",
        "shadow_decision_source": repository / "src/aorta_drive/evaluation/shadow_decision.py",
        "runner_source": Path(__file__).resolve(),
    }.items():
        if sha256_file(path) != implementation[source_name]:
            raise ValueError(f"implementation hash mismatch: {source_name}")
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)

    groups = _groups(args.group_role_manifest)
    sources = _anchor_sources(args.time_set, groups)
    payload = cast(dict[str, Any], json.loads(args.fit_result.read_text(encoding="utf-8")))
    ledgers = cast(list[dict[str, Any]], payload["anchor_ledgers"])
    ledger_by_key = {
        (str(row["recording_group"]), int(row["anchor_index"])): row for row in ledgers
    }
    expected = {(group, anchor) for group in groups for anchor in range(16)}
    if set(ledger_by_key) != expected or len(ledgers) != 672:
        raise ValueError("fit result does not exactly cover 42x16")

    roi = cast(Mapping[str, Any], stage["roi"])
    grid = RectangularGrid(
        x_min=float(roi["x_min_m"]),
        y_min=float(roi["y_min_m"]),
        columns=int(roi["columns"]),
        rows=int(roi["rows"]),
        resolution=float(roi["resolution_m"]),
    )
    vehicle = cast(Mapping[str, Any], stage["vehicle"])
    body = VehicleRectangle(
        rear_overhang=float(vehicle["rear_overhang_m"]),
        front_extent=float(vehicle["front_extent_m"]),
        half_width=float(vehicle["half_width_m"]),
    )
    projection = cast(Mapping[str, Any], stage["projection"])
    log_cache: dict[str, list[dict[str, Any]]] = {}
    log_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    group_order = {group: index for index, group in enumerate(groups)}
    resized_values = tuple(int(v) for v in projection["resized_shape"])
    raw_values = tuple(int(v) for v in projection["raw_shape"])
    if len(resized_values) != 2 or len(raw_values) != 2:
        raise ValueError("projection image shapes must have two dimensions")
    resized_shape = (resized_values[0], resized_values[1])
    raw_shape = (raw_values[0], raw_values[1])
    for anchor_key in sorted(expected, key=lambda item: (group_order[item[0]], item[1])):
        source = sources[anchor_key]
        log_name = source["log_name"]
        if log_name not in log_cache:
            log_path = args.log_root / f"{log_name}.pkl"
            if not log_path.is_file():
                raise FileNotFoundError(log_path)
            log_hashes[log_name] = sha256_file(log_path)
            with log_path.open("rb") as stream:
                log_cache[log_name] = pickle.load(stream)  # noqa: S301 - frozen official input
        frames = log_cache[log_name]
        token_to_index = {str(frame["token"]): index for index, frame in enumerate(frames)}
        scene_token = source["scene_token"]
        if scene_token not in token_to_index:
            raise ValueError("scene token missing from frozen source log")
        current_index = token_to_index[scene_token]
        if current_index < 3:
            raise ValueError("insufficient camera history")
        has_evidence, slot_count = _projective_evidence(
            grid=grid,
            frames=frames,
            current_index=current_index,
            resized_shape=resized_shape,
            raw_shape=raw_shape,
            patch_size=int(projection["patch_size"]),
            maximum_normalized_radius=float(projection["maximum_normalized_radius"]),
        )

        ledger = ledger_by_key[anchor_key]
        artifact = (args.panel_run_root / str(ledger["panel_artifact"])).resolve()
        if not artifact.is_relative_to(args.panel_run_root.resolve()):
            raise ValueError("panel artifact escapes frozen run root")
        if sha256_file(artifact) != ledger["panel_artifact_sha256"]:
            raise ValueError("panel artifact hash mismatch")
        with np.load(artifact, allow_pickle=False) as panel_payload:
            trajectories = np.asarray(panel_payload["realized_local_poses"], dtype=np.float64)
            eligible = tuple(bool(v) for v in panel_payload["challenger_eligible"])
        panel = build_shadow_panel(
            trajectories,
            challenger_eligible=eligible,
            identifiers=("native",) + tuple(f"challenger_{i}" for i in range(1, 7)),
            deterministic_execution=True,
            grid=grid,
            body=body,
        )
        query_candidates = [candidate for candidate in panel.candidates if candidate.valid]
        metrics = [
            {
                "registered_index": candidate.registered_index,
                **candidate_observability(
                    candidate.node_indices,
                    has_evidence,
                    grid.centers()[:, 0],
                    forward_extension_start_m=float(stage["forward_extension_start_m"]),
                ),
            }
            for candidate in query_candidates
        ]
        rows.append(
            {
                "recording_group": anchor_key[0],
                "anchor_index": anchor_key[1],
                "scene_token": scene_token,
                "log_name": log_name,
                "panel_artifact_sha256": str(ledger["panel_artifact_sha256"]),
                "grid_supported_node_count": int(np.count_nonzero(has_evidence)),
                "grid_supported_node_fraction": float(np.mean(has_evidence)),
                "grid_mean_evidence_slot_count": float(np.mean(slot_count)),
                "candidate_observability": metrics,
                "all_query_candidates_have_direct_evidence": all(
                    bool(metric["has_any_direct_projective_evidence"]) for metric in metrics
                ),
            }
        )

    summary = summarize_group_observability(
        rows,
        group_names=groups,
        anchors_per_group=16,
        minimum_anchors_per_group=int(stage["minimum_direct_evidence_anchors_per_group"]),
    )
    summary["unique_source_log_count"] = len(log_hashes)
    summary["mean_grid_supported_node_fraction"] = float(
        np.mean([row["grid_supported_node_fraction"] for row in rows])
    )
    elapsed = time.perf_counter() - started
    if elapsed > float(stage["resources"]["maximum_wall_minutes"]) * 60.0:
        raise RuntimeError("H7BS4A exceeded frozen wall-time cap")
    result = {
        "experiment_id": config["experiment_id"],
        "code_commit": args.code_commit,
        "forbidden_payload_accessed": False,
        "source_log_sha256": log_hashes,
        "anchor_ledgers": rows,
        "summary": summary,
        "runtime_seconds": elapsed,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if len(encoded.encode("utf-8")) > float(stage["resources"]["maximum_output_mib"]) * 2**20:
        raise RuntimeError("H7BS4A exceeded frozen output-size cap")
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--panel-run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--h7bs3-result", type=Path, required=True)
    parser.add_argument("--r004b-result", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, allow_nan=False))
