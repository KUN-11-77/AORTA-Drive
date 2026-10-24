"""Attribute H7BS4A failures to pair-common or distinguishing query regions."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.shadow_decision import build_shadow_panel
from aorta_drive.evaluation.visual_observability import pairwise_contrast_observability
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4a_fit_projective_observability import _projective_evidence, _write_json


def _summarize(rows: list[dict[str, Any]], blind_keys: set[tuple[str, int, int]]) -> dict[str, Any]:
    pair_rows = [pair for row in rows for pair in row["pair_attributions"]]
    distinct = [pair for pair in pair_rows if pair["geometrically_distinct"]]
    blind_pair_rows = [
        pair
        for row in rows
        for pair in row["pair_attributions"]
        if (row["recording_group"], row["anchor_index"], pair["reference_registered_index"])
        in blind_keys
    ]
    blind_native_keys = {key for key in blind_keys if key[2] == 0}
    blind_native_keys_with_pairs = {
        (str(row["recording_group"]), int(row["anchor_index"]), 0)
        for row in rows
        if row["native_is_zero_evidence_h7bs4a_failure"] and row["pair_attributions"]
    }
    zero_supported_distinguishing = sum(
        pair["symmetric_difference"]["supported_node_count"] == 0 for pair in distinct
    )
    blind_all_common = blind_native_keys == blind_native_keys_with_pairs and all(
        pair["unsupported_reference_nodes_all_common"] for pair in blind_pair_rows
    )
    blind_all_distinguishing_supported = (
        blind_native_keys == blind_native_keys_with_pairs
        and bool(blind_pair_rows)
        and all(
        pair["symmetric_difference"]["supported_node_count"] > 0
        for pair in blind_pair_rows
        if pair["geometrically_distinct"]
        )
    )
    if zero_supported_distinguishing:
        attribution = "CONTRAST_INTERFACE_ALSO_HAS_ZERO_EVIDENCE_PAIRS"
    elif blind_all_common and blind_all_distinguishing_supported:
        attribution = "COMMON_REGION_ATTRIBUTION_SUPPORTED__SUFFICIENCY_UNRESOLVED"
    else:
        attribution = "MIXED_ATTRIBUTION__SUFFICIENCY_UNRESOLVED"
    return {
        "anchor_count": len(rows),
        "native_challenger_pair_count": len(pair_rows),
        "geometrically_distinct_pair_count": len(distinct),
        "geometrically_identical_pair_count": len(pair_rows) - len(distinct),
        "distinct_pairs_with_zero_supported_symmetric_difference": zero_supported_distinguishing,
        "blind_native_pair_count": len(blind_pair_rows),
        "blind_native_candidate_count": len(blind_native_keys),
        "blind_native_candidates_without_eligible_pair": len(
            blind_native_keys - blind_native_keys_with_pairs
        ),
        "blind_native_unsupported_nodes_all_pair_common": blind_all_common,
        "blind_native_distinguishing_regions_all_have_evidence": (
            blind_all_distinguishing_supported
        ),
        "attribution": attribution,
        "scientific_decision": "POSTHOC_ATTRIBUTION_ONLY__H7BS4A_STOP_UNCHANGED",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(Mapping[str, Any], config["H7BS4AD0"])
    repository = Path(__file__).resolve().parents[1]
    frozen = cast(Mapping[str, str], stage["frozen_input_sha256"])
    for name, path in {
        "h7bs4a_result": args.h7bs4a_result,
        "h7bs4a_config": args.h7bs4a_config,
        "h7bs1_fit_result": args.fit_result,
    }.items():
        if sha256_file(path) != frozen[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")
    implementation = cast(Mapping[str, str], stage["implementation_sha256"])
    for name, path in {
        "observability_source": repository / "src/aorta_drive/evaluation/visual_observability.py",
        "shadow_decision_source": repository / "src/aorta_drive/evaluation/shadow_decision.py",
        "h7bs4a_runner_source": repository / "scripts/run_r004h7bs4a_fit_projective_observability.py",
        "runner_source": Path(__file__).resolve(),
    }.items():
        if sha256_file(path) != implementation[name]:
            raise ValueError(f"implementation hash mismatch: {name}")
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)

    source = cast(dict[str, Any], json.loads(args.h7bs4a_result.read_text(encoding="utf-8")))
    if source.get("forbidden_payload_accessed") is not False:
        raise ValueError("H7BS4A source does not preserve the forbidden-payload boundary")
    source_rows = cast(list[dict[str, Any]], source["anchor_ledgers"])
    if len(source_rows) != 672:
        raise ValueError("H7BS4A result must contain exactly 672 anchors")
    fit = cast(dict[str, Any], json.loads(args.fit_result.read_text(encoding="utf-8")))
    fit_rows = cast(list[dict[str, Any]], fit["anchor_ledgers"])
    fit_by_key = {
        (str(row["recording_group"]), int(row["anchor_index"])): row for row in fit_rows
    }
    source_keys = {
        (str(row["recording_group"]), int(row["anchor_index"])) for row in source_rows
    }
    if set(fit_by_key) != source_keys or len(source_keys) != 672:
        raise ValueError("fit and H7BS4A anchor identities do not align")

    blind_keys = {
        (str(row["recording_group"]), int(row["anchor_index"]), int(metric["registered_index"]))
        for row in source_rows
        for metric in row["candidate_observability"]
        if not bool(metric["has_any_direct_projective_evidence"])
    }
    if len(blind_keys) != int(source["summary"]["zero_direct_evidence_candidate_count"]):
        raise ValueError("blind candidate count does not match frozen H7BS4A summary")
    if any(key[2] != 0 for key in blind_keys):
        raise ValueError("registered D0 attribution assumes every H7BS4A blind candidate is native")

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
    raw_shape = tuple(int(v) for v in projection["raw_shape"])
    resized_shape = tuple(int(v) for v in projection["resized_shape"])
    if len(raw_shape) != 2 or len(resized_shape) != 2:
        raise ValueError("projection image shapes must have two dimensions")

    log_cache: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    source_log_hashes = cast(dict[str, str], source["source_log_sha256"])
    for source_row in source_rows:
        group = str(source_row["recording_group"])
        anchor = int(source_row["anchor_index"])
        log_name = str(source_row["log_name"])
        if log_name not in log_cache:
            path = args.log_root / f"{log_name}.pkl"
            if sha256_file(path) != source_log_hashes[log_name]:
                raise ValueError("source log hash mismatch")
            with path.open("rb") as stream:
                log_cache[log_name] = pickle.load(stream)  # noqa: S301 - frozen official input
        frames = log_cache[log_name]
        token_to_index = {str(frame["token"]): index for index, frame in enumerate(frames)}
        current_index = token_to_index[str(source_row["scene_token"])]
        has_evidence, _ = _projective_evidence(
            grid=grid,
            frames=frames,
            current_index=current_index,
            resized_shape=(resized_shape[0], resized_shape[1]),
            raw_shape=(raw_shape[0], raw_shape[1]),
            patch_size=int(projection["patch_size"]),
            maximum_normalized_radius=float(projection["maximum_normalized_radius"]),
        )

        fit_row = fit_by_key[(group, anchor)]
        artifact = (args.panel_run_root / str(fit_row["panel_artifact"])).resolve()
        if not artifact.is_relative_to(args.panel_run_root.resolve()):
            raise ValueError("panel artifact escapes frozen run root")
        if sha256_file(artifact) != source_row["panel_artifact_sha256"]:
            raise ValueError("panel artifact hash mismatch")
        with np.load(artifact, allow_pickle=False) as payload:
            trajectories = np.asarray(payload["realized_local_poses"], dtype=np.float64)
            eligible = tuple(bool(value) for value in payload["challenger_eligible"])
        panel = build_shadow_panel(
            trajectories,
            challenger_eligible=eligible,
            identifiers=("native",) + tuple(f"challenger_{index}" for index in range(1, 7)),
            deterministic_execution=True,
            grid=grid,
            body=body,
        )
        query_candidates = tuple(candidate for candidate in panel.candidates if candidate.valid)
        if not query_candidates:
            raise ValueError("frozen anchor has no valid native query")
        native = query_candidates[0]
        if native.registered_index != 0:
            raise ValueError("native candidate is not first")
        pairs = []
        for challenger in query_candidates[1:]:
            pairs.append(
                {
                    "reference_registered_index": native.registered_index,
                    "comparison_registered_index": challenger.registered_index,
                    **pairwise_contrast_observability(
                        native.node_indices, challenger.node_indices, has_evidence
                    ),
                }
            )
        rows.append(
            {
                "recording_group": group,
                "anchor_index": anchor,
                "scene_token": source_row["scene_token"],
                "log_name": log_name,
                "panel_artifact_sha256": source_row["panel_artifact_sha256"],
                "native_is_zero_evidence_h7bs4a_failure": (group, anchor, 0) in blind_keys,
                "pair_attributions": pairs,
            }
        )

    summary = _summarize(rows, blind_keys)
    elapsed = time.perf_counter() - started
    if elapsed > float(stage["resources"]["maximum_wall_minutes"]) * 60.0:
        raise RuntimeError("H7BS4A-D0 exceeded frozen wall-time cap")
    result = {
        "experiment_id": config["experiment_id"],
        "code_commit": args.code_commit,
        "parent_h7bs4a_decision_preserved": True,
        "forbidden_payload_accessed": False,
        "anchor_ledgers": rows,
        "summary": summary,
        "runtime_seconds": elapsed,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if len(encoded.encode("utf-8")) > float(stage["resources"]["maximum_output_mib"]) * 2**20:
        raise RuntimeError("H7BS4A-D0 exceeded frozen output-size cap")
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--h7bs4a-result", type=Path, required=True)
    parser.add_argument("--h7bs4a-config", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--panel-run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, allow_nan=False))
