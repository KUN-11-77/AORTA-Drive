"""One-shot outcome-blind development support gate for forward 96x64."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.evaluation.candidate_information import evaluate_roi_panel
from aorta_drive.vision.feature_cache import sha256_file


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _groups(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = tuple(row["recording_group"] for row in rows if row["role"] == "development")
    if len(groups) != 24 or len(set(groups)) != 24:
        raise ValueError("role manifest does not define 24 unique development groups")
    return groups


def _verify_inputs(stage: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    hashes = cast(Mapping[str, str], stage["frozen_input_sha256"])
    for key, path in paths.items():
        if sha256_file(path) != hashes[key]:
            raise ValueError(f"frozen input hash mismatch: {key}")


def compute_rows(
    *,
    panel_manifest: Mapping[str, Any],
    panel_root: Path,
    groups: tuple[str, ...],
    grid: RectangularGrid,
    body: VehicleRectangle,
) -> list[dict[str, Any]]:
    records = cast(list[dict[str, Any]], panel_manifest["records"])
    group_order = {group: index for index, group in enumerate(groups)}
    records.sort(key=lambda row: (group_order[str(row["recording_group"])], int(row["anchor_index"])))
    expected = {(group, anchor) for group in groups for anchor in range(16)}
    observed = {(str(row["recording_group"]), int(row["anchor_index"])) for row in records}
    if observed != expected or len(records) != 384:
        raise ValueError("panel manifest does not exactly cover 24x16 development anchors")
    rows: list[dict[str, Any]] = []
    for record in records:
        artifact = (panel_root / str(record["panel_artifact"])).resolve()
        if not artifact.is_relative_to(panel_root.resolve()):
            raise ValueError("panel artifact escapes frozen root")
        if sha256_file(artifact) != record["panel_artifact_sha256"]:
            raise ValueError("panel artifact hash mismatch")
        with np.load(artifact, allow_pickle=False) as payload:
            trajectories = np.asarray(payload["realized_local_poses"], dtype=np.float64)
            eligible = tuple(bool(value) for value in payload["challenger_eligible"])
        if trajectories.shape != (7, 41, 3) or len(eligible) != 6:
            raise ValueError("development panel artifact has unexpected shape")
        status = evaluate_roi_panel(
            trajectories,
            challenger_eligible=eligible,
            deterministic_execution=True,
            grid=grid,
            body=body,
        )
        candidate_valid = tuple(bool(value) for value in status["candidate_valid"])
        all_eligible_supported = candidate_valid[0] and all(
            candidate_valid[index] for index in range(1, 7) if eligible[index - 1]
        )
        rows.append(
            {
                "recording_group": str(record["recording_group"]),
                "anchor_index": int(record["anchor_index"]),
                "scene_token": str(record["scene_token"]),
                "panel_artifact": str(record["panel_artifact"]),
                "panel_artifact_sha256": str(record["panel_artifact_sha256"]),
                "eligible_challenger_count": int(sum(eligible)),
                "supported_eligible_challenger_count": int(
                    sum(candidate_valid[index] for index in range(1, 7) if eligible[index - 1])
                ),
                "native_supported": candidate_valid[0],
                "all_eligible_candidates_supported": all_eligible_supported,
                "panel_valid": bool(status["panel_valid"]),
                "panel_reason": str(status["panel_reason"]),
                "candidate_valid": list(candidate_valid),
                "candidate_reasons": status["candidate_reasons"],
            }
        )
    return rows


def summarize(
    rows: list[dict[str, Any]], *, groups: tuple[str, ...], minimum_valid: int
) -> dict[str, Any]:
    native_failures = sum(not bool(row["native_supported"]) for row in rows)
    eligible = sum(int(row["eligible_challenger_count"]) for row in rows)
    supported = sum(int(row["supported_eligible_challenger_count"]) for row in rows)
    all_candidate_failures = sum(
        not bool(row["all_eligible_candidates_supported"]) for row in rows
    )
    panel_counts = {
        group: sum(bool(row["panel_valid"]) for row in rows if row["recording_group"] == group)
        for group in groups
    }
    passing_groups = sum(value >= minimum_valid for value in panel_counts.values())
    support_pass = native_failures == 0 and supported == eligible and all_candidate_failures == 0
    panel_pass = passing_groups == len(groups)
    return {
        "group_count": len(groups),
        "anchor_count": len(rows),
        "native_support_failures": native_failures,
        "eligible_challenger_count": eligible,
        "supported_eligible_challenger_count": supported,
        "anchors_with_any_candidate_support_failure": all_candidate_failures,
        "valid_panel_count": sum(panel_counts.values()),
        "valid_panel_count_by_group": panel_counts,
        "groups_meeting_minimum_panel_coverage": passing_groups,
        "minimum_valid_panels_per_group": minimum_valid,
        "all_candidate_support_pass": support_pass,
        "panel_coverage_pass": panel_pass,
        "decision": (
            "DEVELOPMENT_FULL_DOMAIN_PASS"
            if support_pass and panel_pass
            else "STOP_FORWARD_96X64_DOMAIN"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(Mapping[str, Any], config["H7BS3"])
    repository = Path(__file__).resolve().parents[1]
    _verify_inputs(
        stage,
        {
            "group_role_manifest": args.group_role_manifest,
            "development_panel_manifest": args.panel_manifest,
            "h7bs1_development_support_result": args.h7bs1_development_result,
            "h7bs2_fit_result": args.h7bs2_result,
            "h7bs2_mechanical_audit": args.h7bs2_audit,
        },
    )
    implementation = cast(Mapping[str, str], stage["implementation_sha256"])
    for key, path in {
        "candidate_information_source": repository
        / "src/aorta_drive/evaluation/candidate_information.py",
        "runner_source": Path(__file__).resolve(),
    }.items():
        if sha256_file(path) != implementation[key]:
            raise ValueError(f"implementation hash mismatch: {key}")
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)
    groups = _groups(args.group_role_manifest)
    manifest = cast(dict[str, Any], json.loads(args.panel_manifest.read_text(encoding="utf-8")))
    if manifest.get("outcome_fields_included") is not False:
        raise ValueError("development panel manifest is not outcome-free")
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
    rows = compute_rows(
        panel_manifest=manifest,
        panel_root=args.panel_root,
        groups=groups,
        grid=grid,
        body=body,
    )
    summary = summarize(
        rows,
        groups=groups,
        minimum_valid=int(stage["minimum_valid_anchors_per_group"]),
    )
    elapsed = time.perf_counter() - started
    if elapsed > float(stage["resources"]["maximum_wall_minutes"]) * 60.0:
        raise RuntimeError("H7BS3 exceeded its frozen wall-time cap")
    result = {
        "experiment_id": config["experiment_id"],
        "code_commit": args.code_commit,
        "forbidden_payload_accessed": False,
        "anchor_ledgers": rows,
        "summary": summary,
        "runtime_seconds": elapsed,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if len(encoded.encode("utf-8")) > float(stage["resources"]["maximum_output_mib"]) * 2**20:
        raise RuntimeError("H7BS3 exceeded its frozen output-size cap")
    _write_json(args.output_directory / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--h7bs1-development-result", type=Path, required=True)
    parser.add_argument("--h7bs2-result", type=Path, required=True)
    parser.add_argument("--h7bs2-audit", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    payload = run(parse_args())
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, allow_nan=False))
