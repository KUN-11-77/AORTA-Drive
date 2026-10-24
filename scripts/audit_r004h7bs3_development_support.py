"""Mechanical replay of the R004H7-BS3 development support result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.geometry import RectangularGrid, VehicleRectangle
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs3_development_support import (
    _groups,
    _verify_inputs,
    _write_json,
    compute_rows,
    summarize,
)


def main(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(Mapping[str, Any], config["H7BS3"])
    implementation = cast(Mapping[str, str], stage["implementation_sha256"])
    repository = Path(__file__).resolve().parents[1]
    for key, path in {
        "candidate_information_source": repository
        / "src/aorta_drive/evaluation/candidate_information.py",
        "runner_source": repository / "scripts/run_r004h7bs3_development_support.py",
        "audit_source": Path(__file__).resolve(),
    }.items():
        if sha256_file(path) != implementation[key]:
            raise ValueError(f"implementation hash mismatch: {key}")
    _verify_inputs(
        stage,
        {
            "source_split_manifest": args.source_split_manifest,
            "development_panel_manifest": args.panel_manifest,
            "h7bs1_development_support_result": args.h7bs1_development_result,
            "h7bs2_fit_result": args.h7bs2_result,
            "h7bs2_mechanical_audit": args.h7bs2_audit,
        },
    )
    groups = _groups(args.source_split_manifest)
    manifest = cast(dict[str, Any], json.loads(args.panel_manifest.read_text(encoding="utf-8")))
    roi = cast(Mapping[str, Any], stage["roi"])
    vehicle = cast(Mapping[str, Any], stage["vehicle"])
    rows = compute_rows(
        panel_manifest=manifest,
        panel_root=args.panel_root,
        groups=groups,
        grid=RectangularGrid(
            x_min=float(roi["x_min_m"]),
            y_min=float(roi["y_min_m"]),
            columns=int(roi["columns"]),
            rows=int(roi["rows"]),
            resolution=float(roi["resolution_m"]),
        ),
        body=VehicleRectangle(
            rear_overhang=float(vehicle["rear_overhang_m"]),
            front_extent=float(vehicle["front_extent_m"]),
            half_width=float(vehicle["half_width_m"]),
        ),
    )
    replay_summary = summarize(
        rows,
        groups=groups,
        minimum_valid=int(stage["minimum_valid_anchors_per_group"]),
    )
    result = cast(dict[str, Any], json.loads(args.result.read_text(encoding="utf-8")))
    exact_rows = result["anchor_ledgers"] == rows
    exact_summary = result["summary"] == replay_summary
    audit = {
        "experiment_id": config["experiment_id"],
        "result_sha256": sha256_file(args.result),
        "artifact_count": len(rows),
        "all_anchor_ledgers_exact": exact_rows,
        "summary_exact": exact_summary,
        "failure_count": int(not exact_rows) + int(not exact_summary),
        "mechanical_audit": "PASS" if exact_rows and exact_summary else "FAIL",
        "external_scientific_review": "REVIEW_UNAVAILABLE",
    }
    _write_json(args.audit_output, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--h7bs1-development-result", type=Path, required=True)
    parser.add_argument("--h7bs2-result", type=Path, required=True)
    parser.add_argument("--h7bs2-audit", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(main(parse_args()), ensure_ascii=False, indent=2, allow_nan=False))
