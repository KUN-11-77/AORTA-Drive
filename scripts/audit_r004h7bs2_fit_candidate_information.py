"""Mechanically replay every R004H7-BS2 fit ledger from frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.evaluation.candidate_information import summarize_candidate_information
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs2_fit_candidate_information import _groups, _verify, compute_rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(Mapping[str, Any], config["H7BS2"])
    implementation = cast(Mapping[str, str], stage["implementation_sha256"])
    if sha256_file(Path(__file__).resolve()) != implementation["audit_source"]:
        raise ValueError("implementation hash mismatch: audit_source")
    _verify(
        config,
        fit_result=args.fit_result,
        group_role_manifest=args.group_role_manifest,
        repository=Path(__file__).resolve().parents[1],
    )
    result = cast(dict[str, Any], json.loads(args.result.read_text(encoding="utf-8")))
    fit_payload = cast(dict[str, Any], json.loads(args.fit_result.read_text(encoding="utf-8")))
    groups = _groups(args.group_role_manifest)
    replay_rows = compute_rows(
        config=config,
        fit_payload=fit_payload,
        panel_run_root=args.panel_run_root,
        groups=groups,
    )
    roi_rows = cast(list[dict[str, Any]], stage["roi_candidates"])
    replay_summary = summarize_candidate_information(
        replay_rows,
        group_names=groups,
        anchors_per_group=16,
        prefix_index=int(stage["prefix_index"]),
        primary_divergence_threshold_m=float(stage["primary_divergence_threshold_m"]),
        divergence_thresholds_m=tuple(
            float(value) for value in stage["divergence_thresholds_m"]
        ),
        roi_precedence=tuple(str(row["id"]) for row in roi_rows),
        roi_node_counts={
            str(row["id"]): int(row["columns"]) * int(row["rows"]) for row in roi_rows
        },
        minimum_valid_anchors_per_group=int(stage["minimum_valid_anchors_per_group"]),
        bootstrap_seed=int(stage["statistics"]["bootstrap_seed"]),
        bootstrap_replicates=int(stage["statistics"]["bootstrap_replicates"]),
    )
    stored_rows = cast(list[dict[str, Any]], result["anchor_ledgers"])
    row_match = stored_rows == replay_rows
    summary_match = result["summary"] == replay_summary
    audit = {
        "experiment_id": config["experiment_id"],
        "result_sha256": sha256_file(args.result),
        "artifact_count": len(replay_rows),
        "all_anchor_ledgers_exact": row_match,
        "summary_exact": summary_match,
        "failure_count": int(not row_match) + int(not summary_match),
        "mechanical_audit": "PASS" if row_match and summary_match else "FAIL",
        "external_scientific_review": "REVIEW_UNAVAILABLE",
    }
    _write_json(args.audit_output, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--panel-run-root", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(main(parse_args()), ensure_ascii=False, indent=2, allow_nan=False))
