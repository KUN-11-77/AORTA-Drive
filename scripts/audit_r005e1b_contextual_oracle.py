"""Mechanical primary/replay audit for R005E1B."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.evaluation.contextual_oracle import (
    contextual_utility,
    summarize_contextual_oracle,
)
from aorta_drive.evaluation.contextual_target_qualification import filtered_one_stage
from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4b4_preference_oracle import _groups, _write_json
from run_r004h7bs4b4r1_eligible_preference_oracle import _protocol


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(dict[str, Any], config["R005E1B"])
    b4_config = cast(
        dict[str, Any], yaml.safe_load(args.b4_config.read_text(encoding="utf-8"))
    )
    b4_stage = cast(dict[str, Any], b4_config["H7BS4B4R1"])
    primary = cast(dict[str, Any], json.loads(args.primary.read_text(encoding="utf-8")))
    replay = cast(dict[str, Any], json.loads(args.replay.read_text(encoding="utf-8")))
    primary_rows = cast(list[dict[str, Any]], primary["anchor_ledgers"])
    replay_rows = cast(list[dict[str, Any]], replay["anchor_ledgers"])
    ledgers_exact = primary_rows == replay_rows
    if not ledgers_exact:
        raise ValueError("primary and replay contextual ledgers differ")
    if primary.get("development_calibration_test_navdream_target_accessed") is not False:
        raise ValueError("primary sealed-role flag is not false")
    if replay.get("development_calibration_test_navdream_target_accessed") is not False:
        raise ValueError("replay sealed-role flag is not false")
    if primary.get("full_official_pdms_claimed") is not False:
        raise ValueError("primary improperly claims full official PDMS")
    if primary.get("q_pi_ref_materialized") is not False:
        raise ValueError("undefined Q_pi_ref branch was materialized")

    candidate_recompute_count = 0
    for row in primary_rows:
        if row.get("b4r1_candidate_scores_exact") is not True:
            raise ValueError("B4R1 candidate score replay was not exact")
        for candidate in row["candidates"]:
            filtered = filtered_one_stage(candidate)
            contextual = contextual_utility(
                candidate, float(candidate["two_frame_extended_comfort"])
            )
            if filtered != float(candidate["filtered_one_frame_score"]):
                raise ValueError("filtered one-frame score mechanical mismatch")
            if contextual != float(candidate["contextual_utility"]):
                raise ValueError("contextual utility mechanical mismatch")
            candidate_recompute_count += 1

    groups = _groups(args.group_role_manifest)
    recomputed_summary = summarize_contextual_oracle(
        primary_rows,
        group_names=groups,
        protocol=_protocol(b4_stage),
        diagnostic_margins=tuple(
            float(value)
            for value in b4_stage["decision_protocol"]["diagnostic_absolute_margins"]
        ),
    )
    summary_exact = recomputed_summary == primary["summary"] == replay["summary"]
    if not summary_exact:
        raise ValueError("stored and mechanically recomputed summaries differ")
    decision_exact = (
        primary["decision"] == replay["decision"] == recomputed_summary["decision"]
    )
    if not decision_exact:
        raise ValueError("stored and mechanically recomputed decisions differ")
    return {
        "experiment_id": config["experiment_id"],
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(args.config),
        "primary_sha256": sha256_file(args.primary),
        "replay_sha256": sha256_file(args.replay),
        "primary_replay_anchor_ledgers_exact": ledgers_exact,
        "mechanically_recomputed_candidate_count": candidate_recompute_count,
        "summary_exact": summary_exact,
        "decision_exact": decision_exact,
        "decision": recomputed_summary["decision"],
        "q_pi_ref_materialized": False,
        "development_calibration_test_navdream_target_accessed": False,
        "claim_boundary": (
            "Mechanically audited source-fit contextual oracle only; not learned, "
            "held-out, policy-level, driving-safety, or robustness evidence."
        ),
        "frozen_input_sha256": dict(stage["frozen_input_sha256"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--b4-config", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    result = run(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
