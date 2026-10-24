"""Mechanically audit two H7BS4B4R1 full materializations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.evaluation.eligible_preference_oracle import (
    EligiblePreferenceOracleProtocol,
    summarize_eligible_preference_oracle,
)
from aorta_drive.vision.feature_cache import sha256_file


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(dict[str, Any], config["H7BS4B4R1"])
    if sha256_file(Path(__file__).resolve()) != stage["implementation_sha256"]["audit_source"]:
        raise ValueError("audit source hash mismatch")
    primary = cast(dict[str, Any], json.loads(args.primary_result.read_text(encoding="utf-8")))
    replay = cast(dict[str, Any], json.loads(args.replay_result.read_text(encoding="utf-8")))
    if (
        primary["experiment_id"] != config["experiment_id"]
        or replay["experiment_id"] != config["experiment_id"]
    ):
        raise ValueError("result experiment identity mismatch")

    role_rows = list(
        csv.DictReader(args.group_role_manifest.open("r", encoding="utf-8", newline=""))
    )
    groups = tuple(row["recording_group"] for row in role_rows if row["role"] == "fit")
    values = cast(Mapping[str, Any], stage["decision_protocol"])
    domain = cast(Mapping[str, Any], stage["candidate_domain"])
    protocol = EligiblePreferenceOracleProtocol(
        numerical_tie_tolerance=float(values["numerical_tie_tolerance"]),
        minimum_groups_with_nonzero_pair=int(values["minimum_groups_with_nonzero_pair"]),
        minimum_groups_with_strict_native_improvement=int(
            values["minimum_groups_with_strict_native_improvement"]
        ),
        bootstrap_seed=int(values["bootstrap_seed"]),
        bootstrap_resamples=int(values["bootstrap_resamples"]),
        expected_eligible_challenger_count=int(domain["expected_eligible_challengers"]),
        expected_zero_eligible_anchor_count=int(domain["expected_zero_eligible_anchors"]),
        expected_nonzero_eligible_anchor_count=int(
            domain["expected_nonzero_eligible_anchors"]
        ),
    )
    recomputed = summarize_eligible_preference_oracle(
        primary["anchor_ledgers"],
        group_names=groups,
        protocol=protocol,
        diagnostic_margins=tuple(float(value) for value in values["diagnostic_absolute_margins"]),
    )
    ledgers_exact = primary["anchor_ledgers"] == replay["anchor_ledgers"]
    primary_summary_exact = primary["summary"] == recomputed
    replay_summary_exact = replay["summary"] == recomputed
    root_keys = (
        "experiment_id",
        "label_estimand",
        "zero_eligible_anchor_policy",
        "full_official_pdms_claimed",
        "development_calibration_test_navdream_target_accessed",
        "decision",
        "claim_boundary",
    )
    root_exact = all(primary[key] == replay[key] for key in root_keys)
    passed = ledgers_exact and primary_summary_exact and replay_summary_exact and root_exact
    result = {
        "experiment_id": config["experiment_id"],
        "primary_result_sha256": sha256_file(args.primary_result),
        "replay_result_sha256": sha256_file(args.replay_result),
        "full_anchor_ledgers_exact": ledgers_exact,
        "primary_summary_recomputed_exact": primary_summary_exact,
        "replay_summary_recomputed_exact": replay_summary_exact,
        "root_contract_exact": root_exact,
        "failure_count": sum(
            not value
            for value in (
                ledgers_exact,
                primary_summary_exact,
                replay_summary_exact,
                root_exact,
            )
        ),
        "mechanical_audit": "PASS" if passed else "FAIL",
        "external_scientific_review": "REVIEW_UNAVAILABLE",
    }
    _write_json(args.audit_output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--primary-result", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, ensure_ascii=False, allow_nan=False))
