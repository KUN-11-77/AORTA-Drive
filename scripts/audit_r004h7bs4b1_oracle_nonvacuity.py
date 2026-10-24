"""Mechanically replay the frozen H7BS4B1 oracle non-vacuity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4a_fit_projective_observability import _write_json
from run_r004h7bs4b1_oracle_nonvacuity import run


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(Mapping[str, Any], config["H7BS4B1"])
    expected_audit_hash = str(stage["implementation_sha256"]["audit_source"])
    if sha256_file(Path(__file__).resolve()) != expected_audit_hash:
        raise ValueError("audit implementation hash mismatch")
    original = cast(dict[str, Any], json.loads(args.result.read_text(encoding="utf-8")))
    replay_args = argparse.Namespace(
        config=args.config,
        h7bs4a_result=args.h7bs4a_result,
        h7bs4b0b_result=args.h7bs4b0b_result,
        fit_result=args.fit_result,
        teacher_manifest=args.teacher_manifest,
        teacher_root=args.teacher_root,
        panel_run_root=args.panel_run_root,
        log_root=args.log_root,
        output_directory=args.replay_output_directory,
        code_commit=args.code_commit,
    )
    replay = run(replay_args)
    ledgers_exact = original.get("anchor_ledgers") == replay.get("anchor_ledgers")
    summary_exact = original.get("summary") == replay.get("summary")
    root_contract_exact = all(
        original.get(key) == replay.get(key)
        for key in (
            "experiment_id",
            "code_commit",
            "forbidden_payload_accessed",
            "teacher_oracle_not_deployable",
            "parent_h7bs4a_decision_preserved",
        )
    )
    failures = sum(not value for value in (ledgers_exact, summary_exact, root_contract_exact))
    payload = {
        "experiment_id": config["experiment_id"],
        "result_sha256": sha256_file(args.result),
        "artifact_count": len(cast(list[Any], replay["anchor_ledgers"])),
        "all_anchor_ledgers_exact": ledgers_exact,
        "summary_exact": summary_exact,
        "root_contract_exact": root_contract_exact,
        "failure_count": failures,
        "mechanical_audit": "PASS" if failures == 0 else "FAIL",
        "external_scientific_review": "REVIEW_UNAVAILABLE",
    }
    _write_json(args.audit_output, payload)
    if failures:
        raise RuntimeError("H7BS4B1 mechanical replay failed")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--h7bs4a-result", type=Path, required=True)
    parser.add_argument("--h7bs4b0b-result", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--panel-run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--replay-output-directory", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(
        json.dumps(
            audit(parse_args()),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
