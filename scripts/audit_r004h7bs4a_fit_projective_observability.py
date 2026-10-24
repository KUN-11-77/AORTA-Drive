"""Mechanical replay of the H7BS4A fit projective observability result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4a_fit_projective_observability import _write_json, run


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    implementation = cast(dict[str, str], config["H7BS4A"]["implementation_sha256"])
    repository = Path(__file__).resolve().parents[1]
    if sha256_file(Path(__file__).resolve()) != implementation["audit_source"]:
        raise ValueError("audit implementation hash mismatch")
    replay_arguments = argparse.Namespace(
        config=args.config,
        time_set=args.time_set,
        group_role_manifest=args.group_role_manifest,
        fit_result=args.fit_result,
        panel_run_root=args.panel_run_root,
        log_root=args.log_root,
        h7bs3_result=args.h7bs3_result,
        r004b_result=args.r004b_result,
        output_directory=args.replay_output_directory,
        code_commit=args.code_commit,
    )
    replay = run(replay_arguments)
    original = cast(dict[str, Any], json.loads(args.result.read_text(encoding="utf-8")))
    ledgers_exact = original["anchor_ledgers"] == replay["anchor_ledgers"]
    summary_exact = original["summary"] == replay["summary"]
    log_hashes_exact = original["source_log_sha256"] == replay["source_log_sha256"]
    output = {
        "experiment_id": config["experiment_id"],
        "result_sha256": sha256_file(args.result),
        "artifact_count": len(replay["anchor_ledgers"]),
        "all_anchor_ledgers_exact": ledgers_exact,
        "summary_exact": summary_exact,
        "source_log_hashes_exact": log_hashes_exact,
        "failure_count": sum(not value for value in (ledgers_exact, summary_exact, log_hashes_exact)),
        "mechanical_audit": (
            "PASS" if ledgers_exact and summary_exact and log_hashes_exact else "FAIL"
        ),
        "external_scientific_review": "REVIEW_UNAVAILABLE",
        "audit_source_repository": str(repository.name),
    }
    _write_json(args.audit_output, output)
    return output


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
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--replay-output-directory", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), ensure_ascii=False, indent=2, allow_nan=False))
