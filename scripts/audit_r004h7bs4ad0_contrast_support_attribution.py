"""Mechanical replay of the post-hoc H7BS4A-D0 attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4a_fit_projective_observability import _write_json
from run_r004h7bs4ad0_contrast_support_attribution import run


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    implementation = cast(dict[str, str], config["H7BS4AD0"]["implementation_sha256"])
    if sha256_file(Path(__file__).resolve()) != implementation["audit_source"]:
        raise ValueError("audit implementation hash mismatch")
    replay = run(
        argparse.Namespace(
            config=args.config,
            h7bs4a_result=args.h7bs4a_result,
            h7bs4a_config=args.h7bs4a_config,
            fit_result=args.fit_result,
            panel_run_root=args.panel_run_root,
            log_root=args.log_root,
            output_directory=args.replay_output_directory,
            code_commit=args.code_commit,
        )
    )
    original = cast(dict[str, Any], json.loads(args.result.read_text(encoding="utf-8")))
    ledgers_exact = original["anchor_ledgers"] == replay["anchor_ledgers"]
    summary_exact = original["summary"] == replay["summary"]
    output = {
        "experiment_id": config["experiment_id"],
        "result_sha256": sha256_file(args.result),
        "artifact_count": len(replay["anchor_ledgers"]),
        "all_anchor_ledgers_exact": ledgers_exact,
        "summary_exact": summary_exact,
        "failure_count": sum(not value for value in (ledgers_exact, summary_exact)),
        "mechanical_audit": "PASS" if ledgers_exact and summary_exact else "FAIL",
        "external_scientific_review": "REVIEW_UNAVAILABLE",
    }
    _write_json(args.audit_output, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--h7bs4a-result", type=Path, required=True)
    parser.add_argument("--h7bs4a-config", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--panel-run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--replay-output-directory", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), ensure_ascii=False, indent=2, allow_nan=False))
