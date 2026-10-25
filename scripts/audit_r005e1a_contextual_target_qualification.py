"""Run the outcome-free R005E1A contextual-target qualification audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.evaluation.contextual_target_qualification import (
    qualify_contextual_target,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.config.read_text(encoding="utf-8")))
    stage = cast(dict[str, Any], config["R005E1A"])
    frozen_inputs = cast(Mapping[str, str], stage["frozen_input_sha256"])
    input_paths = {
        "b4r1_primary_result": args.b4_result,
        "temporal_context_roster": args.context_roster,
        "metric_manifest": args.metric_manifest,
        "group_role_manifest": args.group_role_manifest,
        "h7bs1_fit_result": args.fit_result,
    }
    for name, path in input_paths.items():
        if _sha256(path) != frozen_inputs[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")

    official = cast(Mapping[str, str], stage["official_source_sha256"])
    observed_sources: dict[str, str] = {}
    for relative, expected in official.items():
        path = args.navsim_root / relative
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"official source hash mismatch: {relative}")
        observed_sources[relative] = observed

    mapping_payloads = [
        cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        for path in args.two_stage_mapping
    ]
    qualification = qualify_contextual_target(
        _json(args.b4_result),
        _json(args.context_roster),
        mapping_payloads,
        previous_metric_root=args.previous_metric_root,
    )
    return {
        "experiment_id": config["experiment_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_role": "outcome_free_source_history_candidate_and_mapping_qualification",
        "config_sha256": _sha256(args.config),
        "frozen_input_sha256": {name: _sha256(path) for name, path in input_paths.items()},
        "official_source_sha256": observed_sources,
        "two_stage_mapping_sha256": {
            str(path): _sha256(path) for path in args.two_stage_mapping
        },
        **qualification,
        "claim_boundary": {
            "permitted": [
                "source_fit_contextual_target_definition_qualification",
                "previous_metric_cache_readiness",
                "official_two_stage_mapping_coverage",
            ],
            "prohibited": [
                "contextual_oracle_headroom",
                "RGB_recoverability",
                "network_training_or_superiority",
                "NavDream_or_driving_gain",
                "safety_claim",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--b4-result", type=Path, required=True)
    parser.add_argument("--context-roster", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--group-role-manifest", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--navsim-root", type=Path, required=True)
    parser.add_argument("--two-stage-mapping", type=Path, action="append", required=True)
    parser.add_argument("--previous-metric-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

