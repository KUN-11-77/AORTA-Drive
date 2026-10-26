"""Mechanically audit primary/replay R005E3C read-only diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

from aorta_drive.vision.feature_cache import sha256_file


IGNORED_REPLAY_FIELDS = {"completed_at_utc"}


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text("utf-8")))


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return False


def _replay_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items() if key not in IGNORED_REPLAY_FIELDS
    }


def audit(
    primary: dict[str, Any],
    replay: dict[str, Any],
    *,
    config_sha256: str,
    runner_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    gates = {
        "schema_exact": primary.get("schema")
        == replay.get("schema")
        == "aorta.r005e3b.gradient_mass_diagnostic.v1",
        "status_complete": primary.get("status") == replay.get("status") == "COMPLETED",
        "primary_replay_exact_except_timestamp": _replay_payload(primary)
        == _replay_payload(replay),
        "finite_values": _finite(primary) and _finite(replay),
        "weights_never_updated": primary.get("weights_updated") is False
        and replay.get("weights_updated") is False,
        "model_state_unchanged": primary.get("model_state_unchanged") is True
        and replay.get("model_state_unchanged") is True,
        "sealed_roles_untouched": primary.get(
            "development_calibration_test_navdream_target_accessed"
        )
        is False
        and replay.get("development_calibration_test_navdream_target_accessed")
        is False,
        "config_hash_exact": primary.get("input_sha256", {}).get("config")
        == replay.get("input_sha256", {}).get("config")
        == config_sha256,
        "runner_hash_exact": primary.get("runner_sha256")
        == replay.get("runner_sha256")
        == runner_sha256,
        "code_commit_exact": primary.get("code_commit")
        == replay.get("code_commit")
        == expected_commit,
        "same_physical_gpu": primary.get("physical_gpu") == replay.get("physical_gpu"),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        "schema": "aorta.r005e3c.gradient_mass_audit.v1",
        "status": status,
        "decision": (
            "MECHANICAL_PASS__DESCRIPTIVE_DIAGNOSTIC_ONLY"
            if status == "PASS"
            else "INVALID_DIAGNOSTIC"
        ),
        "mechanical_gates": gates,
        "scientific_result_gate": "NOT_DEFINED_BY_PROTOCOL",
        "claim_boundary": (
            "Read-only fold-0 outer-train optimization diagnosis only; no method "
            "PASS, training unlock, OOF, visual binding, architecture, appearance, "
            "behavior, or safety claim."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = audit(
        _read_json(args.primary),
        _read_json(args.replay),
        config_sha256=sha256_file(args.config),
        runner_sha256=sha256_file(args.runner),
        expected_commit=args.expected_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 2)
