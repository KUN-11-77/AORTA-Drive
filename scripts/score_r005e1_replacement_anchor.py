"""Score the frozen legal candidate domain for the R005 replacement anchor."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from aorta_drive.vision.feature_cache import sha256_file
from run_r004h7bs4b4_preference_oracle import _write_json
from run_r004h7bs4b4r1_eligible_preference_oracle import _process_anchor


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = cast(dict[str, Any], yaml.safe_load(args.b4_config.read_text("utf-8")))
    stage = copy.deepcopy(cast(dict[str, Any], config["H7BS4B4R1"]))
    manifest = cast(dict[str, Any], json.loads(args.metric_manifest.read_text("utf-8")))
    support = cast(dict[str, Any], json.loads(args.support_result.read_text("utf-8")))
    records = [
        row
        for row in manifest["records"]
        if row["recording_group"] == support["recording_group"]
        and int(row["anchor_index"]) == int(support["anchor_index"])
    ]
    if len(records) != 1 or records[0]["scene_token"] != support["scene_token"]:
        raise ValueError("replacement metric record does not match support qualification")
    if support.get("decision") != "REPLACEMENT_SUPPORT_PASS":
        raise ValueError("replacement support did not pass")
    stage["panel_sha256_by_token"] = {
        str(support["scene_token"]): str(support["panel_artifact_sha256"])
    }
    row = _process_anchor(
        (
            records[0],
            stage,
            str(args.metric_root),
            str(args.panel_root),
        )
    )
    result = {
        "experiment_id": "R005E1A-R1-replacement-anchor-one-frame-v1",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "b4_config_sha256": sha256_file(args.b4_config),
        "metric_manifest_sha256": sha256_file(args.metric_manifest),
        "support_result_sha256": sha256_file(args.support_result),
        "anchor_ledger": row,
        "development_calibration_test_navdream_target_accessed": False,
        "full_official_pdms_claimed": False,
        "claim_boundary": "Single source-fit replacement anchor one-frame labels only.",
    }
    _write_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b4-config", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--support-result", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    print(json.dumps(run(arguments), ensure_ascii=False, indent=2, allow_nan=False))
