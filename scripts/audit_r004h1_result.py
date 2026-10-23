"""Mechanically audit the immutable R004H1 result and checkpoints without payloads."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.models.quotient_field import QuotientFieldPredictor
from aorta_drive.training.r004h import (
    StagePCheckpointCandidate,
    qualifies_stage_p_signal,
    select_stage_p_checkpoint,
)
from aorta_drive.vision.feature_cache import sha256_file


def _checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return cast(dict[str, Any], payload)


def _finite_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--latest-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not _finite_tree(result):
        raise ValueError("result must be a finite JSON object")
    if result.get("implementation_revision") != args.implementation_revision:
        raise ValueError("result implementation revision disagrees with the audit")
    if result.get("config_sha256") != sha256_file(args.config):
        raise ValueError("result config hash disagrees with the immutable config")
    if result.get("status") != "COMPLETED" or result.get("decision") != "PASS":
        raise ValueError("R004H1 did not record a completed PASS")
    if result.get("prohibited_payload_accessed") is not False:
        raise ValueError("R004H1 recorded prohibited payload access")
    if result.get("external_checkpoint_loaded") is not False:
        raise ValueError("R004H1 recorded an external checkpoint load")
    if result.get("fresh_random_initialization") is not True:
        raise ValueError("R004H1 did not record fresh random initialization")

    ledger = result.get("epoch_ledger")
    if not isinstance(ledger, list) or len(ledger) != 20:
        raise ValueError("R004H1 must contain exactly 20 completed epoch records")
    candidates: list[StagePCheckpointCandidate] = []
    for expected_epoch, row in enumerate(ledger, start=1):
        if not isinstance(row, dict) or row.get("epoch") != expected_epoch:
            raise ValueError("R004H1 epoch ledger is incomplete or out of order")
        means = row.get("selection_group_mean_edge_mae_m")
        maxima = row.get("selection_group_max_residual_m")
        if not isinstance(means, dict) or not isinstance(maxima, dict):
            raise ValueError("R004H1 epoch lacks complete selection maps")
        if len(means) != 8 or set(means) != set(maxima):
            raise ValueError("R004H1 epoch must contain the same eight selection groups")
        candidates.append(
            StagePCheckpointCandidate(
                epoch=expected_epoch,
                group_mean_edge_mae_m=tuple(float(means[key]) for key in sorted(means)),
                group_max_residual_m=tuple(float(maxima[key]) for key in sorted(maxima)),
            )
        )
    selected = select_stage_p_checkpoint(candidates)
    if selected.epoch != result.get("selected_epoch"):
        raise AssertionError("reported selected epoch disagrees with mechanical replay")
    if selected.macro_group_mean_edge_mae_m != result.get("selected_primary_m"):
        raise AssertionError("reported primary selection metric disagrees with replay")
    if selected.worst_two_group_max_mean_m != result.get("selected_secondary_m"):
        raise AssertionError("reported secondary selection metric disagrees with replay")
    gate = qualifies_stage_p_signal(
        learned_mean_m=selected.macro_group_mean_edge_mae_m,
        zero_mean_m=float(result["zero_field_selection_primary_m"]),
    )
    if gate is not True or result.get("selection_signal_gate_passed") is not True:
        raise AssertionError("reported Stage-P signal gate disagrees with replay")
    if result.get("completed_epochs") != 20 or result.get("optimizer_updates") != 13440:
        raise ValueError("R004H1 completion/update counts disagree with registration")

    best_hash = sha256_file(args.best_checkpoint)
    latest_hash = sha256_file(args.latest_checkpoint)
    if best_hash != result.get("best_checkpoint_sha256"):
        raise ValueError("best checkpoint hash disagrees with result")
    if latest_hash != result.get("latest_checkpoint_sha256"):
        raise ValueError("latest checkpoint hash disagrees with result")
    best = _checkpoint(args.best_checkpoint)
    latest = _checkpoint(args.latest_checkpoint)
    for checkpoint in (best, latest):
        if checkpoint.get("fresh_random_initialization") is not True:
            raise ValueError("checkpoint lacks the fresh-initialization declaration")
        if checkpoint.get("config_sha256") != result.get("config_sha256"):
            raise ValueError("checkpoint config hash disagrees with result")
        if checkpoint.get("role_manifest_sha256") != result.get("role_manifest_sha256"):
            raise ValueError("checkpoint role hash disagrees with result")
        model = QuotientFieldPredictor()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if best.get("completed_epoch") != selected.epoch:
        raise ValueError("best checkpoint epoch disagrees with selector replay")
    if latest.get("completed_epoch") != 20:
        raise ValueError("latest checkpoint does not contain the final epoch")
    expected_key = (
        selected.macro_group_mean_edge_mae_m,
        selected.worst_two_group_max_mean_m,
        selected.epoch,
    )
    if tuple(best.get("selection_key", ())) != expected_key:
        raise ValueError("best checkpoint key disagrees with selector replay")

    peak_gib = int(result["peak_cuda_bytes"]) / 1024**3
    elapsed_minutes = float(result["elapsed_seconds"]) / 60.0
    output_gib = int(result["output_bytes"]) / 1024**3
    resources = config["resources"]
    if peak_gib > float(resources["maximum_peak_gpu_gib"]):
        raise ValueError("recorded GPU allocation exceeds the registered cap")
    if elapsed_minutes > float(resources["maximum_wall_minutes"]):
        raise ValueError("recorded wall time exceeds the registered cap")
    if output_gib > float(resources["maximum_output_gib"]):
        raise ValueError("recorded output size exceeds the registered cap")

    audit = {
        "experiment_id": "R004H1-result-mechanical-audit-v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "raw_result_sha256": sha256_file(args.result),
        "best_checkpoint_sha256": best_hash,
        "latest_checkpoint_sha256": latest_hash,
        "replayed_selected_epoch": selected.epoch,
        "replayed_selected_primary_m": selected.macro_group_mean_edge_mae_m,
        "replayed_selected_secondary_m": selected.worst_two_group_max_mean_m,
        "replayed_zero_field_gate": gate,
        "strict_checkpoint_loads": 2,
        "payload_files_read": 0,
        "claim_boundary": "Mechanical ledger and checkpoint audit only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
