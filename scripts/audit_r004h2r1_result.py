"""Mechanically audit the immutable R004H2R1 PASS without reading payloads."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.models.heterogeneous_envelope import BoundedEdgeScaleHead
from aorta_drive.training.r004h import (
    StageSCheckpointCandidate,
    qualifies_scale_caps,
    select_stage_s_checkpoint,
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
    parser.add_argument("--training-script", type=Path, required=True)
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
    if result.get("experiment_id") != config.get("experiment_id"):
        raise ValueError("experiment id mismatch")
    if result.get("implementation_revision") != args.implementation_revision:
        raise ValueError("implementation revision mismatch")
    if result.get("config_sha256") != sha256_file(args.config):
        raise ValueError("config hash mismatch")
    if result.get("script_sha256") != sha256_file(args.training_script):
        raise ValueError("training script hash mismatch")
    if result.get("status") != "COMPLETED" or result.get("decision") != "PASS":
        raise ValueError("R004H2R1 did not record the expected completed PASS")
    if result.get("scale_input_normalization") != "uncentered_global_rms":
        raise ValueError("R004H2R1 did not record the sole authorized repair")
    if config["model"].get("scale_input_normalization") != "uncentered_global_rms":
        raise ValueError("config does not contain the authorized normalization")
    if result.get("prohibited_payload_accessed") is not False:
        raise ValueError("R004H2R1 recorded prohibited payload access")
    if result.get("potential_parameter_updates") != 0:
        raise ValueError("R004H2R1 recorded a Stage-P update")

    ledger = result.get("epoch_ledger")
    if not isinstance(ledger, list) or len(ledger) != 10:
        raise ValueError("R004H2R1 must contain exactly 10 epoch records")
    candidates: list[StageSCheckpointCandidate] = []
    for expected_epoch, row in enumerate(ledger, start=1):
        if not isinstance(row, dict) or row.get("epoch") != expected_epoch:
            raise ValueError("epoch ledger is incomplete or out of order")
        losses = row.get("selection_group_mean_proper_loss")
        if not isinstance(losses, dict) or len(losses) != 8:
            raise ValueError("epoch lacks eight selection-group losses")
        candidates.append(
            StageSCheckpointCandidate(
                epoch=expected_epoch,
                group_mean_proper_loss=tuple(float(losses[key]) for key in sorted(losses)),
            )
        )
    selected = select_stage_s_checkpoint(candidates)
    if selected.epoch != result.get("selected_epoch"):
        raise AssertionError("selected epoch disagrees with replay")
    if selected.macro_group_mean_proper_loss != result.get("selected_primary"):
        raise AssertionError("selected proper loss disagrees with replay")
    uniform = float(result["uniform_selection_primary"])
    proper_gate = selected.macro_group_mean_proper_loss < uniform
    if proper_gate is not True or result.get("proper_loss_improvement_gate_passed") is not True:
        raise AssertionError("proper-loss PASS disagrees with replay")
    diagnostics = result.get("selected_scale_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("selected scale diagnostics are missing")
    cap_gate = qualifies_scale_caps(
        float(diagnostics["lower_cap_fraction"]),
        float(diagnostics["upper_cap_fraction"]),
        collapse_fraction=float(config["cap_collapse"]["collapse_fraction"]),
    )
    if cap_gate is not True or result.get("scale_cap_gate_passed") is not True:
        raise AssertionError("cap PASS disagrees with replay")
    if result.get("completed_epochs") != 10 or result.get("optimizer_updates") != 6720:
        raise ValueError("completion/update counts disagree with registration")
    if result.get("r004h3_registration_authorized") is not True:
        raise ValueError("R004H3 authorization disagrees with PASS")

    best_hash = sha256_file(args.best_checkpoint)
    latest_hash = sha256_file(args.latest_checkpoint)
    if best_hash != result.get("best_scale_checkpoint_sha256"):
        raise ValueError("best scale checkpoint hash mismatch")
    if latest_hash != result.get("latest_scale_checkpoint_sha256"):
        raise ValueError("latest scale checkpoint hash mismatch")
    for path, expected_epoch in (
        (args.best_checkpoint, selected.epoch),
        (args.latest_checkpoint, 10),
    ):
        checkpoint = _checkpoint(path)
        if checkpoint.get("completed_epoch") != expected_epoch:
            raise ValueError("scale checkpoint epoch mismatch")
        if checkpoint.get("config_sha256") != result.get("config_sha256"):
            raise ValueError("scale checkpoint config provenance mismatch")
        if checkpoint.get("potential_parameter_updates") != 0:
            raise ValueError("scale checkpoint recorded a Stage-P update")
        if checkpoint.get("stage_p_checkpoint_sha256") != result.get(
            "stage_p_checkpoint_sha256"
        ):
            raise ValueError("scale checkpoint Stage-P provenance mismatch")
        head = BoundedEdgeScaleHead()
        head.load_state_dict(checkpoint["scale_head_state_dict"], strict=True)

    resources = config["resources"]
    if int(result["peak_cuda_bytes"]) / 1024**3 > float(resources["maximum_peak_gpu_gib"]):
        raise ValueError("GPU allocation exceeded the registered cap")
    if float(result["elapsed_seconds"]) / 60 > float(resources["maximum_wall_minutes"]):
        raise ValueError("wall time exceeded the registered cap")
    if int(result["output_bytes"]) / 1024**3 > float(resources["maximum_output_gib"]):
        raise ValueError("output size exceeded the registered cap")

    audit = {
        "experiment_id": "R004H2R1-result-mechanical-audit-v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "raw_result_sha256": sha256_file(args.result),
        "best_scale_checkpoint_sha256": best_hash,
        "latest_scale_checkpoint_sha256": latest_hash,
        "replayed_selected_epoch": selected.epoch,
        "replayed_selected_primary": selected.macro_group_mean_proper_loss,
        "replayed_uniform_primary": uniform,
        "replayed_proper_loss_gate": proper_gate,
        "replayed_scale_cap_gate": cap_gate,
        "verified_scale_input_normalization": "uncentered_global_rms",
        "strict_scale_checkpoint_loads": 2,
        "payload_files_read": 0,
        "claim_boundary": "Mechanical H2R1 fit-gate audit only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
