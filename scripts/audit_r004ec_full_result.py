"""Mechanically recompute the R004E-C2 ledger and checkpoint selection."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch

from aorta_drive.models.quotient_field import (
    DevelopmentCheckpoint,
    QuotientFieldPredictor,
    select_development_checkpoint,
)
from aorta_drive.vision.feature_cache import sha256_file


def _checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return cast(dict[str, Any], payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--latest-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload: dict[str, Any] = json.loads(args.result.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError("training result is not complete PASS")
    if payload.get("calibration_test_or_target_accessed") is not False:
        raise ValueError("training result does not preserve the data boundary")
    config_hash = sha256_file(args.config)
    if payload.get("config_sha256") != config_hash:
        raise ValueError("result config hash mismatch")
    ledger = payload.get("epoch_ledger")
    if not isinstance(ledger, list) or len(ledger) != 20:
        raise ValueError("result must contain exactly 20 completed epochs")

    candidates: list[DevelopmentCheckpoint] = []
    recomputed: list[dict[str, float | int]] = []
    failures: list[str] = []
    for expected_epoch, item in enumerate(ledger, start=1):
        if not isinstance(item, dict) or item.get("epoch") != expected_epoch:
            failures.append(f"epoch_{expected_epoch}_identity")
            continue
        groups = item.get("development_group_max_residuals_m")
        if not isinstance(groups, dict) or len(groups) != 24:
            failures.append(f"epoch_{expected_epoch}_group_count")
            continue
        values = tuple(float(groups[name]) for name in sorted(groups))
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            failures.append(f"epoch_{expected_epoch}_finite_residual")
            continue
        ordered = sorted(values, reverse=True)
        tail = sum(ordered[:5]) / 5
        mean = sum(ordered) / 24
        if not math.isclose(
            tail, float(item["development_worst5_mean_m"]), rel_tol=0.0, abs_tol=1e-12
        ):
            failures.append(f"epoch_{expected_epoch}_tail_recompute")
        if not math.isclose(
            mean, float(item["development_all24_mean_m"]), rel_tol=0.0, abs_tol=1e-12
        ):
            failures.append(f"epoch_{expected_epoch}_mean_recompute")
        if item.get("updates_completed") != expected_epoch * 1056:
            failures.append(f"epoch_{expected_epoch}_update_count")
        candidates.append(
            DevelopmentCheckpoint(epoch=expected_epoch, session_max_residuals_m=values)
        )
        recomputed.append(
            {"epoch": expected_epoch, "worst5_mean_m": tail, "all24_mean_m": mean}
        )

    selection = select_development_checkpoint(candidates)
    if payload.get("selected_epoch") != selection.epoch:
        failures.append("selected_epoch")
    if not math.isclose(
        float(payload["selected_development_worst5_mean_m"]),
        selection.tail_mean_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        failures.append("selected_worst5")
    if not math.isclose(
        float(payload["selected_development_all24_mean_m"]),
        selection.mean_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        failures.append("selected_all24")
    best_hash = sha256_file(args.best_checkpoint)
    latest_hash = sha256_file(args.latest_checkpoint)
    if payload.get("best_checkpoint_sha256") != best_hash:
        failures.append("best_checkpoint_hash")
    if payload.get("latest_checkpoint_sha256") != latest_hash:
        failures.append("latest_checkpoint_hash")

    best = _checkpoint(args.best_checkpoint)
    latest = _checkpoint(args.latest_checkpoint)
    for name, checkpoint in (("best", best), ("latest", latest)):
        for key in (
            "config_sha256",
            "feature_manifest_sha256",
            "label_manifest_sha256",
        ):
            if checkpoint.get(key) != payload.get(key):
                failures.append(f"{name}_{key}")
    if best.get("completed_epoch") != selection.epoch:
        failures.append("best_epoch")
    if latest.get("completed_epoch") != 20 or latest.get("update_count") != 21120:
        failures.append("latest_completion")
    expected_key = (selection.tail_mean_m, selection.mean_m, selection.epoch)
    if tuple(best.get("selection_key", ())) != expected_key:
        failures.append("best_selection_key")
    model = QuotientFieldPredictor()
    model.load_state_dict(best["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 4_000_225:
        failures.append("best_model_parameter_count")

    audit = {
        "experiment_id": "R004E-C2-independent-mechanical-audit",
        "status": "PASS" if not failures else "FAIL",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_sha256": sha256_file(args.result),
        "config_sha256": config_hash,
        "best_checkpoint_sha256": best_hash,
        "latest_checkpoint_sha256": latest_hash,
        "recomputed_epoch_ledger": recomputed,
        "recomputed_selected_epoch": selection.epoch,
        "recomputed_selected_worst5_mean_m": selection.tail_mean_m,
        "recomputed_selected_all24_mean_m": selection.mean_m,
        "failure_count": len(failures),
        "failures": failures,
        "calibration_test_or_target_accessed": False,
        "claim_boundary": "Mechanical ledger and checkpoint audit only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
