"""Execute the frozen one-read R004H3 proxy-calibration rehearsal."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import numpy.typing as npt
import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.calibration.heterogeneous_proxy import (
    CONTROL_NAMES,
    calibrate_one_control_session_scores,
    canonical_grid_edges,
)
from aorta_drive.certification.heterogeneous import (
    deterministic_group_roles,
    outward_edge_budgets,
    outward_session_max_score,
)
from aorta_drive.models.heterogeneous_envelope import HeterogeneousEnvelopePredictor
from aorta_drive.models.quotient_field import QuotientFieldPredictor
from aorta_drive.training.r004ec import (
    JoinedAnchor,
    load_joined_anchors,
    load_training_anchor,
    sample_dense_features_torch,
)
from aorta_drive.training.r004h import (
    evidence_only_scale_control,
    pooled_scale_control,
    uniform_scale_control,
)
from aorta_drive.vision.feature_cache import sha256_file


def _checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return cast(dict[str, Any], payload)


def _read_roles(path: Path) -> dict[str, tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"recording_group", "role", "historical_exposure"}
    if not rows or set(rows[0]) != required:
        raise ValueError("role manifest columns disagree with the registered contract")
    roles = {
        role: tuple(row["recording_group"] for row in rows if row["role"] == role)
        for role in ("fit", "selection", "proxy_cal")
    }
    if tuple(len(roles[role]) for role in roles) != (42, 8, 16):
        raise ValueError("role counts disagree with the registered contract")
    expected_exposure = "previously_used_by_v8_3_training_or_method_development"
    if any(row["historical_exposure"] != expected_exposure for row in rows):
        raise ValueError("historical exposure ledger is incomplete")
    return roles


def _source_train_groups(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = tuple(row["recording_group"] for row in rows if row["split"] == "train")
    if len(groups) != 66 or len(set(groups)) != 66:
        raise ValueError("source manifest must contain 66 unique train groups")
    return groups


def _load_inputs(
    record: JoinedAnchor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, npt.NDArray[np.float64]]:
    loaded = load_training_anchor(record)
    dense = torch.from_numpy(loaded.dense_features).to(device)
    patch_xy = torch.from_numpy(loaded.patch_xy).to(device)
    mask = torch.from_numpy(loaded.mask).to(device)
    features = sample_dense_features_torch(dense, patch_xy, mask).unsqueeze(0)
    labels = np.asarray(loaded.label_m, dtype=np.float64).reshape(1, -1)
    return features, mask.unsqueeze(0), labels


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--recording-group-manifest", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--stage-p-checkpoint", type=Path, required=True)
    parser.add_argument("--scale-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/r004h3_proxy_rehearsal.yaml")
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_hashes = {
        "recording_group_manifest_sha256": sha256_file(args.recording_group_manifest),
        "role_manifest_sha256": sha256_file(args.role_manifest),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "label_manifest_sha256": sha256_file(args.label_manifest),
    }
    for name, actual in expected_hashes.items():
        if actual != config["inputs"][name]:
            raise ValueError(f"{name} disagrees with the frozen H3 config")
    stage_p_hash = sha256_file(args.stage_p_checkpoint)
    scale_hash = sha256_file(args.scale_checkpoint)
    if stage_p_hash != config["predecessor"]["stage_p_checkpoint_sha256"]:
        raise ValueError("Stage-P checkpoint hash disagrees with H3")
    if scale_hash != config["predecessor"]["selected_scale_checkpoint_sha256"]:
        raise ValueError("scale checkpoint hash disagrees with H3")
    stage_p_payload = _checkpoint(args.stage_p_checkpoint)
    scale_payload = _checkpoint(args.scale_checkpoint)
    if stage_p_payload.get("config_sha256") != config["predecessor"][
        "stage_p_config_sha256"
    ]:
        raise ValueError("Stage-P checkpoint provenance mismatch")
    if scale_payload.get("config_sha256") != config["predecessor"]["scale_config_sha256"]:
        raise ValueError("scale checkpoint provenance mismatch")

    roles = _read_roles(args.role_manifest)
    regenerated = deterministic_group_roles(
        _source_train_groups(args.recording_group_manifest),
        namespace="AORTA-R004EG-roles-v1",
        fit_count=42,
        selection_count=8,
        proxy_cal_count=16,
    )
    if roles["proxy_cal"] != regenerated.proxy_cal:
        raise ValueError("proxy role order is not a deterministic replay")
    records = load_joined_anchors(
        args.feature_manifest,
        args.label_manifest,
        feature_root=args.feature_root,
        label_root=args.label_root,
        allowed_splits=("train",),
    )
    by_group: dict[str, list[JoinedAnchor]] = defaultdict(list)
    for record in records:
        by_group[record.recording_group].append(record)
    proxy_records: list[JoinedAnchor] = []
    for group in roles["proxy_cal"]:
        ordered = sorted(by_group[group], key=lambda item: item.anchor_index)
        if [record.anchor_index for record in ordered] != list(range(16)):
            raise ValueError("every proxy group must contain anchor indices 0 through 15")
        proxy_records.extend(ordered)
    if len(proxy_records) != int(config["inputs"]["total_anchors"]):
        raise ValueError("proxy anchor count disagrees with H3 registration")

    torch.manual_seed(20261024)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("R004H3 requires the registered CUDA device")
    torch.cuda.init()  # type: ignore[no-untyped-call]
    potential = QuotientFieldPredictor().to(device)
    potential.load_state_dict(stage_p_payload["model_state_dict"], strict=True)
    model = HeterogeneousEnvelopePredictor(
        potential,
        scale_input_normalization="uncentered_global_rms",
    ).to(device)
    model.scale_head.load_state_dict(scale_payload["scale_head_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("H3 model must be entirely frozen")

    sources, targets = canonical_grid_edges(int(config["score"]["graph_grid_size"]))
    cached: dict[str, dict[str, list[npt.NDArray[np.float64]]]] = {
        group: {"potential": [], "labels": [], **{name: [] for name in CONTROL_NAMES}}
        for group in roles["proxy_cal"]
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        for index, record in enumerate(proxy_records, start=1):
            features, mask, labels = _load_inputs(record, device)
            output = model(features, mask)
            controls = {
                "learned": output.edge_scales_m,
                "pooled": pooled_scale_control(output.edge_scales_m),
                "uniform": uniform_scale_control(output.edge_scales_m),
                "evidence_only": evidence_only_scale_control(mask),
            }
            group_cache = cached[record.recording_group]
            group_cache["potential"].append(
                output.potential.detach().cpu().numpy().astype(np.float64).reshape(1, -1)
            )
            group_cache["labels"].append(labels)
            for name in CONTROL_NAMES:
                group_cache[name].append(
                    controls[name].detach().cpu().numpy().astype(np.float64)
                )
            if index % 32 == 0:
                print(json.dumps({"anchors_cached": index}), flush=True)

    scale_min = float(config["score"]["scale_min_m"])
    scale_max = float(config["score"]["scale_max_m"])
    scores: dict[str, list[float]] = {name: [] for name in CONTROL_NAMES}
    invalid_groups: dict[str, list[dict[str, str]]] = {
        name: [] for name in CONTROL_NAMES
    }
    score_ledger: list[dict[str, object]] = []
    for group in roles["proxy_cal"]:
        group_cache = cached[group]
        potentials = np.concatenate(group_cache["potential"], axis=0)
        labels = np.concatenate(group_cache["labels"], axis=0)
        row: dict[str, object] = {"recording_group": group, "anchor_count": 16}
        group_scores: dict[str, float | None] = {}
        for name in CONTROL_NAMES:
            scales = np.concatenate(group_cache[name], axis=0)
            try:
                value = outward_session_max_score(
                    potentials,
                    labels,
                    sources,
                    targets,
                    scales,
                    scale_min=scale_min,
                    scale_max=scale_max,
                )
                replay = outward_session_max_score(
                    potentials,
                    labels,
                    sources,
                    targets,
                    scales,
                    scale_min=scale_min,
                    scale_max=scale_max,
                )
                if value != replay:
                    raise AssertionError("cached session score did not replay exactly")
            except (ValueError, FloatingPointError, OverflowError, AssertionError) as error:
                invalid_groups[name].append(
                    {"recording_group": group, "reason": f"{type(error).__name__}: {error}"}
                )
                group_scores[name] = None
            else:
                scores[name].append(value)
                group_scores[name] = value
        row["session_max_scores"] = group_scores
        score_ledger.append(row)

    control_results: dict[str, object] = {}
    for name in CONTROL_NAMES:
        if invalid_groups[name]:
            control_results[name] = {
                "status": "INFINITE_INVALID",
                "threshold": None,
                "threshold_representation": "+infinity_due_to_invalid_group",
                "valid_group_count": len(scores[name]),
                "invalid_group_count": len(invalid_groups[name]),
                "invalid_groups": invalid_groups[name],
            }
            continue
        calibration = calibrate_one_control_session_scores(
            tuple(scores[name]),
            alpha=float(config["calibration"]["alpha"]),
            expected_sessions=int(config["calibration"]["independent_sessions"]),
            control_name=name,
        )
        if calibration.rank != int(config["calibration"]["order_statistic_rank"]):
            raise AssertionError("calibration rank disagrees with H3 registration")
        budget_min = math.inf
        budget_max = 0.0
        budget_values: list[npt.NDArray[np.float64]] = []
        for group in roles["proxy_cal"]:
            scales = np.concatenate(cached[group][name], axis=0)
            budgets = outward_edge_budgets(
                calibration.threshold,
                scales,
                scale_min=scale_min,
                scale_max=scale_max,
            )
            budget_min = min(budget_min, float(np.min(budgets)))
            budget_max = max(budget_max, float(np.max(budgets)))
            budget_values.append(budgets.reshape(-1))
        all_budgets = np.concatenate(budget_values)
        control_results[name] = {
            "status": "FINITE",
            "rank": calibration.rank,
            "threshold": calibration.threshold,
            "minimum_session_score": min(calibration.session_scores),
            "median_session_score": float(np.median(calibration.session_scores)),
            "maximum_session_score": max(calibration.session_scores),
            "threshold_equals_maximum": calibration.threshold
            == max(calibration.session_scores),
            "edge_budget_minimum_m": budget_min,
            "edge_budget_median_m": float(np.median(all_budgets)),
            "edge_budget_maximum_m": budget_max,
            "valid_group_count": len(calibration.session_scores),
            "invalid_group_count": 0,
        }
    decision = (
        "PASS" if all(not invalid_groups[name] for name in CONTROL_NAMES) else "STOP"
    )
    elapsed = time.perf_counter() - started
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    result: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": decision,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "stage_p_checkpoint_sha256": stage_p_hash,
        "scale_checkpoint_sha256": scale_hash,
        **expected_hashes,
        "proxy_group_count": len(roles["proxy_cal"]),
        "proxy_anchor_count": len(proxy_records),
        "payload_read_passes": 1,
        "cached_score_replay_passes": 2,
        "control_results": control_results,
        "score_ledger": score_ledger,
        "trainable_parameters": 0,
        "parameter_updates": 0,
        "checkpoint_written": False,
        "prohibited_payload_accessed": False,
        "h4_registration_authorized": decision == "PASS",
        "coverage_established": False,
        "official_calibration_access_authorized": False,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": peak_cuda_bytes,
        "elapsed_seconds": elapsed,
        "claim_boundary": (
            "Historically exposed proxy workflow rehearsal only. No coverage, task utility, "
            "Gate B, OOD robustness, official calibration, or novelty claim is established."
        ),
    }
    resources = config["resources"]
    if peak_cuda_bytes / 1024**3 > float(resources["maximum_peak_gpu_gib"]):
        raise RuntimeError("H3 peak GPU allocation exceeded the registered cap")
    if elapsed / 60.0 > float(resources["maximum_wall_minutes"]):
        raise RuntimeError("H3 wall time exceeded the registered cap")
    encoded_result = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    projected_output_gib = (
        _directory_bytes(args.output_directory) + len(encoded_result)
    ) / 1024**3
    if projected_output_gib > float(resources["maximum_output_gib"]):
        raise RuntimeError("H3 output size exceeded the registered cap")
    result_path = args.output_directory / "result.json"
    _write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
