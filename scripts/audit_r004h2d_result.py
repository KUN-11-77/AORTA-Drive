"""Mechanically audit an R004H2D result without reading any experiment payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, cast

import yaml  # type: ignore[import-untyped]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--diagnostic-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()

    config = _mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")), "config")
    result = _mapping(json.loads(args.result.read_text(encoding="utf-8")), "result")
    if result.get("experiment_id") != config.get("experiment_id"):
        raise ValueError("result experiment id disagrees with config")
    if result.get("status") != "COMPLETED":
        raise ValueError("diagnostic result is not complete")
    if result.get("implementation_revision") != args.implementation_revision:
        raise ValueError("implementation revision mismatch")
    if result.get("config_sha256") != _sha256(args.config):
        raise ValueError("config hash mismatch")
    if result.get("script_sha256") != _sha256(args.diagnostic_script):
        raise ValueError("diagnostic script hash mismatch")
    if result.get("stage_p_checkpoint_sha256") != config["predecessor"][
        "selected_stage_p_checkpoint_sha256"
    ]:
        raise ValueError("Stage-P checkpoint provenance mismatch")
    for name in (
        "recording_group_manifest_sha256",
        "role_manifest_sha256",
        "feature_manifest_sha256",
        "label_manifest_sha256",
    ):
        if result.get(name) != config["inputs"][name]:
            raise ValueError(f"{name} mismatch")
    if result.get("sample_size") != 8 or result.get("sample_roles") != {
        "fit": 4,
        "selection": 4,
    }:
        raise ValueError("sample counts disagree with registration")
    ledger = result.get("anchor_ledger")
    if not isinstance(ledger, list) or len(ledger) != 8:
        raise ValueError("anchor ledger must contain exactly eight rows")
    for role, offset in (("fit", 0), ("selection", 4)):
        rows = [_mapping(row, "anchor row") for row in ledger[offset : offset + 4]]
        if any(row.get("role") != role for row in rows):
            raise ValueError(f"{role} ledger order mismatch")
        if len({row.get("recording_group") for row in rows}) != 1:
            raise ValueError(f"{role} rows must use one frozen group")
        if [row.get("anchor_index") for row in rows] != [0, 1, 2, 3]:
            raise ValueError(f"{role} rows must use the four lowest anchor indices")

    diagnosis = config["diagnosis"]
    first_step = config["first_step_counterfactual"]
    recomputed_votes = {
        "activation_scale": 0,
        "parameterization": 0,
        "learning_rate": 0,
    }
    raw_gradients: list[float] = []
    normalized_gradients: list[float] = []
    safe_rates: list[float] = []
    for item in ledger:
        row = _mapping(item, "anchor row")
        raw_gradient = _mapping(row.get("raw_gradient"), "raw gradient")
        normalized_gradient = _mapping(
            row.get("normalized_gradient"), "normalized gradient"
        )
        raw_step = _mapping(row.get("raw_first_step"), "raw first step")
        normalized_step = _mapping(
            row.get("normalized_first_step"), "normalized first step"
        )
        raw_norm = _finite(raw_gradient.get("full_norm"), "raw gradient norm")
        normalized_norm = _finite(
            normalized_gradient.get("full_norm"), "normalized gradient norm"
        )
        raw_gradients.append(raw_norm)
        normalized_gradients.append(normalized_norm)
        safe_rate = _finite(raw_step.get("safe_learning_rate"), "safe learning rate")
        safe_rates.append(safe_rate)
        raw_cap = _finite(raw_step.get("lower_cap_fraction"), "raw lower cap") + _finite(
            raw_step.get("upper_cap_fraction"), "raw upper cap"
        )
        normalized_cap = _finite(
            normalized_step.get("lower_cap_fraction"), "normalized lower cap"
        ) + _finite(normalized_step.get("upper_cap_fraction"), "normalized upper cap")
        raw_jump = _finite(
            raw_step.get("maximum_absolute_logit_change"), "raw logit jump"
        )
        normalized_jump = _finite(
            normalized_step.get("maximum_absolute_logit_change"),
            "normalized logit jump",
        )
        ratio = raw_jump / max(normalized_jump, 1.1754943508222875e-38)
        activation_vote = (
            raw_cap
            >= diagnosis["activation_scale_vote_requires"][
                "raw_first_step_cap_fraction_at_least"
            ]
            and normalized_cap == 0.0
            and ratio
            >= diagnosis["activation_scale_vote_requires"][
                "raw_to_normalized_max_logit_change_ratio_at_least"
            ]
            and normalized_jump
            <= diagnosis["activation_scale_vote_requires"][
                "normalized_maximum_absolute_logit_change_at_most"
            ]
        )
        derivative_ratio = _finite(
            row.get("bounded_to_log_scale_initial_derivative_ratio"),
            "parameterization derivative ratio",
        )
        parameterization_vote = (
            derivative_ratio
            >= diagnosis["parameterization_vote_requires"][
                "bounded_to_log_scale_initial_derivative_ratio_at_least"
            ]
        )
        learning_rate_vote = (
            not activation_vote
            and raw_cap
            >= diagnosis["learning_rate_vote_requires"][
                "raw_first_step_cap_fraction_at_least"
            ]
            and safe_rate
            <= first_step["learning_rate"]
            * diagnosis["learning_rate_vote_requires"][
                "derived_safe_learning_rate_at_most_fraction_of_registered"
            ]
        )
        expected_votes = {
            "activation_scale": activation_vote,
            "parameterization": parameterization_vote,
            "learning_rate": learning_rate_vote,
        }
        if row.get("votes") != expected_votes:
            raise ValueError("stored cause votes do not replay")
        for name, vote in expected_votes.items():
            recomputed_votes[name] += int(vote)
        if _finite(row.get("autograd_maximum_absolute_error"), "autograd error") > 2.0e-3:
            raise ValueError("analytic/autograd gradient agreement is outside tolerance")

    if result.get("vote_counts") != recomputed_votes:
        raise ValueError("aggregate vote counts do not replay")
    support = int(diagnosis["minimum_supporting_anchors"])
    if recomputed_votes["parameterization"] >= support:
        conclusion = "PARAMETERIZATION_DOMINANT"
    elif recomputed_votes["activation_scale"] >= support:
        conclusion = "TRUNK_ACTIVATION_SCALE_DOMINANT"
    elif recomputed_votes["learning_rate"] >= support:
        conclusion = "REGISTERED_LEARNING_RATE_DOMINANT"
    else:
        conclusion = "UNRESOLVED"
    if result.get("conclusion") != conclusion:
        raise ValueError("diagnostic conclusion does not replay")
    summary = _mapping(result.get("summary"), "summary")
    expected_summary = {
        "median_raw_full_gradient_norm": median(raw_gradients),
        "median_normalized_full_gradient_norm": median(normalized_gradients),
        "median_raw_to_normalized_gradient_norm_ratio": median(
            raw / normalized
            for raw, normalized in zip(raw_gradients, normalized_gradients, strict=True)
        ),
        "median_raw_safe_learning_rate": median(safe_rates),
    }
    if summary != expected_summary:
        raise ValueError("summary statistics do not replay exactly")
    for field in (
        "optimizer_constructed",
        "parameter_mutation",
        "checkpoint_written",
        "prohibited_payload_accessed",
    ):
        if result.get(field) is not False:
            raise ValueError(f"{field} must be false")
    if result.get("h2_revision_registration_authorized") is not (
        conclusion != "UNRESOLVED"
    ):
        raise ValueError("H2 revision authorization disagrees with conclusion")
    peak_gib = _finite(result.get("peak_cuda_bytes"), "peak CUDA bytes") / 1024**3
    elapsed = _finite(result.get("elapsed_seconds"), "elapsed seconds")
    if peak_gib > config["resources"]["maximum_peak_gpu_gib"]:
        raise ValueError("registered GPU cap was exceeded")
    if elapsed > config["resources"]["maximum_wall_minutes"] * 60.0:
        raise ValueError("registered wall-time cap was exceeded")

    audit = {
        "experiment_id": config["experiment_id"],
        "status": "PASS",
        "result_sha256": _sha256(args.result),
        "config_sha256": _sha256(args.config),
        "diagnostic_script_sha256": _sha256(args.diagnostic_script),
        "implementation_revision": args.implementation_revision,
        "replayed_vote_counts": recomputed_votes,
        "replayed_conclusion": conclusion,
        "payload_files_read": 0,
        "checks": {
            "immutable_hashes": True,
            "fixed_sample": True,
            "vote_replay": True,
            "summary_replay": True,
            "analytic_autograd_agreement": True,
            "no_optimizer_or_mutation": True,
            "no_prohibited_payload": True,
            "resource_caps": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
