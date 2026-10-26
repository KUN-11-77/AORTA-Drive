"""Read-only post-hoc anatomy for the failed R005E3D qualification.

This script does not modify checkpoints or select a deployable threshold. Any
threshold frontier is an in-sample diagnostic upper bound and must be confirmed
with nested group-held-out selection before it can support a method claim.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt


TIE_TOLERANCE = 1.0e-12


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _safe_mean(value: npt.NDArray[np.float64]) -> float | None:
    return float(np.mean(value)) if value.size else None


def _safe_std(value: npt.NDArray[np.float64]) -> float | None:
    return float(np.std(value)) if value.size else None


def _first_divergence_time(
    distance: npt.NDArray[np.float64], threshold_m: float, sample_interval_s: float
) -> npt.NDArray[np.float64]:
    crossed = distance > threshold_m
    first = np.argmax(crossed, axis=-1)
    return np.where(
        np.any(crossed, axis=-1), first * sample_interval_s, np.nan
    ).astype(np.float64)


def _geometry_features(
    candidate_poses: npt.NDArray[np.float32], sample_interval_s: float = 0.1
) -> dict[str, npt.NDArray[np.float64]]:
    if candidate_poses.ndim != 4 or candidate_poses.shape[-1] != 3:
        raise ValueError("candidate_poses must have shape [anchor,candidate,time,3]")
    native = candidate_poses[:, :1]
    position_delta = np.asarray(
        np.linalg.norm(candidate_poses[..., :2] - native[..., :2], axis=-1),
        dtype=np.float64,
    )
    step = np.diff(candidate_poses[..., :2], axis=2)
    path_length = np.asarray(np.sum(np.linalg.norm(step, axis=-1), axis=-1), dtype=np.float64)
    yaw_delta = np.asarray(
        np.arctan2(
            np.sin(candidate_poses[..., 2] - native[..., 2]),
            np.cos(candidate_poses[..., 2] - native[..., 2]),
        ),
        dtype=np.float64,
    )
    return {
        "mean_position_delta_m": np.mean(position_delta, axis=-1),
        "final_position_delta_m": position_delta[..., -1],
        "max_position_delta_m": np.max(position_delta, axis=-1),
        "final_abs_yaw_delta_rad": np.abs(yaw_delta[..., -1]),
        "max_abs_yaw_delta_rad": np.max(np.abs(yaw_delta), axis=-1),
        "path_length_m": path_length,
        "path_length_delta_m": path_length - path_length[:, :1],
        "first_divergence_time_0p25_s": _first_divergence_time(
            position_delta, 0.25, sample_interval_s
        ),
        "first_divergence_time_0p5_s": _first_divergence_time(
            position_delta, 0.5, sample_interval_s
        ),
        "first_divergence_time_1p0_s": _first_divergence_time(
            position_delta, 1.0, sample_interval_s
        ),
    }


def _contextual_components(
    ledgers: list[dict[str, Any]],
    metadata: dict[int, dict[str, Any]],
    candidate_count: int,
) -> dict[str, npt.NDArray[np.float64]]:
    ledger_by_key = {
        (str(row["recording_group"]), int(row["anchor_index"])): row
        for row in ledgers
    }
    shape = (len(metadata), candidate_count)
    filtered = np.full(shape, np.nan, dtype=np.float64)
    comfort = np.full(shape, np.nan, dtype=np.float64)
    hard_gate = np.full(shape, np.nan, dtype=np.float64)
    contextual = np.full(shape, np.nan, dtype=np.float64)
    for bundle_index, meta in metadata.items():
        key = (str(meta["recording_group"]), int(meta["anchor_index"]))
        ledger = ledger_by_key[key]
        for candidate in cast(list[dict[str, Any]], ledger["candidates"]):
            index = int(candidate["registered_index"])
            hard = cast(dict[str, Any], candidate["hard_metrics"])
            gate = math.prod(float(value) for value in hard.values())
            filtered[bundle_index, index] = float(candidate["filtered_one_frame_score"])
            comfort[bundle_index, index] = float(
                candidate["two_frame_extended_comfort"]
            )
            hard_gate[bundle_index, index] = gate
            contextual[bundle_index, index] = float(candidate["contextual_utility"])
    return {
        "filtered": filtered,
        "comfort": comfort,
        "hard_gate": hard_gate,
        "contextual": contextual,
    }


def _finite_summary(value: npt.NDArray[np.float64]) -> dict[str, float | int | None]:
    finite = value[np.isfinite(value)]
    return {
        "count": int(value.size),
        "finite_count": int(finite.size),
        "mean": _safe_mean(finite),
        "std": _safe_std(finite),
        "minimum": float(np.min(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def _finite_scalar(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _selection(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
) -> dict[str, npt.NDArray[Any]]:
    row = np.arange(prediction.shape[0])
    selected = np.argmax(np.where(mask, prediction, -np.inf), axis=1)
    gain = np.asarray(target[row, selected] - target[:, 0], dtype=np.float64)
    selected_score = np.asarray(prediction[row, selected], dtype=np.float64)
    return {"row": row, "selected": selected, "gain": gain, "score": selected_score}


def _threshold_frontier(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
) -> list[dict[str, float | int | str]]:
    row = np.arange(prediction.shape[0])
    challenger_mask = mask.copy()
    challenger_mask[:, 0] = False
    challenger_score = np.where(challenger_mask, prediction, -np.inf)
    challenger_index = np.argmax(challenger_score, axis=1)
    top_score = np.asarray(challenger_score[row, challenger_index], dtype=np.float64)
    finite_nonnegative = np.unique(top_score[np.isfinite(top_score) & (top_score >= 0.0)])
    thresholds = np.concatenate(
        [np.asarray([0.0], dtype=np.float64), finite_nonnegative, np.asarray([math.inf])]
    )
    rows: list[dict[str, float | int | str]] = []
    for threshold in np.unique(thresholds):
        intervene = np.isfinite(top_score) & (top_score > threshold)
        selected = np.where(intervene, challenger_index, 0)
        gain = np.asarray(target[row, selected] - target[:, 0], dtype=np.float64)
        rows.append(
            {
                "threshold": "inf" if math.isinf(float(threshold)) else float(threshold),
                "mean_gain_over_native": float(np.mean(gain)),
                "nonnative_choice_count": int(np.sum(intervene)),
                "beneficial_count": int(np.sum(intervene & (gain > TIE_TOLERANCE))),
                "harmful_count": int(np.sum(intervene & (gain < -TIE_TOLERANCE))),
            }
        )
    return rows


def analyze(result_root: Path) -> dict[str, Any]:
    primary_result = cast(
        dict[str, Any],
        json.loads((result_root / "primary" / "result.json").read_text("utf-8")),
    )
    audit = cast(
        dict[str, Any], json.loads((result_root / "audit_local.json").read_text("utf-8"))
    )
    if (
        primary_result.get("schema") != "aorta.r005e3d.proper_mean_qualification.v1"
        or audit.get("status") != "PASS"
        or audit.get("decision") != "FAIL_FROZEN_QUALIFICATION_GATE"
    ):
        raise ValueError("R005E3D completed mechanical-fail-science contract is absent")
    payload = np.load(result_root / "primary" / "predictions.npz")
    prediction = np.asarray(payload["correct_prediction"], dtype=np.float32)
    target = np.asarray(payload["target"], dtype=np.float32)
    mask = np.asarray(payload["candidate_mask"], dtype=np.bool_)
    bundle_index = np.asarray(payload["bundle_index"], dtype=np.int64)
    manifest = cast(
        dict[str, Any],
        json.loads(
            (result_root / "audit_inputs" / "bundle" / "manifest.json").read_text(
                "utf-8"
            )
        ),
    )
    metadata = {
        int(row["bundle_index"]): row
        for row in cast(list[dict[str, Any]], manifest["anchors"])
    }
    candidate_poses = np.asarray(
        np.load(result_root / "audit_inputs" / "bundle" / "candidate_poses.npy"),
        dtype=np.float32,
    )
    if (
        candidate_poses.shape[1] != prediction.shape[1]
        or int(np.max(bundle_index)) >= candidate_poses.shape[0]
    ):
        raise ValueError("candidate-pose panel shape does not match predictions")
    geometry = _geometry_features(candidate_poses)
    e1b_result = cast(
        dict[str, Any],
        json.loads(
            (
                result_root / "audit_inputs" / "r005e1b_primary_result.json"
            ).read_text("utf-8")
        ),
    )
    components = _contextual_components(
        cast(list[dict[str, Any]], e1b_result["anchor_ledgers"]),
        metadata,
        prediction.shape[1],
    )
    component_contextual = components["contextual"][bundle_index]
    component_relation = component_contextual - component_contextual[:, :1]
    if not np.allclose(component_relation[mask], target[mask], atol=1.0e-6, rtol=0.0):
        raise ValueError("E1B component ledger does not reproduce the target relation")
    filtered_relation = (
        components["filtered"][bundle_index]
        - components["filtered"][bundle_index, :1]
    )
    temporal_gate_value = components["hard_gate"] * components["comfort"]
    temporal_gate_relation = (
        temporal_gate_value[bundle_index]
        - temporal_gate_value[bundle_index, :1]
    )
    reconstructed_relation = 0.875 * filtered_relation + 0.125 * temporal_gate_relation
    if not np.allclose(reconstructed_relation[mask], target[mask], atol=1.0e-6, rtol=0.0):
        raise ValueError("7/8+1/8 contextual decomposition does not reproduce target")
    selected_payload = _selection(prediction, target, mask)
    selected = cast(npt.NDArray[np.int64], selected_payload["selected"])
    gain = cast(npt.NDArray[np.float64], selected_payload["gain"])
    selected_score = cast(npt.NDArray[np.float64], selected_payload["score"])
    nonnative = selected != 0
    beneficial = nonnative & (gain > TIE_TOLERANCE)
    harmful = nonnative & (gain < -TIE_TOLERANCE)
    neutral = nonnative & ~(beneficial | harmful)

    challenger = mask.copy()
    challenger[:, 0] = False
    candidate_score = np.asarray(prediction[challenger], dtype=np.float64)
    candidate_target = np.asarray(target[challenger], dtype=np.float64)
    candidate_filtered = np.asarray(filtered_relation[challenger], dtype=np.float64)
    candidate_temporal = np.asarray(
        temporal_gate_relation[challenger], dtype=np.float64
    )
    predicted_positive = candidate_score > 0.0
    target_positive = candidate_target > TIE_TOLERANCE

    temporal_buckets: dict[str, dict[str, float | int | None]] = {}
    for label, bucket in (
        ("negative", candidate_temporal < -TIE_TOLERANCE),
        ("zero", np.abs(candidate_temporal) <= TIE_TOLERANCE),
        ("positive", candidate_temporal > TIE_TOLERANCE),
    ):
        temporal_buckets[label] = {
            "count": int(np.sum(bucket)),
            "prediction_mean": _safe_mean(candidate_score[bucket]),
            "filtered_relation_mean": _safe_mean(candidate_filtered[bucket]),
            "contextual_relation_mean": _safe_mean(candidate_target[bucket]),
            "predicted_positive_count": int(
                np.sum(bucket & (candidate_score > 0.0))
            ),
        }

    by_group: dict[str, list[float]] = defaultdict(list)
    for local_index, index in enumerate(bundle_index):
        by_group[str(metadata[int(index)]["recording_group"])].append(
            float(gain[local_index])
        )
    group_mean = {group: float(np.mean(values)) for group, values in by_group.items()}

    frontier = _threshold_frontier(prediction, target, mask)
    best_threshold = max(
        frontier,
        key=lambda row: (
            float(row["mean_gain_over_native"]),
            -int(row["nonnative_choice_count"]),
        ),
    )
    harmful_rows = np.flatnonzero(harmful)
    worst = sorted(harmful_rows, key=lambda index: float(gain[index]))[:10]
    worst_cases = [
        {
            "bundle_index": int(bundle_index[index]),
            "identity": str(metadata[int(bundle_index[index])]["identity"]),
            "recording_group": str(
                metadata[int(bundle_index[index])]["recording_group"]
            ),
            "selected_candidate": int(selected[index]),
            "selected_score": float(selected_score[index]),
            "realized_gain": float(gain[index]),
            "geometry": {
                name: _finite_scalar(
                    float(value[int(bundle_index[index]), int(selected[index])])
                )
                for name, value in geometry.items()
            },
            "target_decomposition": {
                "filtered_one_frame_relation": float(
                    filtered_relation[index, int(selected[index])]
                ),
                "weighted_filtered_contribution": float(
                    0.875 * filtered_relation[index, int(selected[index])]
                ),
                "temporal_hard_gate_relation": float(
                    temporal_gate_relation[index, int(selected[index])]
                ),
                "weighted_temporal_contribution": float(
                    0.125 * temporal_gate_relation[index, int(selected[index])]
                ),
            },
        }
        for index in worst
    ]
    by_candidate_index: dict[str, dict[str, Any]] = {}
    for candidate_index in range(1, prediction.shape[1]):
        valid = mask[:, candidate_index]
        chosen = selected == candidate_index
        chosen_beneficial = chosen & beneficial
        chosen_harmful = chosen & harmful
        index_target = np.asarray(target[valid, candidate_index], dtype=np.float64)
        index_prediction = np.asarray(
            prediction[valid, candidate_index], dtype=np.float64
        )
        by_candidate_index[str(candidate_index)] = {
            "valid_count": int(np.sum(valid)),
            "target_positive_count": int(np.sum(index_target > TIE_TOLERANCE)),
            "target_negative_count": int(np.sum(index_target < -TIE_TOLERANCE)),
            "target_mean": _safe_mean(index_target),
            "target_std": _safe_std(index_target),
            "prediction_mean": _safe_mean(index_prediction),
            "prediction_std": _safe_std(index_prediction),
            "prediction_target_pearson": (
                float(np.corrcoef(index_prediction, index_target)[0, 1])
                if index_target.size > 1
                and float(np.std(index_target)) > 0.0
                and float(np.std(index_prediction)) > 0.0
                else None
            ),
            "chosen_count": int(np.sum(chosen)),
            "chosen_beneficial_count": int(np.sum(chosen_beneficial)),
            "chosen_harmful_count": int(np.sum(chosen_harmful)),
            "chosen_gain_sum": float(np.sum(gain[chosen])),
            "chosen_gain_mean": _safe_mean(gain[chosen]),
            "chosen_score_mean": _safe_mean(selected_score[chosen]),
            "geometry_all_valid": {
                name: _finite_summary(value[bundle_index[valid], candidate_index])
                for name, value in geometry.items()
            },
            "geometry_when_chosen": {
                name: _finite_summary(value[bundle_index[chosen], candidate_index])
                for name, value in geometry.items()
            },
            "geometry_when_harmful_chosen": {
                name: _finite_summary(
                    value[bundle_index[chosen_harmful], candidate_index]
                )
                for name, value in geometry.items()
            },
            "target_component_when_chosen": {
                "filtered_relation": _finite_summary(
                    filtered_relation[chosen, candidate_index]
                ),
                "temporal_hard_gate_relation": _finite_summary(
                    temporal_gate_relation[chosen, candidate_index]
                ),
            },
            "target_component_when_beneficial_chosen": {
                "filtered_relation": _finite_summary(
                    filtered_relation[chosen_beneficial, candidate_index]
                ),
                "temporal_hard_gate_relation": _finite_summary(
                    temporal_gate_relation[chosen_beneficial, candidate_index]
                ),
            },
            "target_component_when_harmful_chosen": {
                "filtered_relation": _finite_summary(
                    filtered_relation[chosen_harmful, candidate_index]
                ),
                "temporal_hard_gate_relation": _finite_summary(
                    temporal_gate_relation[chosen_harmful, candidate_index]
                ),
            },
        }
    epoch_ledger = cast(list[dict[str, Any]], primary_result["epoch_ledger"])
    return {
        "schema": "aorta.r005e3d.failure_anatomy.v1",
        "status": "POSTHOC_EXPLORATORY_READ_ONLY",
        "parent_decision_immutable": audit["decision"],
        "selection_anatomy": {
            "anchor_count": int(prediction.shape[0]),
            "nonnative_count": int(np.sum(nonnative)),
            "beneficial_count": int(np.sum(beneficial)),
            "harmful_count": int(np.sum(harmful)),
            "neutral_count": int(np.sum(neutral)),
            "beneficial_gain_sum": float(np.sum(gain[beneficial])),
            "harmful_gain_sum": float(np.sum(gain[harmful])),
            "beneficial_gain_mean": _safe_mean(gain[beneficial]),
            "harmful_gain_mean": _safe_mean(gain[harmful]),
            "selected_score_mean_beneficial": _safe_mean(selected_score[beneficial]),
            "selected_score_mean_harmful": _safe_mean(selected_score[harmful]),
            "selected_score_minus_gain_mean": _safe_mean(
                selected_score[nonnative] - gain[nonnative]
            ),
        },
        "candidate_sign_calibration": {
            "predicted_positive_count": int(np.sum(predicted_positive)),
            "target_positive_count": int(np.sum(target_positive)),
            "true_positive_count": int(np.sum(predicted_positive & target_positive)),
            "false_positive_count": int(np.sum(predicted_positive & ~target_positive)),
            "false_negative_count": int(np.sum(~predicted_positive & target_positive)),
            "precision_at_zero": (
                float(np.mean(target_positive[predicted_positive]))
                if bool(np.any(predicted_positive))
                else None
            ),
            "recall_at_zero": (
                float(np.mean(predicted_positive[target_positive]))
                if bool(np.any(target_positive))
                else None
            ),
            "pearson_score_target": float(
                np.corrcoef(candidate_score, candidate_target)[0, 1]
            ),
        },
        "component_readout_diagnostic": {
            "pearson_prediction_vs_contextual_relation": float(
                np.corrcoef(candidate_score, candidate_target)[0, 1]
            ),
            "pearson_prediction_vs_filtered_relation": float(
                np.corrcoef(candidate_score, candidate_filtered)[0, 1]
            ),
            "pearson_prediction_vs_temporal_hard_gate_relation": float(
                np.corrcoef(candidate_score, candidate_temporal)[0, 1]
            ),
            "temporal_relation_buckets": temporal_buckets,
            "selected_harmful_with_negative_temporal_count": int(
                np.sum(
                    temporal_gate_relation[
                        np.arange(prediction.shape[0])[harmful], selected[harmful]
                    ]
                    < -TIE_TOLERANCE
                )
            ),
            "selected_harmful_count": int(np.sum(harmful)),
            "interpretation_boundary": (
                "Association only: this can reveal which registered target component "
                "the frozen scorer failed to read, but cannot prove a replacement "
                "architecture will generalize."
            ),
        },
        "group_anatomy": {
            "group_count": len(group_mean),
            "positive_mean_gain_group_count": sum(value > 0.0 for value in group_mean.values()),
            "negative_mean_gain_group_count": sum(value < 0.0 for value in group_mean.values()),
            "zero_mean_gain_group_count": sum(value == 0.0 for value in group_mean.values()),
            "worst_group_mean_gain": min(group_mean.values()),
            "best_group_mean_gain": max(group_mean.values()),
            "group_mean_gain": group_mean,
        },
        "candidate_index_and_geometry_anatomy": {
            "sample_interval_s": 0.1,
            "native_candidate_index": 0,
            "rows": by_candidate_index,
            "warning": (
                "Candidate-index and trajectory-geometry associations are post-hoc. "
                "They can identify a discriminating next experiment but cannot establish "
                "causality or justify candidate-specific treatment."
            ),
        },
        "target_decomposition": {
            "formula": (
                "Delta_Uctx = 0.875 * Delta_filtered_one_frame + "
                "0.125 * Delta_(hard_gate*two_frame_comfort)"
            ),
            "maximum_absolute_reconstruction_error": float(
                np.max(np.abs(reconstructed_relation[mask] - target[mask]))
            ),
            "selected_beneficial": {
                "filtered_relation": _finite_summary(
                    filtered_relation[
                        np.arange(prediction.shape[0])[beneficial], selected[beneficial]
                    ]
                ),
                "temporal_hard_gate_relation": _finite_summary(
                    temporal_gate_relation[
                        np.arange(prediction.shape[0])[beneficial], selected[beneficial]
                    ]
                ),
            },
            "selected_harmful": {
                "filtered_relation": _finite_summary(
                    filtered_relation[
                        np.arange(prediction.shape[0])[harmful], selected[harmful]
                    ]
                ),
                "temporal_hard_gate_relation": _finite_summary(
                    temporal_gate_relation[
                        np.arange(prediction.shape[0])[harmful], selected[harmful]
                    ]
                ),
            },
        },
        "posthoc_threshold_frontier": {
            "rule": "select top challenger only when predicted native-relative score exceeds threshold",
            "best_in_sample_row": best_threshold,
            "rows": frontier,
            "warning": (
                "Threshold is selected on the same outer-train anchors and is not a valid "
                "method result; it only tests whether nested group-held-out selective "
                "authorization is worth preregistering."
            ),
        },
        "epoch_path_provenance_limited": {
            "rows": [
                {
                    "epoch": int(row["epoch"]),
                    "mean_gain_over_native": float(row["mean_gain_over_native"]),
                    "relation_mse": float(row["relation_mse"]),
                    "nonnative_choice_count": int(row["nonnative_choice_count"]),
                    "beneficial_count": int(row["beneficial_nonnative_count"]),
                    "harmful_count": int(row["harmful_nonnative_count"]),
                }
                for row in epoch_ledger
            ],
            "warning": "Epoch predictions were not archived, so ledger values are exact-replay self-consistent but not independently recomputed.",
        },
        "worst_harmful_cases": worst_cases,
        "claim_boundary": (
            "Post-hoc fold-0 outer-train failure anatomy only. It cannot change the "
            "frozen FAIL, select a deployable threshold, or support OOF, visual, "
            "Differential Query, appearance, behavior, or safety claims."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = analyze(arguments.result_root)
    _write_json(arguments.output, result)
    print(json.dumps(result, sort_keys=True))
