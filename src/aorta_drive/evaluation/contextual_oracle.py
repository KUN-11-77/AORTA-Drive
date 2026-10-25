"""Pure scoring and aggregation for the R005 contextual oracle."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Mapping, Sequence, cast

import numpy as np

from aorta_drive.evaluation.contextual_target_qualification import (
    HARD_NAMES,
    WEIGHTED_NAMES,
    filtered_one_stage,
)
from aorta_drive.evaluation.eligible_preference_oracle import (
    EligiblePreferenceOracleProtocol,
    summarize_eligible_preference_oracle,
)


def contextual_utility(candidate: Mapping[str, Any], two_frame_comfort: float) -> float:
    """Compute the official one-stage row after inserting two-frame comfort."""

    if two_frame_comfort not in (0.0, 1.0):
        raise ValueError("two-frame comfort must be binary")
    hard = cast(Mapping[str, Any], candidate["hard_metrics"])
    weighted = cast(Mapping[str, Any], candidate["weighted_metrics"])
    multiplier = math.prod(float(hard[name]) for name in HARD_NAMES)
    numerator = (
        5.0 * float(weighted["ego_progress"])
        + 5.0 * float(weighted["time_to_collision_within_bound"])
        + 2.0 * float(weighted["lane_keeping"])
        + 2.0 * float(weighted["history_comfort"])
        + 2.0 * two_frame_comfort
    )
    value = multiplier * numerator / 16.0
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("contextual utility is outside [0, 1]")
    return value


def _sign(value: float, tie: float) -> int:
    if value > tie:
        return 1
    if value < -tie:
        return -1
    return 0


def _top_set(values: np.ndarray[Any, np.dtype[np.float64]], tie: float) -> set[int]:
    maximum = float(np.max(values))
    return {index for index, value in enumerate(values) if maximum - value <= tie}


def summarize_contextual_oracle(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_names: Sequence[str],
    protocol: EligiblePreferenceOracleProtocol,
    diagnostic_margins: Sequence[float],
) -> dict[str, Any]:
    """Summarize contextual opportunity and its change from one-frame targets."""

    contextual_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    tie = protocol.numerical_tie_tolerance
    for row in rows:
        candidates = cast(Sequence[Mapping[str, Any]], row["candidates"])
        raw = np.asarray(
            [float(candidate["one_frame_pdm_score"]) for candidate in candidates],
            dtype=np.float64,
        )
        filtered = np.asarray(
            [filtered_one_stage(candidate) for candidate in candidates],
            dtype=np.float64,
        )
        contextual = np.asarray(
            [float(candidate["contextual_utility"]) for candidate in candidates],
            dtype=np.float64,
        )
        if not all(np.all(np.isfinite(values)) for values in (raw, filtered, contextual)):
            raise ValueError("nonfinite candidate utility")
        pair_indices = tuple(combinations(range(len(candidates)), 2))
        raw_context_sign_disagreements = sum(
            _sign(float(raw[j] - raw[i]), tie)
            != _sign(float(contextual[j] - contextual[i]), tie)
            for i, j in pair_indices
        )
        filtered_context_sign_disagreements = sum(
            _sign(float(filtered[j] - filtered[i]), tie)
            != _sign(float(contextual[j] - contextual[i]), tie)
            for i, j in pair_indices
        )
        filtered_choice = int(np.argmax(filtered))
        comparisons.append(
            {
                "recording_group": str(row["recording_group"]),
                "raw_context_top_set_disagreement": _top_set(raw, tie)
                != _top_set(contextual, tie),
                "filtered_context_top_set_disagreement": _top_set(filtered, tie)
                != _top_set(contextual, tie),
                "raw_context_sign_disagreements": raw_context_sign_disagreements,
                "filtered_context_sign_disagreements": (
                    filtered_context_sign_disagreements
                ),
                "pair_count": len(pair_indices),
                "ignored_context_regret": float(
                    np.max(contextual) - contextual[filtered_choice]
                ),
            }
        )
        copied = dict(row)
        copied["candidates"] = [
            {**dict(candidate), "one_frame_pdm_score": float(contextual[index])}
            for index, candidate in enumerate(candidates)
        ]
        contextual_rows.append(copied)

    base = summarize_eligible_preference_oracle(
        contextual_rows,
        group_names=group_names,
        protocol=protocol,
        diagnostic_margins=diagnostic_margins,
    )
    group_comparisons: list[dict[str, Any]] = []
    for group in group_names:
        records = [record for record in comparisons if record["recording_group"] == group]
        if len(records) != 16:
            raise ValueError(f"contextual comparison group is incomplete: {group}")
        pair_count = sum(int(record["pair_count"]) for record in records)
        group_comparisons.append(
            {
                "recording_group": group,
                "raw_context_top_set_disagreement_fraction": float(
                    np.mean([record["raw_context_top_set_disagreement"] for record in records])
                ),
                "filtered_context_top_set_disagreement_fraction": float(
                    np.mean(
                        [
                            record["filtered_context_top_set_disagreement"]
                            for record in records
                        ]
                    )
                ),
                "raw_context_pair_sign_disagreement_fraction": (
                    sum(int(record["raw_context_sign_disagreements"]) for record in records)
                    / pair_count
                    if pair_count
                    else 0.0
                ),
                "filtered_context_pair_sign_disagreement_fraction": (
                    sum(
                        int(record["filtered_context_sign_disagreements"])
                        for record in records
                    )
                    / pair_count
                    if pair_count
                    else 0.0
                ),
                "mean_ignored_context_regret_all_anchors": float(
                    np.mean([record["ignored_context_regret"] for record in records])
                ),
            }
        )

    groups_with_improvement = int(base["groups_with_strict_native_improvement"])
    gain_lower = float(base["group_bootstrap_oracle_gain_ci95"][0])
    implementation_pass = bool(base["implementation_pass"])
    signal_pass = (
        int(base["groups_with_nonzero_pair"])
        >= protocol.minimum_groups_with_nonzero_pair
        and groups_with_improvement
        >= protocol.minimum_groups_with_strict_native_improvement
        and gain_lower > 0.0
    )
    if not implementation_pass:
        decision = "INVALID_TARGET_CONSTRUCTION"
    elif signal_pass:
        decision = "GO_TO_VISUAL_RECOVERABILITY_PREREGISTRATION__NO_TRAINING_YET"
    elif groups_with_improvement == 0:
        decision = "STOP_CURRENT_PANEL_SUPERVISION"
    else:
        decision = "HOLD_CONTEXTUAL_SIGNAL_INSUFFICIENT"

    total_pairs = sum(int(record["pair_count"]) for record in comparisons)
    return {
        **base,
        "decision": decision,
        "raw_context_top_set_disagreement_anchor_count": sum(
            bool(record["raw_context_top_set_disagreement"]) for record in comparisons
        ),
        "filtered_context_top_set_disagreement_anchor_count": sum(
            bool(record["filtered_context_top_set_disagreement"])
            for record in comparisons
        ),
        "raw_context_pair_sign_disagreement_count": sum(
            int(record["raw_context_sign_disagreements"]) for record in comparisons
        ),
        "filtered_context_pair_sign_disagreement_count": sum(
            int(record["filtered_context_sign_disagreements"])
            for record in comparisons
        ),
        "comparison_pair_count": total_pairs,
        "group_equal_mean_ignored_context_regret_all_anchors": float(
            np.mean(
                [
                    group["mean_ignored_context_regret_all_anchors"]
                    for group in group_comparisons
                ]
            )
        ),
        "context_comparison_group_summaries": group_comparisons,
    }
