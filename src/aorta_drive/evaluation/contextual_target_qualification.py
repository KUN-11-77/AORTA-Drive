"""Outcome-free qualification for R005 contextual decision targets.

This module deliberately does not compute candidate utilities.  It verifies
that the frozen B4R1 candidate domain can be joined to causal previous-frame
context and reports whether official two-stage mappings define a source-fit
counterfactual action value.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


HARD_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
)
WEIGHTED_NAMES = (
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
)
WEIGHTS = (5.0, 5.0, 2.0, 2.0)


def _key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["recording_group"]), int(row["anchor_index"])


def filtered_one_stage(candidate: Mapping[str, Any]) -> float:
    """Recompute the post-filter one-stage score from returned components."""

    hard = cast(Mapping[str, Any], candidate["hard_metrics"])
    weighted = cast(Mapping[str, Any], candidate["weighted_metrics"])
    multiplier = math.prod(float(hard[name]) for name in HARD_NAMES)
    numerator = sum(
        weight * float(weighted[name])
        for name, weight in zip(WEIGHTED_NAMES, WEIGHTS)
    )
    value = multiplier * numerator / sum(WEIGHTS)
    if not math.isfinite(value):
        raise ValueError("nonfinite post-filter one-stage score")
    return value


def _mapping_pairs(payloads: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for payload in payloads:
        rows = payload.get("reactive_all_mapping") or []
        for row in cast(Sequence[Sequence[Any]], rows):
            if len(row) < 2:
                raise ValueError("two-stage mapping row has fewer than two tokens")
            pairs.add((str(row[0]), str(row[1])))
    return pairs


def qualify_contextual_target(
    b4_result: Mapping[str, Any],
    context_roster: Mapping[str, Any],
    mapping_payloads: Sequence[Mapping[str, Any]],
    *,
    previous_metric_root: Path | None = None,
) -> dict[str, Any]:
    """Qualify source-fit U_ctx and fixed-continuation Q without outcomes."""

    b4_rows = cast(Sequence[Mapping[str, Any]], b4_result["anchor_ledgers"])
    context_rows = cast(Sequence[Mapping[str, Any]], context_roster["anchor_ledger"])
    if len(b4_rows) != 672 or len(context_rows) != 672:
        raise ValueError("expected exactly 672 B4 and context rows")
    if b4_result.get("development_calibration_test_navdream_target_accessed") is not False:
        raise ValueError("sealed-role access flag is not false")
    if b4_result.get("full_official_pdms_claimed") is not False:
        raise ValueError("B4 result unexpectedly claims full official PDMS")
    if context_roster.get("development_calibration_test_navdream_target_accessed") is not False:
        raise ValueError("context roster sealed-role access flag is not false")
    if context_roster.get("scientific_outcomes_accessed") is not False:
        raise ValueError("context roster accessed scientific outcomes")

    b4_by_key = {_key(row): row for row in b4_rows}
    context_by_key = {_key(row): row for row in context_rows}
    if len(b4_by_key) != 672 or len(context_by_key) != 672:
        raise ValueError("duplicate B4 or context anchor keys")
    if set(b4_by_key) != set(context_by_key):
        raise ValueError("B4 and context rosters do not have identical anchor keys")

    candidate_count = 0
    challenger_count = 0
    pair_count = 0
    filter_difference_count = 0
    previous_cache_count = 0
    current_previous_pairs: set[tuple[str, str]] = set()
    groups: set[str] = set()

    for key in sorted(b4_by_key):
        b4_row = b4_by_key[key]
        context = context_by_key[key]
        groups.add(key[0])
        current_token = str(b4_row["scene_token"])
        if str(context["current_token"]) != current_token:
            raise ValueError(f"current token mismatch at {key}")
        if str(context["log_name"]) != str(b4_row["log_name"]):
            raise ValueError(f"log mismatch at {key}")
        if context.get("status") != "AVAILABLE":
            raise ValueError(f"previous context unavailable at {key}")
        gap = float(context["gap_s"])
        if not (0.0 < gap < 0.55):
            raise ValueError(f"invalid previous-context gap at {key}: {gap}")
        previous_token = str(context["previous_token"])
        current_previous_pairs.add((current_token, previous_token))
        if previous_metric_root is not None:
            previous_path = (
                previous_metric_root
                / "metric_cache"
                / str(context["log_name"])
                / "unknown"
                / previous_token
                / "metric_cache.pkl"
            )
            previous_cache_count += previous_path.is_file()

        legal = [int(value) for value in cast(Sequence[Any], b4_row["legal_registered_indices"])]
        candidates = cast(Sequence[Mapping[str, Any]], b4_row["candidates"])
        candidate_indices = [int(candidate["registered_index"]) for candidate in candidates]
        if not legal or legal[0] != 0 or candidate_indices != legal:
            raise ValueError(f"candidate membership mismatch at {key}")
        if b4_row.get("label_generation_valid") is not True:
            raise ValueError(f"invalid B4 label generation at {key}")
        if b4_row.get("panel_pose_reconstruction_bit_exact") is not True:
            raise ValueError(f"pose reconstruction mismatch at {key}")
        if b4_row.get("membership_artifact_exact") is not True:
            raise ValueError(f"membership artifact mismatch at {key}")
        if int(b4_row.get("ineligible_candidate_score_count", -1)) != 0:
            raise ValueError(f"ineligible candidate was scored at {key}")

        raw_values: list[float] = []
        filtered_values: list[float] = []
        for candidate in candidates:
            raw = float(candidate["one_frame_pdm_score"])
            if not math.isfinite(raw):
                raise ValueError(f"nonfinite raw score at {key}")
            raw_values.append(raw)
            filtered_values.append(filtered_one_stage(candidate))
        filter_difference_count += sum(
            abs(raw - filtered) > 1.0e-12
            for raw, filtered in zip(raw_values, filtered_values)
        )
        candidate_count += len(candidates)
        challenger_count += len(candidates) - 1
        pair_count += len(candidates) * (len(candidates) - 1) // 2

    mapping_pairs = _mapping_pairs(mapping_payloads)
    mapping_covered_pairs = len(current_previous_pairs & mapping_pairs)
    uctx_definition_valid = True
    previous_cache_ready = previous_metric_root is not None and previous_cache_count == 672
    q_counterfactual_mapping_valid = mapping_covered_pairs == 672
    if previous_cache_ready and q_counterfactual_mapping_valid:
        decision = "E1A_PASS_UCTX_AND_Q_SOURCE_READY"
    elif previous_cache_ready:
        decision = "E1A_PASS_UCTX_READY__Q_PI_REF_UNDEFINED_ON_SOURCE_FIT"
    else:
        decision = "E1A_PASS_DEFINITION__PREVIOUS_CACHE_BUILD_REQUIRED__Q_PI_REF_UNDEFINED_ON_SOURCE_FIT"

    return {
        "status": "PASS",
        "decision": decision,
        "anchor_count": len(b4_rows),
        "group_count": len(groups),
        "candidate_count": candidate_count,
        "challenger_count": challenger_count,
        "pair_count": pair_count,
        "filter_difference_candidate_count": filter_difference_count,
        "context_available_count": len(context_rows),
        "current_previous_pair_count": len(current_previous_pairs),
        "previous_metric_cache_available_count": previous_cache_count,
        "previous_metric_cache_required_count": 672,
        "uctx_definition_valid": uctx_definition_valid,
        "uctx_materialization_ready": previous_cache_ready,
        "official_two_stage_mapping_pair_count": len(mapping_pairs),
        "source_fit_mapping_covered_pair_count": mapping_covered_pairs,
        "q_pi_ref_counterfactual_mapping_valid": q_counterfactual_mapping_valid,
        "scientific_outcomes_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
    }

