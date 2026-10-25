"""Audit whether B4R1 ``pdm_score`` equals the filtered one-stage score.

NAVSIM's candidate-local scorer calculates ``pdm_score`` before the optional
human false-positive filter.  The returned metric components can then be
modified by that filter, while the complete evaluator later recomputes its
``score`` from the modified components.  This audit measures that semantic gap
without changing the registered B4R1 result or its decision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


WEIGHTS = {
    "ego_progress": 5.0,
    "time_to_collision_within_bound": 5.0,
    "lane_keeping": 2.0,
    "history_comfort": 2.0,
}
HARD_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
)


def filtered_one_stage_score(candidate: Mapping[str, Any]) -> float:
    """Recompute the one-stage score from the returned filtered components."""

    hard = cast(Mapping[str, Any], candidate["hard_metrics"])
    weighted = cast(Mapping[str, Any], candidate["weighted_metrics"])
    multiplier = math.prod(float(hard[name]) for name in HARD_METRICS)
    numerator = sum(float(weighted[name]) * weight for name, weight in WEIGHTS.items())
    return multiplier * numerator / sum(WEIGHTS.values())


def _sign(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _top_set(values: Sequence[float], tolerance: float) -> tuple[int, ...]:
    maximum = max(values)
    return tuple(index for index, value in enumerate(values) if maximum - value <= tolerance)


def audit(result: Mapping[str, Any], tolerance: float = 1.0e-12) -> dict[str, Any]:
    anchors = cast(Sequence[Mapping[str, Any]], result["anchor_ledgers"])
    mismatch_candidates = 0
    mismatch_anchors = 0
    mismatch_groups: set[str] = set()
    max_abs_difference = 0.0
    pair_sign_disagreements = 0
    pair_count = 0
    top_set_disagreements = 0
    raw_positive_gain_anchors = 0
    filtered_positive_gain_anchors = 0
    gain_sign_disagreements = 0

    for anchor in anchors:
        candidates = cast(Sequence[Mapping[str, Any]], anchor["candidates"])
        raw = [float(candidate["one_frame_pdm_score"]) for candidate in candidates]
        filtered = [filtered_one_stage_score(candidate) for candidate in candidates]
        differences = [abs(left - right) for left, right in zip(raw, filtered)]
        candidate_mismatches = sum(value > tolerance for value in differences)
        mismatch_candidates += candidate_mismatches
        max_abs_difference = max(max_abs_difference, *differences)
        if candidate_mismatches:
            mismatch_anchors += 1
            mismatch_groups.add(str(anchor["recording_group"]))

        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                pair_count += 1
                if _sign(raw[left] - raw[right], tolerance) != _sign(
                    filtered[left] - filtered[right], tolerance
                ):
                    pair_sign_disagreements += 1

        if _top_set(raw, tolerance) != _top_set(filtered, tolerance):
            top_set_disagreements += 1

        raw_gain = max(raw) - raw[0]
        filtered_gain = max(filtered) - filtered[0]
        raw_positive_gain_anchors += raw_gain > tolerance
        filtered_positive_gain_anchors += filtered_gain > tolerance
        gain_sign_disagreements += _sign(raw_gain, tolerance) != _sign(
            filtered_gain, tolerance
        )

    return {
        "experiment_id": result["experiment_id"],
        "audit_role": "posthoc_semantic_diagnostic_only__registered_decision_unchanged",
        "tolerance": tolerance,
        "anchor_count": len(anchors),
        "candidate_count": sum(len(anchor["candidates"]) for anchor in anchors),
        "pair_count": pair_count,
        "mismatch_candidate_count": mismatch_candidates,
        "mismatch_anchor_count": mismatch_anchors,
        "mismatch_group_count": len(mismatch_groups),
        "max_abs_score_difference": max_abs_difference,
        "pair_sign_disagreement_count": pair_sign_disagreements,
        "top_set_disagreement_anchor_count": top_set_disagreements,
        "raw_positive_gain_anchor_count": raw_positive_gain_anchors,
        "filtered_positive_gain_anchor_count": filtered_positive_gain_anchors,
        "gain_sign_disagreement_anchor_count": gain_sign_disagreements,
        "registered_b4r1_decision_changed": False,
        "full_target_claim_supported": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = cast(dict[str, Any], json.loads(args.result.read_text(encoding="utf-8")))
    report = audit(result, tolerance=args.tolerance)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

