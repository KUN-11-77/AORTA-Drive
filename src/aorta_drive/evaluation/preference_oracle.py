"""Source-only oracle diagnostics for candidate preference supervision.

The functions in this module intentionally operate on already materialized
candidate outcomes.  NAVSIM execution and score generation stay in the runner;
the scientific decision can therefore be replayed without importing NAVSIM.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PreferenceOracleProtocol:
    """Frozen source-only qualification thresholds."""

    numerical_tie_tolerance: float
    minimum_groups_with_nonzero_pair: int
    minimum_groups_with_strict_native_improvement: int
    bootstrap_seed: int
    bootstrap_resamples: int

    def __post_init__(self) -> None:
        if self.numerical_tie_tolerance < 0.0:
            raise ValueError("numerical tie tolerance must be nonnegative")
        if not 1 <= self.minimum_groups_with_nonzero_pair <= 42:
            raise ValueError("nonzero-pair group gate must lie in [1, 42]")
        if not 1 <= self.minimum_groups_with_strict_native_improvement <= 42:
            raise ValueError("improvement group gate must lie in [1, 42]")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap resamples must be positive")


def _bootstrap_mean_ci(
    values: FloatArray, *, seed: int, resamples: int
) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be a nonempty finite vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    estimates = np.mean(values[draws], axis=1)
    return float(np.quantile(estimates, 0.025)), float(
        np.quantile(estimates, 0.975)
    )


def _candidate_scores(row: Mapping[str, Any]) -> FloatArray:
    candidates = row.get("candidates")
    if not isinstance(candidates, Sequence) or len(candidates) != 7:
        raise ValueError("every anchor must contain native plus six candidates")
    indices = [int(candidate["registered_index"]) for candidate in candidates]
    if indices != list(range(7)):
        raise ValueError("candidate indices must be the frozen order 0..6")
    scores = np.asarray(
        [float(candidate["one_frame_pdm_score"]) for candidate in candidates],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("candidate scores must be finite values in [0, 1]")
    return scores


def _hard_product(candidate: Mapping[str, Any]) -> float:
    names = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "traffic_light_compliance",
    )
    metrics = candidate.get("hard_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(names):
        raise ValueError("candidate hard metrics do not match the frozen contract")
    values = np.asarray([float(metrics[name]) for name in names], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("hard metrics must be finite values in [0, 1]")
    return float(np.prod(values))


def summarize_preference_oracle(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_names: Sequence[str],
    protocol: PreferenceOracleProtocol,
    diagnostic_margins: Sequence[float],
) -> dict[str, Any]:
    """Summarize label legality and the source-only preference oracle ceiling.

    Recording groups are the independent units.  Candidate pairs and anchors
    are repeated observations and never enter the bootstrap independently.
    """

    if len(rows) != 672:
        raise ValueError("preference oracle requires exactly 672 fit anchors")
    if len(group_names) != 42 or len(set(group_names)) != 42:
        raise ValueError("preference oracle requires 42 unique fit groups")
    expected_keys = {(group, anchor) for group in group_names for anchor in range(16)}
    observed_keys = {
        (str(row["recording_group"]), int(row["anchor_index"])) for row in rows
    }
    if observed_keys != expected_keys or len(observed_keys) != len(rows):
        raise ValueError("anchor identities do not equal 42 groups x 16 anchors")

    margins = tuple(float(value) for value in diagnostic_margins)
    if not margins or margins != tuple(sorted(set(margins))) or margins[0] < 0.0:
        raise ValueError("diagnostic margins must be sorted unique nonnegative values")

    tie = protocol.numerical_tie_tolerance
    anchor_records: list[dict[str, Any]] = []
    for row in rows:
        scores = _candidate_scores(row)
        candidates = row["candidates"]
        hard_products = np.asarray(
            [_hard_product(candidate) for candidate in candidates], dtype=np.float64
        )
        pair_differences = np.asarray(
            [scores[j] - scores[i] for i, j in combinations(range(7), 2)],
            dtype=np.float64,
        )
        oracle_index = int(np.argmax(scores))
        oracle_gain = float(scores[oracle_index] - scores[0])
        native_hard_failure = bool(hard_products[0] < 1.0 - tie)
        hard_rescue = bool(
            native_hard_failure
            and np.any(
                (hard_products[1:] >= 1.0 - tie)
                & (scores[1:] > scores[0] + tie)
            )
        )
        anchor_records.append(
            {
                "recording_group": str(row["recording_group"]),
                "has_nonzero_pair": bool(np.any(np.abs(pair_differences) > tie)),
                "nonzero_pair_fraction": float(
                    np.mean(np.abs(pair_differences) > tie)
                ),
                "strict_native_improvement": bool(oracle_gain > tie),
                "oracle_gain": oracle_gain,
                "unique_top": int(np.sum(np.abs(scores - np.max(scores)) <= tie)) == 1,
                "native_hard_failure": native_hard_failure,
                "hard_rescue": hard_rescue,
                "margin_rates": {
                    f"{margin:.12g}": float(
                        np.mean(np.abs(pair_differences) > margin)
                    )
                    for margin in margins
                },
            }
        )

    group_rows: list[dict[str, Any]] = []
    for group in group_names:
        anchors = [
            record for record in anchor_records if record["recording_group"] == group
        ]
        if len(anchors) != 16:
            raise ValueError(f"group does not contain 16 anchors: {group}")
        group_rows.append(
            {
                "recording_group": group,
                "anchors_with_nonzero_pair": sum(
                    bool(record["has_nonzero_pair"]) for record in anchors
                ),
                "anchors_with_strict_native_improvement": sum(
                    bool(record["strict_native_improvement"]) for record in anchors
                ),
                "mean_nonzero_pair_fraction": float(
                    np.mean(
                        [float(record["nonzero_pair_fraction"]) for record in anchors]
                    )
                ),
                "mean_oracle_gain": float(
                    np.mean([float(record["oracle_gain"]) for record in anchors])
                ),
                "unique_top_fraction": float(
                    np.mean([bool(record["unique_top"]) for record in anchors])
                ),
                "native_hard_failure_count": sum(
                    bool(record["native_hard_failure"]) for record in anchors
                ),
                "hard_rescue_count": sum(
                    bool(record["hard_rescue"]) for record in anchors
                ),
                "margin_rates": {
                    f"{margin:.12g}": float(
                        np.mean(
                            [
                                float(record["margin_rates"][f"{margin:.12g}"])
                                for record in anchors
                            ]
                        )
                    )
                    for margin in margins
                },
            }
        )

    mean_gains = np.asarray(
        [float(group["mean_oracle_gain"]) for group in group_rows], dtype=np.float64
    )
    nonzero_rates = np.asarray(
        [float(group["mean_nonzero_pair_fraction"]) for group in group_rows],
        dtype=np.float64,
    )
    gain_ci = _bootstrap_mean_ci(
        mean_gains,
        seed=protocol.bootstrap_seed,
        resamples=protocol.bootstrap_resamples,
    )
    nonzero_ci = _bootstrap_mean_ci(
        nonzero_rates,
        seed=protocol.bootstrap_seed + 1,
        resamples=protocol.bootstrap_resamples,
    )
    groups_with_nonzero = sum(
        int(group["anchors_with_nonzero_pair"]) > 0 for group in group_rows
    )
    groups_with_improvement = sum(
        int(group["anchors_with_strict_native_improvement"]) > 0
        for group in group_rows
    )
    legal = all(bool(row.get("label_generation_valid")) for row in rows)
    reconstruction_exact = all(
        bool(row.get("panel_reconstruction_bit_exact")) for row in rows
    )
    replay_exact = all(bool(row.get("sentinel_replay_exact")) for row in rows)
    signal_pass = (
        groups_with_nonzero >= protocol.minimum_groups_with_nonzero_pair
        and groups_with_improvement
        >= protocol.minimum_groups_with_strict_native_improvement
        and gain_ci[0] > 0.0
    )
    implementation_pass = legal and reconstruction_exact and replay_exact
    if not implementation_pass:
        decision = "INVALID_PREFERENCE_LABEL_INTERFACE"
    elif signal_pass:
        decision = "GO_TO_MATCHED_STRUCTURE_PREREGISTRATION__NO_TRAINING_YET"
    elif groups_with_improvement == 0:
        decision = "STOP_CURRENT_PANEL_NO_ORACLE_IMPROVEMENT"
    else:
        decision = "HOLD_CURRENT_PANEL_PREFERENCE_SIGNAL_INSUFFICIENT"

    return {
        "anchor_count": len(rows),
        "candidate_count": 7 * len(rows),
        "pair_count": 21 * len(rows),
        "recording_group_count": len(group_names),
        "implementation_pass": implementation_pass,
        "label_generation_valid": legal,
        "panel_reconstruction_bit_exact": reconstruction_exact,
        "sentinel_replay_exact": replay_exact,
        "groups_with_nonzero_pair": groups_with_nonzero,
        "groups_with_strict_native_improvement": groups_with_improvement,
        "group_equal_mean_nonzero_pair_fraction": float(np.mean(nonzero_rates)),
        "group_bootstrap_nonzero_pair_fraction_ci95": list(nonzero_ci),
        "group_equal_mean_oracle_gain": float(np.mean(mean_gains)),
        "group_bootstrap_oracle_gain_ci95": list(gain_ci),
        "unique_top_anchor_count": sum(
            bool(record["unique_top"]) for record in anchor_records
        ),
        "native_hard_failure_anchor_count": sum(
            bool(record["native_hard_failure"]) for record in anchor_records
        ),
        "hard_rescue_anchor_count": sum(
            bool(record["hard_rescue"]) for record in anchor_records
        ),
        "diagnostic_margin_group_equal_pair_rates": {
            f"{margin:.12g}": float(
                np.mean(
                    [float(group["margin_rates"][f"{margin:.12g}"]) for group in group_rows]
                )
            )
            for margin in margins
        },
        "minimum_groups_with_nonzero_pair": protocol.minimum_groups_with_nonzero_pair,
        "minimum_groups_with_strict_native_improvement": (
            protocol.minimum_groups_with_strict_native_improvement
        ),
        "oracle_gain_bootstrap_lower_must_be_strictly_positive": True,
        "group_summaries": group_rows,
        "decision": decision,
    }
