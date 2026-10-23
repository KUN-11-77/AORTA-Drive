"""Group-level decision rule for the R004E-F field-signal diagnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FieldSignalDecision:
    """Frozen interpretation of independent recording-group event counts."""

    minimum_support_groups: int
    tail_objective_mismatch_supported: bool
    global_sign_inversion_supported: bool
    no_transferable_signal_supported: bool
    verdict: str


def decide_field_signal(
    counts: Mapping[str, int],
    *,
    group_count: int = 24,
    support_fraction: float = 0.80,
) -> FieldSignalDecision:
    """Apply the preregistered group-level rule and frozen precedence."""

    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if not math.isfinite(support_fraction) or not 0.0 < support_fraction <= 1.0:
        raise ValueError("support_fraction must lie in (0, 1]")
    required = {
        "tail_objective_mismatch",
        "global_sign_inversion",
        "no_transferable_signal",
    }
    if set(counts) != required:
        raise ValueError("counts must contain exactly the registered rival explanations")
    invalid_counts = any(
        not isinstance(value, int) or not 0 <= value <= group_count
        for value in counts.values()
    )
    if invalid_counts:
        raise ValueError("all counts must be integer recording-group counts")
    minimum = math.ceil(group_count * support_fraction)
    tail = counts["tail_objective_mismatch"] >= minimum
    inversion = counts["global_sign_inversion"] >= minimum
    no_signal = counts["no_transferable_signal"] >= minimum
    if inversion:
        verdict = "global_sign_inversion"
    elif tail:
        verdict = "tail_objective_mismatch"
    elif no_signal:
        verdict = "no_transferable_visual_signal"
    else:
        verdict = "mixed_or_partial_signal"
    return FieldSignalDecision(
        minimum_support_groups=minimum,
        tail_objective_mismatch_supported=tail,
        global_sign_inversion_supported=inversion,
        no_transferable_signal_supported=no_signal,
        verdict=verdict,
    )
