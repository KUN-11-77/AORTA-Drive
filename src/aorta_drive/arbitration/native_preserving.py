"""Reference native-preserving arbitration over joint margin bounds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


class DecisionReason(str, Enum):
    """Auditable outcome categories for selective arbitration."""

    KEEP_NATIVE_NOT_PROVEN_UNSAFE = "keep_native_not_proven_unsafe"
    REPLACE_WITH_CERTIFIED_CANDIDATE = "replace_with_certified_candidate"
    KEEP_NATIVE_NO_CERTIFIED_ALTERNATIVE = "keep_native_no_certified_alternative"


@dataclass(frozen=True)
class ArbitrationDecision:
    """Selected candidate and the reason for the decision."""

    selected_index: int
    intervened: bool
    reason: DecisionReason


def arbitrate(
    lower_bounds: npt.ArrayLike,
    upper_bounds: npt.ArrayLike,
    *,
    native_index: int,
    preference_order: Sequence[int],
    discretization_margin: float = 0.0,
) -> ArbitrationDecision:
    """Apply the conservative native-preserving arbitration rule.

    The native trajectory is changed only when its upper margin bound is below
    zero and at least one alternative has a lower bound above the discretization
    margin. ``preference_order`` is supplied by the frozen planner and is used
    only after physical feasibility has been established.
    """

    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if lower.ndim != 1 or upper.shape != lower.shape or lower.size == 0:
        raise ValueError("lower_bounds and upper_bounds must be non-empty aligned vectors")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("bounds must be finite")
    if np.any(lower > upper):
        raise ValueError("each lower bound must not exceed its upper bound")
    if not 0 <= native_index < lower.size:
        raise ValueError("native_index is out of range")
    if discretization_margin < 0:
        raise ValueError("discretization_margin must be non-negative")
    if sorted(preference_order) != list(range(lower.size)):
        raise ValueError("preference_order must be a permutation of all candidate indices")

    if upper[native_index] >= 0:
        return ArbitrationDecision(
            selected_index=native_index,
            intervened=False,
            reason=DecisionReason.KEEP_NATIVE_NOT_PROVEN_UNSAFE,
        )

    for index in preference_order:
        if index != native_index and lower[index] >= discretization_margin:
            return ArbitrationDecision(
                selected_index=index,
                intervened=True,
                reason=DecisionReason.REPLACE_WITH_CERTIFIED_CANDIDATE,
            )

    return ArbitrationDecision(
        selected_index=native_index,
        intervened=False,
        reason=DecisionReason.KEEP_NATIVE_NO_CERTIFIED_ALTERNATIVE,
    )
