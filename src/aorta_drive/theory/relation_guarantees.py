"""Reference guarantees for native-relative decision relations.

The functions in this module are data-free contract primitives.  They do not
train a model, estimate OOD coverage, or claim driving safety.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

OFFICIAL_CURRENT_WEIGHT = 7.0 / 8.0
OFFICIAL_TEMPORAL_WEIGHT = 1.0 / 8.0
ONE_FRAME_SIGN_CERTIFICATE_THRESHOLD = 1.0 / 7.0


class RelationDecisionReason(str, Enum):
    """Auditable outcomes of native-relative lower-bound arbitration."""

    KEEP_NATIVE_NO_CHALLENGER = "keep_native_no_challenger"
    KEEP_NATIVE_INVALID_NUMERICS = "keep_native_invalid_numerics"
    KEEP_NATIVE_NO_POSITIVE_LOWER_BOUND = "keep_native_no_positive_lower_bound"
    REPLACE_WITH_POSITIVE_LOWER_BOUND = "replace_with_positive_lower_bound"


@dataclass(frozen=True)
class SelectiveRelationDecision:
    """Result over a legal ragged challenger vector.

    Candidate index zero is the implicit native. Challenger vector position
    ``j`` maps to candidate index ``j + 1``. Ties follow the frozen vector
    order through NumPy's first-occurrence ``argmax`` policy.
    """

    selected_index: int
    intervened: bool
    reason: RelationDecisionReason
    selected_lower_bound: float


def compose_full_pair_margin(
    one_frame_pair_margin: float,
    temporal_factor_difference: float,
) -> float:
    """Compose the adjacent-context official pair margin.

    ``temporal_factor_difference`` is the challenger-minus-reference difference
    of ``H * C2`` and must lie in ``[-1, 1]``.
    """

    if not np.isfinite(one_frame_pair_margin):
        raise ValueError("one_frame_pair_margin must be finite")
    if (
        not np.isfinite(temporal_factor_difference)
        or temporal_factor_difference < -1.0
        or temporal_factor_difference > 1.0
    ):
        raise ValueError("temporal_factor_difference must lie in [-1, 1]")
    return float(
        OFFICIAL_CURRENT_WEIGHT * one_frame_pair_margin
        + OFFICIAL_TEMPORAL_WEIGHT * temporal_factor_difference
    )


def one_frame_sign_is_full_certified(one_frame_pair_margin: float) -> bool:
    """Return whether the worst-case full-target sign is fixed and nonzero."""

    if not np.isfinite(one_frame_pair_margin):
        raise ValueError("one_frame_pair_margin must be finite")
    return abs(one_frame_pair_margin) > ONE_FRAME_SIGN_CERTIFICATE_THRESHOLD


def _aligned_vectors(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("inputs must be aligned one-dimensional vectors")
    return left, right


def arbitrate_native_relative(
    point_margins: npt.ArrayLike,
    radii: npt.ArrayLike,
) -> SelectiveRelationDecision:
    """Select only a challenger with a strictly positive improvement bound.

    ``point_margins`` contains challenger-minus-native task margins for the
    already-filtered legal candidate set.  Invalid numerical values fail closed
    to native.  Structural shape errors raise because no candidate mapping can
    be inferred safely.
    """

    point, radius = _aligned_vectors(point_margins, radii)
    if point.size == 0:
        return SelectiveRelationDecision(
            selected_index=0,
            intervened=False,
            reason=RelationDecisionReason.KEEP_NATIVE_NO_CHALLENGER,
            selected_lower_bound=float("-inf"),
        )
    if (
        not np.all(np.isfinite(point))
        or not np.all(np.isfinite(radius))
        or np.any(radius < 0.0)
    ):
        return SelectiveRelationDecision(
            selected_index=0,
            intervened=False,
            reason=RelationDecisionReason.KEEP_NATIVE_INVALID_NUMERICS,
            selected_lower_bound=float("-inf"),
        )

    lower = point - radius
    winner = int(np.argmax(lower))
    winner_lower = float(lower[winner])
    if winner_lower <= 0.0:
        return SelectiveRelationDecision(
            selected_index=0,
            intervened=False,
            reason=RelationDecisionReason.KEEP_NATIVE_NO_POSITIVE_LOWER_BOUND,
            selected_lower_bound=winner_lower,
        )
    return SelectiveRelationDecision(
        selected_index=winner + 1,
        intervened=True,
        reason=RelationDecisionReason.REPLACE_WITH_POSITIVE_LOWER_BOUND,
        selected_lower_bound=winner_lower,
    )


def group_simultaneous_nonconformity(
    predicted_margins: npt.ArrayLike,
    true_margins: npt.ArrayLike,
    scales: npt.ArrayLike,
) -> float:
    """Return one group's maximum standardized relation residual.

    All registered appearances, anchors, and legal challengers must already be
    present in the aligned arrays.  Their internal dependence is intentionally
    absorbed by the maximum rather than treated as independent samples.
    """

    predicted = np.asarray(predicted_margins, dtype=np.float64)
    truth = np.asarray(true_margins, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.shape != scale.shape:
        raise ValueError("predicted, true, and scale arrays must be aligned")
    if predicted.size == 0:
        raise ValueError("a calibration group must contain at least one comparison")
    if (
        not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise ValueError("calibration values must be finite and scales positive")
    return float(np.max(np.abs(predicted - truth) / scale))


def group_split_quantile_or_infinity(
    group_scores: npt.ArrayLike,
    alpha: float,
) -> float:
    """Return the finite-sample group quantile or infinity when impossible.

    The order statistic is ``ceil((n + 1) * (1 - alpha))``.  Returning infinity
    when that rank exceeds ``n`` makes the downstream rule abstain instead of
    silently weakening the registered coverage level.
    """

    scores = np.asarray(group_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("group_scores must be a non-empty finite vector")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = math.ceil((scores.size + 1) * (1.0 - alpha))
    if rank > scores.size:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])
