"""Mathematical primitives used by registered AORTA theory gates."""

from aorta_drive.theory.decision_quotient import (
    DecisionComparison,
    cut_space_projection,
    decision_error_eta,
    incidence_matrix,
    recover_zero_mean_potential,
    selection_comparison,
    set_min_contrasts,
    set_min_oscillation_bounds,
)
from aorta_drive.theory.relation_guarantees import (
    OFFICIAL_CURRENT_WEIGHT,
    OFFICIAL_TEMPORAL_WEIGHT,
    ONE_FRAME_SIGN_CERTIFICATE_THRESHOLD,
    RelationDecisionReason,
    SelectiveRelationDecision,
    arbitrate_native_relative,
    compose_full_pair_margin,
    group_simultaneous_nonconformity,
    group_split_quantile_or_infinity,
    one_frame_sign_is_full_certified,
)

__all__ = [
    "DecisionComparison",
    "OFFICIAL_CURRENT_WEIGHT",
    "OFFICIAL_TEMPORAL_WEIGHT",
    "ONE_FRAME_SIGN_CERTIFICATE_THRESHOLD",
    "RelationDecisionReason",
    "SelectiveRelationDecision",
    "arbitrate_native_relative",
    "compose_full_pair_margin",
    "cut_space_projection",
    "decision_error_eta",
    "incidence_matrix",
    "group_simultaneous_nonconformity",
    "group_split_quantile_or_infinity",
    "one_frame_sign_is_full_certified",
    "recover_zero_mean_potential",
    "selection_comparison",
    "set_min_contrasts",
    "set_min_oscillation_bounds",
]
