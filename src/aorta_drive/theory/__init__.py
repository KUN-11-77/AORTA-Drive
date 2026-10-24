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
    RelationDecisionReason,
    SelectiveRelationDecision,
    arbitrate_native_relative,
    group_simultaneous_nonconformity,
    group_split_quantile_or_infinity,
)

__all__ = [
    "DecisionComparison",
    "RelationDecisionReason",
    "SelectiveRelationDecision",
    "arbitrate_native_relative",
    "cut_space_projection",
    "decision_error_eta",
    "incidence_matrix",
    "group_simultaneous_nonconformity",
    "group_split_quantile_or_infinity",
    "recover_zero_mean_potential",
    "selection_comparison",
    "set_min_contrasts",
    "set_min_oscillation_bounds",
]
