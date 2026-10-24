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

__all__ = [
    "DecisionComparison",
    "cut_space_projection",
    "decision_error_eta",
    "incidence_matrix",
    "recover_zero_mean_potential",
    "selection_comparison",
    "set_min_contrasts",
    "set_min_oscillation_bounds",
]
