"""Graph certificates, calibration, and continuous correction primitives."""

from aorta_drive.certification.calibration import (
    minimum_calibration_sessions,
    session_max_score,
    split_conformal_threshold,
)
from aorta_drive.certification.difference_constraints import (
    DirectedConstraint,
    directed_single_source_distances,
    reverse_multi_source_envelope,
    robust_set_min_lower_bound_directed,
)
from aorta_drive.certification.geometry import (
    AxisAlignedRectangle,
    ContinuousCorrection,
    RectangleNetCertificate,
    certify_rectangle_internal_net,
)
from aorta_drive.certification.graph import (
    UndirectedEdge,
    multi_source_envelope,
    robust_set_min_lower_bound,
)
from aorta_drive.certification.lipschitz_oracle import (
    OracleLipschitzProblem,
    build_oracle_lipschitz_problem,
    independent_candidate_contrast_interval,
)
from aorta_drive.certification.risk import (
    authorization_mask,
    relative_margin_failure,
)

__all__ = [
    "AxisAlignedRectangle",
    "ContinuousCorrection",
    "DirectedConstraint",
    "OracleLipschitzProblem",
    "RectangleNetCertificate",
    "UndirectedEdge",
    "authorization_mask",
    "build_oracle_lipschitz_problem",
    "certify_rectangle_internal_net",
    "directed_single_source_distances",
    "minimum_calibration_sessions",
    "independent_candidate_contrast_interval",
    "multi_source_envelope",
    "relative_margin_failure",
    "reverse_multi_source_envelope",
    "robust_set_min_lower_bound",
    "robust_set_min_lower_bound_directed",
    "session_max_score",
    "split_conformal_threshold",
]
