"""Pure diagnostics for R004E-E excess graph-gradient residual localization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


EDGE_BOTH_OBSERVED = 0
EDGE_MIXED_MISSING = 1
EDGE_BOTH_MISSING = 2
EDGE_CATEGORY_NAMES = (
    "both_observed",
    "mixed_missing",
    "both_missing",
)


def oriented_edge_endpoint_sums(node_values: Tensor) -> Tensor:
    """Sum endpoint values in the same right-then-down order as graph gradients."""

    if node_values.ndim != 2 or node_values.shape[0] != node_values.shape[1]:
        raise ValueError("node_values must have square shape [height, width]")
    horizontal = node_values[:, :-1] + node_values[:, 1:]
    vertical = node_values[:-1, :] + node_values[1:, :]
    return torch.cat((horizontal.reshape(-1), vertical.reshape(-1)))


def oriented_edge_evidence_categories(node_has_evidence: Tensor) -> Tensor:
    """Classify grid edges by whether zero, one, or two endpoints have evidence."""

    if (
        node_has_evidence.dtype is not torch.bool
        or node_has_evidence.ndim != 2
        or node_has_evidence.shape[0] != node_has_evidence.shape[1]
    ):
        raise ValueError("node_has_evidence must be a square Boolean matrix")
    endpoint_counts = oriented_edge_endpoint_sums(node_has_evidence.to(torch.int64))
    categories = torch.empty_like(endpoint_counts)
    categories[endpoint_counts == 2] = EDGE_BOTH_OBSERVED
    categories[endpoint_counts == 1] = EDGE_MIXED_MISSING
    categories[endpoint_counts == 0] = EDGE_BOTH_MISSING
    return categories


@dataclass(frozen=True)
class AttributionDecision:
    """Group-level support decisions for the three preregistered explanations."""

    minimum_support_groups: int
    gradient_overshoot_supported: bool
    missing_evidence_supported: bool
    observed_nonovershoot_supported: bool
    recommendation: str


def decide_attribution(
    counts: Mapping[str, int],
    *,
    group_count: int = 24,
    support_fraction: float = 0.80,
) -> AttributionDecision:
    """Apply frozen group-level thresholds without treating nested edges as replicates."""

    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if not math.isfinite(support_fraction) or not 0.0 < support_fraction <= 1.0:
        raise ValueError("support_fraction must lie in (0, 1]")
    required = {
        "gradient_overshoot",
        "missing_evidence_incident",
        "fully_observed_nonovershoot",
    }
    if set(counts) != required:
        raise ValueError("counts must contain exactly the three registered explanations")
    invalid_counts = any(
        not isinstance(value, int) or not 0 <= value <= group_count
        for value in counts.values()
    )
    if invalid_counts:
        raise ValueError("all attribution counts must be integer group counts")
    minimum = math.ceil(group_count * support_fraction)
    overshoot = counts["gradient_overshoot"] >= minimum
    missing = counts["missing_evidence_incident"] >= minimum
    observed = counts["fully_observed_nonovershoot"] >= minimum
    if overshoot and missing:
        recommendation = "factorial_projection_x_observable_subgraph_control"
    elif overshoot:
        recommendation = "registered_graph_lipschitz_projection_control"
    elif missing:
        recommendation = "formalize_fixed_observable_qualified_subgraph"
    elif observed:
        recommendation = "redesign_visual_representation_or_objective"
    else:
        recommendation = "mixed_mechanism_factorial_controls_before_redesign"
    return AttributionDecision(
        minimum_support_groups=minimum,
        gradient_overshoot_supported=overshoot,
        missing_evidence_supported=missing,
        observed_nonovershoot_supported=observed,
        recommendation=recommendation,
    )
