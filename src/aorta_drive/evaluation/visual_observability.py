"""Outcome-free observability summaries for frozen candidate query sets.

Projective token support is an upper bound on usable visual evidence: it does
not model occlusion or prove that a network extracts the required road fact.
The functions here deliberately keep node and candidate repeats nested inside
their recording group instead of presenting them as independent samples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


def candidate_observability(
    node_indices: npt.ArrayLike,
    node_has_evidence: npt.ArrayLike,
    node_x_m: npt.ArrayLike,
    *,
    forward_extension_start_m: float = 32.0,
) -> dict[str, int | float | bool]:
    """Summarize direct projective support for one discrete candidate query."""

    indices = np.asarray(node_indices, dtype=np.int64)
    evidence = np.asarray(node_has_evidence)
    x = np.asarray(node_x_m, dtype=np.float64)
    if indices.ndim != 1 or indices.size == 0 or len(np.unique(indices)) != indices.size:
        raise ValueError("node_indices must be a nonempty unique vector")
    if evidence.ndim != 1 or evidence.dtype != np.bool_ or x.shape != evidence.shape:
        raise ValueError("evidence and node_x_m must be aligned one-dimensional arrays")
    if np.any(indices < 0) or np.any(indices >= evidence.size):
        raise ValueError("candidate node index lies outside the field")
    if not np.all(np.isfinite(x)) or not np.isfinite(forward_extension_start_m):
        raise ValueError("node coordinates and extension boundary must be finite")
    selected = evidence[indices]
    extension = x[indices] >= forward_extension_start_m
    extension_count = int(np.count_nonzero(extension))
    extension_supported = int(np.count_nonzero(selected & extension))
    supported = int(np.count_nonzero(selected))
    return {
        "query_node_count": int(indices.size),
        "supported_query_node_count": supported,
        "supported_query_fraction": supported / int(indices.size),
        "has_any_direct_projective_evidence": supported > 0,
        "all_query_nodes_directly_supported": supported == int(indices.size),
        "forward_extension_query_node_count": extension_count,
        "forward_extension_supported_node_count": extension_supported,
        "forward_extension_supported_fraction": (
            extension_supported / extension_count if extension_count else 1.0
        ),
    }


def summarize_group_observability(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_names: Sequence[str],
    anchors_per_group: int,
    minimum_anchors_per_group: int,
) -> dict[str, Any]:
    """Apply the frozen minimal direct-evidence precondition at group level."""

    groups = tuple(group_names)
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("group_names must be nonempty and unique")
    expected = {(group, anchor) for group in groups for anchor in range(anchors_per_group)}
    observed = {
        (str(row["recording_group"]), int(row["anchor_index"])) for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("rows do not exactly cover the frozen group panel")
    per_group: dict[str, Any] = {}
    passing_groups = 0
    zero_evidence_candidates = 0
    candidate_count = 0
    for group in groups:
        group_rows = [row for row in rows if row["recording_group"] == group]
        qualified = sum(bool(row["all_query_candidates_have_direct_evidence"]) for row in group_rows)
        candidate_metrics = [
            metric
            for row in group_rows
            for metric in row["candidate_observability"]
        ]
        group_zero = sum(
            not bool(metric["has_any_direct_projective_evidence"])
            for metric in candidate_metrics
        )
        group_candidates = len(candidate_metrics)
        passing = qualified >= minimum_anchors_per_group
        passing_groups += int(passing)
        zero_evidence_candidates += group_zero
        candidate_count += group_candidates
        per_group[group] = {
            "anchors_whose_all_query_candidates_have_direct_evidence": qualified,
            "minimum_required": minimum_anchors_per_group,
            "passes": passing,
            "query_candidate_count": group_candidates,
            "zero_direct_evidence_candidate_count": group_zero,
            "mean_candidate_supported_query_fraction": float(
                np.mean([metric["supported_query_fraction"] for metric in candidate_metrics])
            ),
        }
    strict_pass = zero_evidence_candidates == 0 and passing_groups == len(groups)
    return {
        "group_count": len(groups),
        "anchor_count": len(rows),
        "query_candidate_count": candidate_count,
        "zero_direct_evidence_candidate_count": zero_evidence_candidates,
        "groups_meeting_minimum_direct_evidence_anchors": passing_groups,
        "minimum_direct_evidence_anchors_per_group": minimum_anchors_per_group,
        "per_group": per_group,
        "decision": (
            "FIT_PROJECTIVE_PRECONDITION_PASS"
            if strict_pass
            else "STOP_OR_REDESIGN_VISUAL_SUPPORT_INTERFACE"
        ),
    }
