"""Tests for the H7BS4A outcome-free visual observability gate."""

from __future__ import annotations

import numpy as np
import pytest

from aorta_drive.evaluation.visual_observability import (
    candidate_observability,
    summarize_group_observability,
)


def test_candidate_observability_separates_forward_extension() -> None:
    metric = candidate_observability(
        np.asarray([0, 2, 3]),
        np.asarray([True, False, True, False], dtype=np.bool_),
        np.asarray([-1.0, 10.0, 32.5, 48.5]),
    )
    assert metric["supported_query_node_count"] == 2
    assert metric["supported_query_fraction"] == pytest.approx(2 / 3)
    assert metric["forward_extension_query_node_count"] == 2
    assert metric["forward_extension_supported_node_count"] == 1


def test_group_summary_does_not_treat_nodes_as_independent() -> None:
    rows = []
    for group in ("g0", "g1"):
        for anchor in range(2):
            rows.append(
                {
                    "recording_group": group,
                    "anchor_index": anchor,
                    "all_query_candidates_have_direct_evidence": True,
                    "candidate_observability": [
                        {
                            "has_any_direct_projective_evidence": True,
                            "supported_query_fraction": 0.5,
                        }
                    ],
                }
            )
    result = summarize_group_observability(
        rows, group_names=("g0", "g1"), anchors_per_group=2, minimum_anchors_per_group=2
    )
    assert result["decision"] == "FIT_PROJECTIVE_PRECONDITION_PASS"
    assert result["group_count"] == 2
    assert result["query_candidate_count"] == 4


def test_zero_evidence_candidate_fails_even_if_anchor_gate_is_loose() -> None:
    rows = [
        {
            "recording_group": "g0",
            "anchor_index": 0,
            "all_query_candidates_have_direct_evidence": False,
            "candidate_observability": [
                {
                    "has_any_direct_projective_evidence": False,
                    "supported_query_fraction": 0.0,
                }
            ],
        }
    ]
    result = summarize_group_observability(
        rows, group_names=("g0",), anchors_per_group=1, minimum_anchors_per_group=0
    )
    assert result["decision"] == "STOP_OR_REDESIGN_VISUAL_SUPPORT_INTERFACE"


def test_candidate_observability_rejects_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="unique"):
        candidate_observability(
            np.asarray([0, 0]),
            np.asarray([True], dtype=np.bool_),
            np.asarray([0.5]),
        )
