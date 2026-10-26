from __future__ import annotations

import pytest

from audit_r005e2s_run import SCHEMES, _classify
from run_r005e2s_fold_job import _sensitivity_donor_map


def _manifest(rows: list[dict[str, str]]) -> dict[str, object]:
    filler = [
        {"recipient": f"group-r::{index:03d}", "donor": f"group-d::{index:03d}"}
        for index in range(112)
    ]
    return {
        "folds": [
            {
                "outer_fold": 0,
                "schemes": {
                    "S2_disjoint_geometry": {"outer_test": {"rows": rows}},
                    "S3_hash_random": {"outer_test": {"rows": filler}},
                },
            }
        ]
    }


def test_sensitivity_donor_map_requires_exact_unique_recipient_roster() -> None:
    rows = [
        {"recipient": f"recipient::{index:03d}", "donor": f"donor::{index:03d}"}
        for index in range(112)
    ]
    result = _sensitivity_donor_map(
        _manifest(rows), outer_fold=0, scheme="S2_disjoint_geometry"
    )
    assert len(result) == 112
    assert result["recipient::000"] == "donor::000"

    duplicated = list(rows)
    duplicated[-1] = dict(duplicated[0])
    with pytest.raises(ValueError, match="duplicate"):
        _sensitivity_donor_map(
            _manifest(duplicated),
            outer_fold=0,
            scheme="S2_disjoint_geometry",
        )


def test_secondary_classification_obeys_frozen_precedence() -> None:
    both = {scheme: True for scheme in SCHEMES}
    neither = {scheme: False for scheme in SCHEMES}
    one = {SCHEMES[0]: True, SCHEMES[1]: False}

    assert _classify(neither, both) == "CONSISTENT_DECISION_BINDING_SECONDARY"
    assert _classify(both, neither) == "RELATION_ONLY_BINDING_SECONDARY"
    assert _classify(one, neither) == "DONOR_SENSITIVE_OR_INCONSISTENT_SECONDARY"
    assert _classify(neither, neither) == "NO_CONSISTENT_SCENE_BINDING_SECONDARY"


def test_secondary_classification_does_not_upgrade_mixed_scheme_support() -> None:
    relation = {SCHEMES[0]: True, SCHEMES[1]: False}
    decision = {SCHEMES[0]: False, SCHEMES[1]: True}
    assert (
        _classify(relation, decision)
        == "DONOR_SENSITIVE_OR_INCONSISTENT_SECONDARY"
    )
