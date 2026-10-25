from __future__ import annotations

import copy

import pytest

from audit_r005e2_mismatch_manifest import _audit_stage


GROUPS = {
    "a::00": "a",
    "b::00": "b",
    "c::00": "c",
}


def _stage() -> dict[str, object]:
    return {
        "rows": [
            {
                "recipient": "a::00",
                "donor": "b::00",
                "recipient_recording_group": "a",
                "donor_recording_group": "b",
                "standardized_squared_distance": 0.2,
            },
            {
                "recipient": "b::00",
                "donor": "c::00",
                "recipient_recording_group": "b",
                "donor_recording_group": "c",
                "standardized_squared_distance": 0.3,
            },
            {
                "recipient": "c::00",
                "donor": "a::00",
                "recipient_recording_group": "c",
                "donor_recording_group": "a",
                "standardized_squared_distance": 0.4,
            },
        ]
    }


def test_independent_stage_audit_accepts_exact_cross_group_permutation() -> None:
    audit = _audit_stage(
        _stage(),
        expected_recipients=set(GROUPS),
        allowed_donors=set(GROUPS),
        group_by_key=GROUPS,
        require_permutation=True,
    )
    assert audit["recipient_count"] == 3
    assert audit["same_group_count"] == 0
    assert audit["donor_marginal_exact_permutation"] is True


@pytest.mark.parametrize("mutation", ["same_group", "reuse", "outside", "identity", "nan"])
def test_independent_stage_audit_fails_closed(mutation: str) -> None:
    stage = copy.deepcopy(_stage())
    rows = stage["rows"]
    assert isinstance(rows, list)
    if mutation == "same_group":
        rows[0]["donor"] = "a::00"
        rows[0]["donor_recording_group"] = "a"
    elif mutation == "reuse":
        rows[1]["donor"] = rows[0]["donor"]
    elif mutation == "outside":
        rows[0]["donor"] = "outside::00"
    elif mutation == "identity":
        rows[0]["recipient_recording_group"] = "wrong"
    else:
        rows[0]["standardized_squared_distance"] = float("nan")
    with pytest.raises((ValueError, KeyError)):
        _audit_stage(
            stage,
            expected_recipients=set(GROUPS),
            allowed_donors=set(GROUPS),
            group_by_key=GROUPS,
            require_permutation=True,
        )
