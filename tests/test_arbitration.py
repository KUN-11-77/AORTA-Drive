import pytest

from aorta_drive.arbitration.native_preserving import DecisionReason, arbitrate


def test_keeps_native_when_not_proven_unsafe() -> None:
    decision = arbitrate(
        lower_bounds=[-0.2, 0.4],
        upper_bounds=[0.1, 0.7],
        native_index=0,
        preference_order=[0, 1],
    )

    assert decision.selected_index == 0
    assert not decision.intervened
    assert decision.reason is DecisionReason.KEEP_NATIVE_NOT_PROVEN_UNSAFE


def test_replaces_only_with_certified_candidate() -> None:
    decision = arbitrate(
        lower_bounds=[-0.5, -0.1, 0.3],
        upper_bounds=[-0.2, 0.2, 0.6],
        native_index=0,
        preference_order=[0, 1, 2],
        discretization_margin=0.1,
    )

    assert decision.selected_index == 2
    assert decision.intervened
    assert decision.reason is DecisionReason.REPLACE_WITH_CERTIFIED_CANDIDATE


def test_rejects_malformed_bounds() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        arbitrate(
            lower_bounds=[0.2],
            upper_bounds=[0.1],
            native_index=0,
            preference_order=[0],
        )
