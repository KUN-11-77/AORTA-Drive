from __future__ import annotations

import pytest

from audit_r005e2_training_resources import _audit_timing_summary


def test_timing_audit_accepts_mean_above_p95_from_cold_start_outlier() -> None:
    _audit_timing_summary(
        {"mean": 0.922, "p95": 0.435, "maximum": 280.743}, "data_wait_ms"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"mean": 2.0, "p95": 3.0, "maximum": 1.0},
        {"mean": float("nan"), "p95": 1.0, "maximum": 2.0},
        {"mean": 1.0, "p95": 0.0, "maximum": 2.0},
    ],
)
def test_timing_audit_rejects_impossible_or_nonfinite_summaries(
    payload: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        _audit_timing_summary(payload, "timing")
