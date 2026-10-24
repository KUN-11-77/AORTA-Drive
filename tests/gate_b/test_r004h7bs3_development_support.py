from __future__ import annotations

from run_r004h7bs3_development_support import summarize


def test_h7bs3_requires_all_candidates_and_group_coverage() -> None:
    rows = []
    for group in ("a", "b"):
        for anchor in range(2):
            rows.append(
                {
                    "recording_group": group,
                    "native_supported": True,
                    "eligible_challenger_count": 2,
                    "supported_eligible_challenger_count": 2,
                    "all_eligible_candidates_supported": True,
                    "panel_valid": True,
                }
            )
    passed = summarize(rows, groups=("a", "b"), minimum_valid=2)
    assert passed["decision"] == "DEVELOPMENT_FULL_DOMAIN_PASS"
    rows[0]["supported_eligible_challenger_count"] = 1
    rows[0]["all_eligible_candidates_supported"] = False
    failed = summarize(rows, groups=("a", "b"), minimum_valid=2)
    assert failed["decision"] == "STOP_FORWARD_96X64_DOMAIN"
    assert failed["anchors_with_any_candidate_support_failure"] == 1
