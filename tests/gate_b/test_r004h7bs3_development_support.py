from __future__ import annotations

from pathlib import Path

from run_r004h7bs3_development_support import _groups, summarize


def test_h7bs3_reads_24_development_groups_from_source_split() -> None:
    repository = Path(__file__).resolve().parents[2]
    groups = _groups(repository / "configs/manifests/r004_recording_group_split.csv")
    assert len(groups) == 24


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
