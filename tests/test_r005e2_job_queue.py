from __future__ import annotations

from run_r005e2_job_queue import _jobs


def test_frozen_job_roster_is_complete_and_unique() -> None:
    jobs = _jobs()
    assert len(jobs) == 54
    assert len({str(job["name"]) for job in jobs}) == 54
    assert {int(job["outer_fold"]) for job in jobs} == set(range(6))
    assert {str(job["arm"]) for job in jobs} == {
        "V0_removed",
        "V1_mismatch",
        "V2_correct",
    }
    assert {int(job["seed"]) for job in jobs} == {20261025, 20261026, 20261027}
