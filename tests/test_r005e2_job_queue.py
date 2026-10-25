from __future__ import annotations

import pytest

from run_r005e2_job_queue import _jobs, _slot_keys


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


def test_multi_job_gpu_slots_are_complete_and_unique() -> None:
    slots = _slot_keys([0, 1, 2, 3], 4)
    assert len(slots) == 16
    assert len(set(slots)) == 16
    assert {gpu for gpu, _ in slots} == {0, 1, 2, 3}
    assert {slot for _, slot in slots} == {0, 1, 2, 3}


def test_multi_job_gpu_slots_reject_nonpositive_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        _slot_keys([0], 0)
