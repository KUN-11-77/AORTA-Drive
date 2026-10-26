from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from run_r005e3a_fold_job import TIE_TOLERANCE, _temperature_from_outer_train
from audit_r005e3a_run import _bootstrap, _classify


def _temperature_fixture(
    root: Path,
    target: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
) -> dict[str, Any]:
    np.save(root / "target_margin.npy", target.astype(np.float32))
    np.save(root / "candidate_mask.npy", mask.astype(np.bool_))
    anchors = [
        {"bundle_index": index, "recording_group": "g0"}
        for index in range(target.shape[0])
    ]
    return {"anchors": anchors}


def test_temperature_uses_only_positive_outer_train_regret_gaps(
    tmp_path: Path,
) -> None:
    target = np.zeros((16, 4), dtype=np.float32)
    mask = np.ones_like(target, dtype=np.bool_)
    target[:, 1:] = np.asarray([0.2, 0.6, 0.6], dtype=np.float32)
    manifest = _temperature_fixture(tmp_path, target, mask)
    temperature, ledger = _temperature_from_outer_train(
        bundle_root=tmp_path, bundle_manifest=manifest, groups=["g0"]
    )
    # Per anchor the positive gaps are 0.6 and 0.4; exact best ties are removed.
    assert temperature == pytest.approx(0.5)
    assert ledger["gap_count"] == 32
    assert ledger["tie_tolerance"] == TIE_TOLERANCE


def test_temperature_uses_midpoint_median_for_even_gap_count(tmp_path: Path) -> None:
    target = np.zeros((16, 2), dtype=np.float32)
    mask = np.ones_like(target, dtype=np.bool_)
    target[:8, 1] = 0.2
    target[8:, 1] = 0.8
    manifest = _temperature_fixture(tmp_path, target, mask)
    temperature, ledger = _temperature_from_outer_train(
        bundle_root=tmp_path, bundle_manifest=manifest, groups=["g0"]
    )
    assert temperature == pytest.approx(0.5)
    assert ledger["even_count_median"] == (
        "arithmetic_mean_of_two_central_order_statistics"
    )


def test_temperature_fails_closed_when_all_candidates_tie(tmp_path: Path) -> None:
    target = np.zeros((16, 3), dtype=np.float32)
    mask = np.ones_like(target, dtype=np.bool_)
    manifest = _temperature_fixture(tmp_path, target, mask)
    with pytest.raises(ValueError, match="no strictly positive"):
        _temperature_from_outer_train(
            bundle_root=tmp_path, bundle_manifest=manifest, groups=["g0"]
        )


@pytest.mark.parametrize(
    ("objective", "effect", "binding", "expected"),
    [
        (
            True,
            True,
            True,
            "GO_TO_FULL_OBJECTIVE_BY_VISUAL_ARM_CONFIRMATION_PREREGISTRATION",
        ),
        (
            True,
            True,
            False,
            "MIXED_DECISION_IMPROVEMENT_WITHOUT_COMPLETE_SCENE_BINDING",
        ),
        (False, True, True, "STOP_SOFT_SELECTION_REGRET_OBJECTIVE"),
        (True, False, True, "STOP_SOFT_SELECTION_REGRET_OBJECTIVE"),
    ],
)
def test_frozen_classification_is_conjunctive(
    objective: bool, effect: bool, binding: bool, expected: str
) -> None:
    assert (
        _classify(
            objective_supported=objective,
            minimum_effect_supported=effect,
            binding_supported=binding,
        )
        == expected
    )


def test_group_bootstrap_requires_exactly_42_independent_units() -> None:
    values = np.arange(42, dtype=np.float64)
    first = _bootstrap(values, seed=20261026, replicates=100)
    second = _bootstrap(values, seed=20261026, replicates=100)
    assert first == second
    with pytest.raises(ValueError, match="42"):
        _bootstrap(values[:-1], seed=20261026, replicates=100)
