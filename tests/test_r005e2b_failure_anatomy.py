from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_r005e2b_failure_anatomy.py"
SPEC = importlib.util.spec_from_file_location("r005e2b_failure_anatomy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_choice_uses_lowest_registered_index_on_tie() -> None:
    prediction = np.asarray([0.0, 0.2, 0.2, -0.1], dtype=np.float64)
    mask = np.asarray([True, True, True, False])
    assert MODULE._choice(prediction, mask) == 1


def test_pair_error_includes_challenger_to_challenger_order() -> None:
    target = np.asarray([0.0, 0.4, 0.2], dtype=np.float64)
    prediction = np.asarray([0.0, 0.1, 0.3], dtype=np.float64)
    mask = np.asarray([True, True, True])
    errors, count = MODULE._pair_error(prediction, target, mask)
    assert count == 3
    assert errors == 1


def test_choice_change_separates_benefit_harm_and_neutral() -> None:
    target = np.asarray(
        [[0.0, 0.5, -0.5], [0.0, 0.5, -0.5], [0.0, 0.5, -0.5]],
        dtype=np.float64,
    )
    mask = np.ones_like(target, dtype=np.bool_)
    reference = np.asarray(
        [[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.2, 0.1]],
        dtype=np.float64,
    )
    alternative = np.asarray(
        [[0.0, 0.2, 0.1], [0.0, 0.2, 0.1], [0.0, 0.2, 0.1]],
        dtype=np.float64,
    )
    summary = MODULE._choice_change(reference, alternative, target, mask)
    assert summary["beneficial_fraction"] == 2 / 3
    assert summary["harmful_fraction"] == 0.0
    assert summary["neutral_fraction"] == 1 / 3


def test_difficulty_bins_are_fixed() -> None:
    assert MODULE._difficulty_bin(0.01) == "top2_gap_le_0p01"
    assert MODULE._difficulty_bin(0.05) == "top2_gap_0p01_to_0p05"
    assert MODULE._difficulty_bin(0.0500001) == "top2_gap_gt_0p05"
