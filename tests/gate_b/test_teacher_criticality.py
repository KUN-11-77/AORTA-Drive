from __future__ import annotations

import numpy as np
import pytest

from aorta_drive.evaluation.teacher_criticality import (
    critical_support,
    outward_node_intervals,
    possible_minimizer_indices,
)


def test_possible_minimizers_match_constructive_interval_oracle() -> None:
    lower = np.asarray([-2.0, -1.0, 0.1, 4.0])
    upper = np.asarray([1.0, 0.0, 2.0, 5.0])
    actual = possible_minimizer_indices([0, 1, 2], lower, upper)
    assert np.array_equal(actual, np.asarray([0, 1]))


def test_shared_low_upper_excludes_other_candidate_nodes() -> None:
    lower = np.asarray([-1.0, -0.5, -3.0])
    upper = np.asarray([2.0, 3.0, -2.0])
    actual = possible_minimizer_indices([0, 1, 2], lower, upper)
    assert np.array_equal(actual, np.asarray([2]))


def test_critical_support_preserves_unsupported_possible_minimizer() -> None:
    lower = np.asarray([-2.0, -1.0, 0.1])
    upper = np.asarray([1.0, 0.0, 2.0])
    result = critical_support([0, 1, 2], lower, upper, [True, False, True])
    assert result.possible_minimizer_indices == (0, 1)
    assert result.supported_possible_minimizer_indices == (0,)
    assert result.unsupported_possible_minimizer_indices == (1,)
    assert not result.all_possible_minimizers_supported


def test_outward_intervals_contain_exact_float_arithmetic_targets() -> None:
    values = np.asarray([0.1, -3.0, 7.5])
    errors = np.asarray([1e-12, 0.0, 0.25])
    lower, upper = outward_node_intervals(values, errors)
    assert np.all(lower <= values - errors)
    assert np.all(upper >= values + errors)
    assert np.all(lower <= upper)


def test_rejects_duplicate_or_empty_candidate_nodes() -> None:
    lower = np.asarray([0.0, 1.0])
    upper = np.asarray([0.0, 1.0])
    with pytest.raises(ValueError, match="non-empty"):
        possible_minimizer_indices([], lower, upper)
    with pytest.raises(ValueError, match="unique"):
        possible_minimizer_indices([0, 0], lower, upper)
