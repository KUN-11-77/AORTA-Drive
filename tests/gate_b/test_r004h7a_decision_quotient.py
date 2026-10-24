from __future__ import annotations

import itertools

import numpy as np

from aorta_drive.theory.decision_quotient import (
    cut_space_projection,
    decision_error_eta,
    incidence_matrix,
    recover_zero_mean_potential,
    selection_comparison,
    set_min_contrasts,
    set_min_oscillation_bounds,
)


def _nonempty_subsets(node_count: int) -> list[tuple[int, ...]]:
    return [
        subset
        for size in range(1, node_count + 1)
        for subset in itertools.combinations(range(node_count), size)
    ]


def test_common_offset_invariance_and_oscillation_bound_exhaustive() -> None:
    potentials = [
        np.asarray(values, dtype=np.float64)
        for values in itertools.product((-1.0, 0.0, 1.0), repeat=3)
    ]
    subsets = _nonempty_subsets(3)
    for first in potentials:
        for native in subsets:
            for challenger in subsets:
                panel = (native, challenger)
                base = set_min_contrasts(first, panel)
                for offset in (-2.0, 0.5, 3.0):
                    assert np.array_equal(base, set_min_contrasts(first + offset, panel))
                for second in potentials:
                    error = np.abs(
                        set_min_contrasts(first, panel)
                        - set_min_contrasts(second, panel)
                    )
                    bound = set_min_oscillation_bounds(first, second, panel)
                    assert np.all(error <= np.nextafter(bound, np.inf))


def test_two_eta_regret_bound_exhaustive() -> None:
    potentials = [
        np.asarray(values, dtype=np.float64)
        for values in itertools.product((-1.0, 0.0, 1.0), repeat=3)
    ]
    subsets = _nonempty_subsets(3)
    for teacher in potentials:
        for predicted in potentials:
            for native in subsets:
                for first_challenger in subsets:
                    for second_challenger in subsets:
                        panel = (native, first_challenger, second_challenger)
                        teacher_margin = set_min_contrasts(teacher, panel)
                        predicted_margin = set_min_contrasts(predicted, panel)
                        comparison = selection_comparison(teacher_margin, predicted_margin)
                        eta = decision_error_eta(teacher_margin, predicted_margin)
                        assert comparison.regret <= np.nextafter(2.0 * eta, np.inf)


def test_potential_and_cut_space_projection_are_equivalent() -> None:
    incidence = incidence_matrix(3, ((0, 1), (1, 2), (0, 2)))
    for values in itertools.product((-1.0, 0.0, 1.0), repeat=3):
        potential = np.asarray(values, dtype=np.float64)
        recovered = recover_zero_mean_potential(incidence @ potential, incidence)
        expected = potential - np.mean(potential)
        assert np.allclose(recovered, expected, rtol=0.0, atol=1e-12)
        assert np.allclose(incidence @ recovered, incidence @ potential, rtol=0.0, atol=1e-12)

    raw_edge = np.asarray((1.0, -2.0, 0.25), dtype=np.float64)
    projected = cut_space_projection(raw_edge, incidence)
    residual = raw_edge - projected
    assert np.allclose(incidence.T @ residual, 0.0, rtol=0.0, atol=1e-12)


def test_global_set_min_query_is_not_linear() -> None:
    panel = ((0,), (1, 2))
    first = np.asarray((0.0, 0.0, 1.0))
    second = np.asarray((0.0, 1.0, 0.0))
    assert set_min_contrasts(first, panel)[0] == 0.0
    assert set_min_contrasts(second, panel)[0] == 0.0
    assert set_min_contrasts(first + second, panel)[0] == 1.0


def test_tie_policy_is_native_first_then_candidate_order() -> None:
    native_tie = selection_comparison((0.0, -1.0), (0.0, -1.0))
    assert native_tie.teacher_index == 0
    assert native_tie.predicted_index == 0
    assert native_tie.exact_teacher_tie

    challenger_tie = selection_comparison((1.0, 1.0), (1.0, 1.0))
    assert challenger_tie.teacher_index == 1
    assert challenger_tie.predicted_index == 1
    assert challenger_tie.exact_teacher_tie


def test_invalid_panels_and_disconnected_graphs_fail_closed() -> None:
    with np.testing.assert_raises(ValueError):
        set_min_contrasts((0.0, 1.0), ((0,),))
    with np.testing.assert_raises(ValueError):
        set_min_contrasts((0.0, 1.0), ((0,), ()))
    with np.testing.assert_raises(ValueError):
        set_min_contrasts((0.0, 1.0), ((0,), (2,)))
    with np.testing.assert_raises(ValueError):
        incidence_matrix(3, ((0, 1),))
