from __future__ import annotations

import numpy as np

from aorta_drive.evaluation.appearance_mechanism import (
    measure_field_mechanism,
    mechanism_condition_supported,
    oriented_grid_edges,
)


def test_edge_metrics_are_gauge_invariant_and_use_registered_order() -> None:
    field = np.asarray([[0.0, 1.0], [2.0, 4.0]])
    np.testing.assert_array_equal(oriented_grid_edges(field), [1.0, 2.0, 2.0, 3.0])
    teacher = field * 0.5
    metrics = measure_field_mechanism(field + 123.0, teacher - 91.0, field - 17.0)
    reference = measure_field_mechanism(field, teacher, field)
    assert metrics == reference


def test_condition_gate_requires_stability_task_signal_and_noncollapse() -> None:
    valid = dict(
        visual_task_mae_m=0.3,
        zero_task_mae_m=0.5,
        appearance_mae_m=0.1,
        clean_edge_energy_mae_m=0.2,
        teacher_cosine=0.4,
    )
    assert mechanism_condition_supported(**valid)
    for key, value in (
        ("visual_task_mae_m", 0.5),
        ("appearance_mae_m", 0.2),
        ("clean_edge_energy_mae_m", 0.0),
        ("teacher_cosine", 0.0),
    ):
        invalid = {**valid, key: value}
        assert not mechanism_condition_supported(**invalid)


def test_constant_field_cannot_support_the_mechanism() -> None:
    teacher = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    constant = np.zeros_like(teacher)
    metrics = measure_field_mechanism(constant, teacher, constant)
    assert metrics.edge_energy_mae_m == 0.0
    assert metrics.teacher_cosine == 0.0
    assert not mechanism_condition_supported(
        visual_task_mae_m=metrics.task_mae_m,
        zero_task_mae_m=metrics.task_mae_m,
        appearance_mae_m=metrics.appearance_mae_m,
        clean_edge_energy_mae_m=metrics.edge_energy_mae_m,
        teacher_cosine=metrics.teacher_cosine,
    )

