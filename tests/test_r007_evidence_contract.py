from __future__ import annotations

from dataclasses import replace

import pytest

from aorta_drive.vision.r007_evidence_contract import (
    CONTEXT_TIMING_CELLS,
    WRITER_TARGET_ARMS,
    ContextTimingCell,
    MatchedBudget,
    WriterTargetArm,
    context_timing_interaction,
    validate_writer_target_arms,
    writer_target_contrast,
)


def _budget() -> MatchedBudget:
    return MatchedBudget("data", "cut", 16, 1024, 4096, 100, 3)


def test_context_timing_interaction_uses_preregistered_direction() -> None:
    risks = (0.50, 0.45, 0.50, 0.30)
    cells = {
        name: ContextTimingCell(risk, _budget())
        for name, risk in zip(CONTEXT_TIMING_CELLS, risks, strict=True)
    }
    assert context_timing_interaction(cells) == pytest.approx(0.15)


def test_context_timing_rejects_missing_nonfinite_or_unmatched_cells() -> None:
    valid = {
        name: ContextTimingCell(0.5, _budget()) for name in CONTEXT_TIMING_CELLS
    }
    with pytest.raises(ValueError):
        context_timing_interaction(dict(list(valid.items())[:-1]))
    invalid = dict(valid)
    invalid[CONTEXT_TIMING_CELLS[0]] = ContextTimingCell(float("nan"), _budget())
    with pytest.raises(ValueError):
        context_timing_interaction(invalid)
    unmatched = dict(valid)
    unmatched[CONTEXT_TIMING_CELLS[0]] = ContextTimingCell(
        0.5, replace(_budget(), parameter_count=2048)
    )
    with pytest.raises(ValueError):
        context_timing_interaction(unmatched)


def test_writer_target_contract_requires_all_matched_arms() -> None:
    arms = {
        name: WriterTargetArm(0.1, 0.2, 0.3, _budget())
        for name in WRITER_TARGET_ARMS
    }
    assert validate_writer_target_arms(arms) == _budget()
    contrast = writer_target_contrast(
        arms,
        left="native_hidden_state_mse",
        right="task_supervision",
    )
    assert contrast.feature_error_delta == pytest.approx(0.0)
    assert contrast.tail_output_error_delta == pytest.approx(0.0)
    assert contrast.task_utility_delta == pytest.approx(0.0)


def test_writer_target_contract_rejects_budget_and_value_failures() -> None:
    arms = {
        name: WriterTargetArm(0.1, 0.2, 0.3, _budget())
        for name in WRITER_TARGET_ARMS
    }
    arms[WRITER_TARGET_ARMS[1]] = replace(
        arms[WRITER_TARGET_ARMS[1]], budget=replace(_budget(), search_trials=4)
    )
    with pytest.raises(ValueError):
        validate_writer_target_arms(arms)
    arms[WRITER_TARGET_ARMS[1]] = WriterTargetArm(
        0.1, float("inf"), 0.3, _budget()
    )
    with pytest.raises(ValueError):
        validate_writer_target_arms(arms)
