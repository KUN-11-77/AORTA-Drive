"""Mechanical contracts for R007 task timing and writer-target audits.

The functions in this module validate experimental comparability and compute
predefined contrasts.  They do not choose thresholds or assign scientific
mechanism labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


CONTEXT_TIMING_CELLS = (
    "compressed_context_free",
    "compressed_task_conditioned",
    "raw_full_context_free",
    "raw_full_task_conditioned",
)
WRITER_TARGET_ARMS = (
    "native_hidden_state_mse",
    "final_feature_model_stitching",
    "frozen_tail_task_output_matching",
    "task_supervision",
)


@dataclass(frozen=True)
class MatchedBudget:
    """Frozen resources that must not explain an experimental contrast."""

    data_fingerprint: str
    writer_cut_fingerprint: str
    output_token_count: int
    parameter_count: int
    flops: int
    optimizer_steps: int
    search_trials: int


@dataclass(frozen=True)
class ContextTimingCell:
    """One risk value in the evidence x task-context timing factorial."""

    risk: float
    budget: MatchedBudget


@dataclass(frozen=True)
class WriterTargetArm:
    """Joint outcomes for one writer training objective."""

    hidden_feature_error: float
    frozen_tail_output_error: float
    task_utility: float
    budget: MatchedBudget


@dataclass(frozen=True)
class WriterTargetContrast:
    """Signed pairwise outcomes; no direction is declared beneficial here."""

    feature_error_delta: float
    tail_output_error_delta: float
    task_utility_delta: float


def _validate_budget(budget: MatchedBudget) -> None:
    if not budget.data_fingerprint or not budget.writer_cut_fingerprint:
        raise ValueError("data and writer-cut fingerprints must be non-empty")
    quantities = (
        budget.output_token_count,
        budget.parameter_count,
        budget.flops,
        budget.optimizer_steps,
        budget.search_trials,
    )
    if any(value < 1 for value in quantities):
        raise ValueError("all matched-budget quantities must be positive")


def _require_matched_budgets(budgets: list[MatchedBudget]) -> MatchedBudget:
    if not budgets:
        raise ValueError("at least one budget is required")
    for budget in budgets:
        _validate_budget(budget)
    reference = budgets[0]
    if any(budget != reference for budget in budgets[1:]):
        raise ValueError("experimental arms do not have an exactly matched budget")
    return reference


def context_timing_interaction(cells: Mapping[str, ContextTimingCell]) -> float:
    """Return the preregistered risk difference-in-differences.

    Positive values mean task conditioning reduced risk more when raw/full
    evidence was available than after compression.  Scientific support still
    requires the preregistered group-level uncertainty and outer confirmation.
    """

    if set(cells) != set(CONTEXT_TIMING_CELLS):
        missing = sorted(set(CONTEXT_TIMING_CELLS) - set(cells))
        extra = sorted(set(cells) - set(CONTEXT_TIMING_CELLS))
        raise ValueError(f"context timing cells disagree: missing={missing}, extra={extra}")
    _require_matched_budgets([cells[name].budget for name in CONTEXT_TIMING_CELLS])
    risks = np.asarray([cells[name].risk for name in CONTEXT_TIMING_CELLS], dtype=np.float64)
    if not np.isfinite(risks).all():
        raise ValueError("context timing risks must be finite")
    compressed_effect = risks[0] - risks[1]
    raw_effect = risks[2] - risks[3]
    return float(raw_effect - compressed_effect)


def validate_writer_target_arms(
    arms: Mapping[str, WriterTargetArm],
) -> MatchedBudget:
    """Require all writer objectives to share one auditable experimental budget."""

    if set(arms) != set(WRITER_TARGET_ARMS):
        missing = sorted(set(WRITER_TARGET_ARMS) - set(arms))
        extra = sorted(set(arms) - set(WRITER_TARGET_ARMS))
        raise ValueError(f"writer target arms disagree: missing={missing}, extra={extra}")
    budget = _require_matched_budgets([arms[name].budget for name in WRITER_TARGET_ARMS])
    outcomes = np.asarray(
        [
            value
            for name in WRITER_TARGET_ARMS
            for value in (
                arms[name].hidden_feature_error,
                arms[name].frozen_tail_output_error,
                arms[name].task_utility,
            )
        ],
        dtype=np.float64,
    )
    if not np.isfinite(outcomes).all():
        raise ValueError("writer target outcomes must be finite")
    return budget


def writer_target_contrast(
    arms: Mapping[str, WriterTargetArm], *, left: str, right: str
) -> WriterTargetContrast:
    """Compute a fixed signed comparison after validating all four objectives."""

    validate_writer_target_arms(arms)
    if left not in arms or right not in arms or left == right:
        raise ValueError("left and right must be distinct registered writer arms")
    lhs = arms[left]
    rhs = arms[right]
    return WriterTargetContrast(
        feature_error_delta=lhs.hidden_feature_error - rhs.hidden_feature_error,
        tail_output_error_delta=lhs.frozen_tail_output_error - rhs.frozen_tail_output_error,
        task_utility_delta=lhs.task_utility - rhs.task_utility,
    )
