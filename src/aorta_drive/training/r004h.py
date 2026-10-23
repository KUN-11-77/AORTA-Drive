"""Frozen Stage-P selection rules for the R004H heterogeneous-envelope probe."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class StagePCheckpointCandidate:
    """Selection-group metrics for one completed Stage-P epoch."""

    epoch: int
    group_mean_edge_mae_m: tuple[float, ...]
    group_max_residual_m: tuple[float, ...]


@dataclass(frozen=True)
class StagePSelection:
    """Frozen lexicographic checkpoint key for Stage P."""

    epoch: int
    macro_group_mean_edge_mae_m: float
    worst_two_group_max_mean_m: float


def select_stage_p_checkpoint(
    candidates: Sequence[StagePCheckpointCandidate],
    *,
    expected_group_count: int = 8,
    tail_group_count: int = 2,
) -> StagePSelection:
    """Select mean visual signal first, then sparse tail, then earliest epoch."""

    if not candidates:
        raise ValueError("at least one Stage-P checkpoint candidate is required")
    if expected_group_count <= 0:
        raise ValueError("expected_group_count must be positive")
    if not 0 < tail_group_count <= expected_group_count:
        raise ValueError("tail_group_count must lie in [1, expected_group_count]")
    seen_epochs: set[int] = set()
    scored: list[StagePSelection] = []
    for candidate in candidates:
        if candidate.epoch <= 0 or candidate.epoch in seen_epochs:
            raise ValueError("checkpoint epochs must be unique positive integers")
        seen_epochs.add(candidate.epoch)
        means = candidate.group_mean_edge_mae_m
        maxima = candidate.group_max_residual_m
        if len(means) != expected_group_count or len(maxima) != expected_group_count:
            raise ValueError("every candidate must contain every selection group")
        if any(not math.isfinite(value) or value < 0.0 for value in (*means, *maxima)):
            raise ValueError("selection metrics must be finite and nonnegative")
        worst = sorted(maxima, reverse=True)[:tail_group_count]
        scored.append(
            StagePSelection(
                epoch=candidate.epoch,
                macro_group_mean_edge_mae_m=sum(means) / expected_group_count,
                worst_two_group_max_mean_m=sum(worst) / tail_group_count,
            )
        )
    return min(
        scored,
        key=lambda item: (
            item.macro_group_mean_edge_mae_m,
            item.worst_two_group_max_mean_m,
            item.epoch,
        ),
    )


def qualifies_stage_p_signal(*, learned_mean_m: float, zero_mean_m: float) -> bool:
    """Require strict selection-side mean-edge improvement over the zero field."""

    if any(
        not math.isfinite(value) or value < 0.0
        for value in (learned_mean_m, zero_mean_m)
    ):
        return False
    return learned_mean_m < zero_mean_m
