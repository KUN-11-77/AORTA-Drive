"""Frozen Stage-P selection rules for the R004H heterogeneous-envelope probe."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor


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


@dataclass(frozen=True)
class StageSCheckpointCandidate:
    """Selection-group proper losses for one completed Stage-S epoch."""

    epoch: int
    group_mean_proper_loss: tuple[float, ...]


@dataclass(frozen=True)
class StageSSelection:
    """Frozen proper-loss checkpoint key for Stage S."""

    epoch: int
    macro_group_mean_proper_loss: float


def select_stage_s_checkpoint(
    candidates: Sequence[StageSCheckpointCandidate],
    *,
    expected_group_count: int = 8,
) -> StageSSelection:
    """Select minimum macro group-mean proper loss, then earliest epoch."""

    if not candidates:
        raise ValueError("at least one Stage-S checkpoint candidate is required")
    if expected_group_count <= 0:
        raise ValueError("expected_group_count must be positive")
    seen_epochs: set[int] = set()
    scored: list[StageSSelection] = []
    for candidate in candidates:
        if candidate.epoch <= 0 or candidate.epoch in seen_epochs:
            raise ValueError("checkpoint epochs must be unique positive integers")
        seen_epochs.add(candidate.epoch)
        values = candidate.group_mean_proper_loss
        if len(values) != expected_group_count:
            raise ValueError("every candidate must contain every selection group")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("proper losses must be finite")
        scored.append(
            StageSSelection(
                epoch=candidate.epoch,
                macro_group_mean_proper_loss=sum(values) / expected_group_count,
            )
        )
    return min(scored, key=lambda item: (item.macro_group_mean_proper_loss, item.epoch))


def scale_cap_fractions(
    scales_m: Tensor,
    *,
    scale_min_m: float = 2.0**-10,
    scale_max_m: float = 2.0**4,
    cap_position_tolerance: float = 1.0e-4,
) -> tuple[float, float]:
    """Return fractions lying in frozen sigmoid-position neighborhoods of either cap."""

    if scales_m.numel() == 0 or not torch.isfinite(scales_m).all():
        raise ValueError("scales must be non-empty and finite")
    if not 0.0 < scale_min_m < scale_max_m:
        raise ValueError("scale bounds must satisfy 0 < min < max")
    if not 0.0 < cap_position_tolerance < 0.5:
        raise ValueError("cap_position_tolerance must lie in (0, 0.5)")
    if torch.any(scales_m < scale_min_m) or torch.any(scales_m > scale_max_m):
        raise ValueError("scales lie outside the registered bounds")
    position = (scales_m - scale_min_m) / (scale_max_m - scale_min_m)
    lower = float((position <= cap_position_tolerance).to(torch.float64).mean().item())
    upper = float(
        (position >= 1.0 - cap_position_tolerance).to(torch.float64).mean().item()
    )
    return lower, upper


def qualifies_scale_caps(
    lower_fraction: float,
    upper_fraction: float,
    *,
    collapse_fraction: float = 0.95,
) -> bool:
    """Reject a selected head if at least 95% of outputs collapse near either cap."""

    values = (lower_fraction, upper_fraction, collapse_fraction)
    if any(not math.isfinite(value) for value in values):
        return False
    if not 0.0 <= lower_fraction <= 1.0 or not 0.0 <= upper_fraction <= 1.0:
        return False
    if not 0.0 < collapse_fraction <= 1.0:
        return False
    return lower_fraction < collapse_fraction and upper_fraction < collapse_fraction


def pooled_scale_control(edge_scales_m: Tensor) -> Tensor:
    """Broadcast each observation's geometric-mean learned scale to every edge."""

    if edge_scales_m.ndim != 2 or edge_scales_m.shape[1] == 0:
        raise ValueError("edge scales must have shape [batch, edges]")
    if not torch.isfinite(edge_scales_m).all() or torch.any(edge_scales_m <= 0.0):
        raise ValueError("edge scales must be finite and positive")
    geometric_mean = torch.exp(torch.log(edge_scales_m).mean(dim=-1, keepdim=True))
    return geometric_mean.expand_as(edge_scales_m)


def uniform_scale_control(edge_scales_m: Tensor) -> Tensor:
    """Return the same-shape registered one-metre scale control."""

    if edge_scales_m.ndim != 2 or edge_scales_m.shape[1] == 0:
        raise ValueError("edge scales must have shape [batch, edges]")
    return torch.ones_like(edge_scales_m)


def evidence_only_scale_control(
    node_slot_mask: Tensor,
    *,
    grid_size: int = 64,
    scale_min_m: float = 2.0**-10,
    scale_max_m: float = 2.0**4,
) -> Tensor:
    """Map projective support alone to a unit-geometric-mean edge scale shape.

    Edge reliability is the mean valid-slot fraction of its two endpoints.  The inverse
    square-root shape uses a one-slot floor, is gauge-normalized within each observation,
    and is then bounded by the same registered scale interval as the learned head.
    """

    if node_slot_mask.ndim != 3 or node_slot_mask.dtype is not torch.bool:
        raise ValueError("node_slot_mask must be boolean [batch, nodes, slots]")
    if grid_size <= 1 or node_slot_mask.shape[1] != grid_size * grid_size:
        raise ValueError("node count must equal grid_size squared")
    slot_count = node_slot_mask.shape[2]
    if slot_count <= 0:
        raise ValueError("slot count must be positive")
    if not 0.0 < scale_min_m < 1.0 < scale_max_m:
        raise ValueError("registered scale bounds must contain one metre")
    reliability = node_slot_mask.to(torch.float64).mean(dim=-1).reshape(
        node_slot_mask.shape[0], grid_size, grid_size
    )
    right = 0.5 * (reliability[:, :, :-1] + reliability[:, :, 1:])
    down = 0.5 * (reliability[:, :-1, :] + reliability[:, 1:, :])
    edge_reliability = torch.cat(
        (right.reshape(node_slot_mask.shape[0], -1), down.reshape(node_slot_mask.shape[0], -1)),
        dim=-1,
    )
    raw = torch.rsqrt(edge_reliability.clamp_min(1.0 / slot_count))
    gauge = torch.exp(torch.log(raw).mean(dim=-1, keepdim=True))
    normalized = raw / gauge
    bounded = normalized.clamp(min=scale_min_m, max=scale_max_m)
    return bounded.to(dtype=torch.float32)
