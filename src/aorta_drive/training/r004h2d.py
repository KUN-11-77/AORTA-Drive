"""Read-only conditioning diagnostics for the R004H2 scale interface."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as functional


@dataclass(frozen=True)
class InitialScaleGradient:
    """Analytic initial gradient of the bounded scale head."""

    weight: Tensor
    bias: Tensor
    logit: Tensor
    feature_rms: float
    bounded_to_log_scale_derivative_ratio: float

    @property
    def weight_norm(self) -> float:
        return float(torch.linalg.vector_norm(self.weight).item())

    @property
    def bias_norm(self) -> float:
        return float(torch.linalg.vector_norm(self.bias).item())

    @property
    def full_norm(self) -> float:
        joined = torch.cat((self.weight.reshape(-1), self.bias.reshape(-1)))
        return float(torch.linalg.vector_norm(joined).item())


@dataclass(frozen=True)
class AdamWFirstStep:
    """Observable consequences of one analytic AdamW step on a cloned head."""

    maximum_absolute_logit_change: float
    rms_logit_change: float
    lower_cap_fraction: float
    upper_cap_fraction: float
    safe_learning_rate: float


def rms_only_normalize(spatial_features: Tensor) -> tuple[Tensor, float]:
    """Remove only per-observation activation magnitude, without centering."""

    if spatial_features.ndim != 4 or spatial_features.shape[0] != 1:
        raise ValueError("diagnostic spatial features must have shape [1, C, H, W]")
    if not torch.isfinite(spatial_features).all():
        raise ValueError("diagnostic spatial features must be finite")
    rms = torch.sqrt(torch.mean(spatial_features.square()))
    if not bool((rms > torch.finfo(rms.dtype).tiny).item()):
        raise ValueError("diagnostic spatial features must have nonzero RMS")
    return spatial_features / rms, float(rms.item())


def initial_scale_gradient(
    spatial_features: Tensor,
    residual_edges_m: Tensor,
    *,
    scale_min_m: float = 2.0**-10,
    scale_max_m: float = 2.0**4,
    initial_scale_m: float = 1.0,
) -> InitialScaleGradient:
    """Compute the exact initial proper-loss gradient of the 1x1 scale head.

    The returned logit tensor includes unused boundary pixels as exact zeros and
    follows the head's rightward-then-downward canonical edge ordering.
    """

    if spatial_features.ndim != 4 or spatial_features.shape[0] != 1:
        raise ValueError("spatial features must have shape [1, C, H, W]")
    height, width = spatial_features.shape[-2:]
    if height != width or height <= 1:
        raise ValueError("spatial features must use a square grid larger than one")
    edge_count = 2 * height * (height - 1)
    if residual_edges_m.shape != (1, edge_count):
        raise ValueError("residual edges disagree with the spatial grid")
    if not torch.isfinite(spatial_features).all() or not torch.isfinite(
        residual_edges_m
    ).all():
        raise ValueError("diagnostic tensors must be finite")
    if torch.any(residual_edges_m < 0.0):
        raise ValueError("edge residuals must be nonnegative")
    if not 0.0 < scale_min_m < initial_scale_m < scale_max_m:
        raise ValueError("scale bounds must strictly contain the initial scale")

    probability = (initial_scale_m - scale_min_m) / (scale_max_m - scale_min_m)
    bounded_derivative = (scale_max_m - scale_min_m) * probability * (
        1.0 - probability
    )
    log_scale_derivative = initial_scale_m
    dloss_dscale = (initial_scale_m - residual_edges_m) / initial_scale_m**2
    edge_logit_gradient = dloss_dscale * bounded_derivative / edge_count
    horizontal_count = height * (width - 1)
    logit_gradient = torch.zeros(
        (1, 2, height, width),
        dtype=spatial_features.dtype,
        device=spatial_features.device,
    )
    logit_gradient[:, 0, :, :-1] = edge_logit_gradient[:, :horizontal_count].reshape(
        1, height, width - 1
    )
    logit_gradient[:, 1, :-1, :] = edge_logit_gradient[:, horizontal_count:].reshape(
        1, height - 1, width
    )
    weight_gradient = torch.einsum(
        "bohw,bchw->oc", logit_gradient, spatial_features
    )
    bias_gradient = logit_gradient.sum(dim=(0, 2, 3))
    feature_rms = float(torch.sqrt(torch.mean(spatial_features.square())).item())
    return InitialScaleGradient(
        weight=weight_gradient,
        bias=bias_gradient,
        logit=logit_gradient,
        feature_rms=feature_rms,
        bounded_to_log_scale_derivative_ratio=(
            bounded_derivative / log_scale_derivative
        ),
    )


def analytic_adamw_first_step(
    spatial_features: Tensor,
    gradient: InitialScaleGradient,
    *,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-4,
    epsilon: float = 1.0e-8,
    gradient_clip_norm: float = 1.0,
    scale_min_m: float = 2.0**-10,
    scale_max_m: float = 2.0**4,
    initial_scale_m: float = 1.0,
    cap_position_tolerance: float = 1.0e-4,
    target_maximum_logit_change: float = 0.25,
) -> AdamWFirstStep:
    """Replay AdamW's bias-corrected first update without mutating parameters."""

    for name, value in (
        ("learning_rate", learning_rate),
        ("epsilon", epsilon),
        ("gradient_clip_norm", gradient_clip_norm),
        ("target_maximum_logit_change", target_maximum_logit_change),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not 0.0 < cap_position_tolerance < 0.5:
        raise ValueError("cap tolerance must lie in (0, 0.5)")

    full_norm = gradient.full_norm
    clip_factor = min(1.0, gradient_clip_norm / full_norm) if full_norm else 1.0
    clipped_weight = gradient.weight * clip_factor
    clipped_bias = gradient.bias * clip_factor
    weight_change = -learning_rate * clipped_weight / (
        torch.abs(clipped_weight) + epsilon
    )
    probability = (initial_scale_m - scale_min_m) / (scale_max_m - scale_min_m)
    initial_logit = math.log(probability / (1.0 - probability))
    bias_change = -learning_rate * clipped_bias / (
        torch.abs(clipped_bias) + epsilon
    ) - learning_rate * weight_decay * initial_logit
    logit_change = functional.conv2d(
        spatial_features,
        weight_change[:, :, None, None],
        bias_change,
    )
    post_logits = initial_logit + logit_change
    post_position = torch.sigmoid(post_logits)
    height, width = spatial_features.shape[-2:]
    edge_positions = torch.cat(
        (
            post_position[:, 0, :, :-1].reshape(1, -1),
            post_position[:, 1, :-1, :].reshape(1, -1),
        ),
        dim=-1,
    )
    maximum = float(torch.max(torch.abs(logit_change)).item())
    safe_learning_rate = learning_rate
    if maximum > target_maximum_logit_change:
        safe_learning_rate *= target_maximum_logit_change / maximum
    expected_edges = 2 * height * (width - 1)
    if edge_positions.shape != (1, expected_edges):
        raise AssertionError("diagnostic edge ordering is inconsistent")
    return AdamWFirstStep(
        maximum_absolute_logit_change=maximum,
        rms_logit_change=float(torch.sqrt(torch.mean(logit_change.square())).item()),
        lower_cap_fraction=float(
            torch.mean((edge_positions <= cap_position_tolerance).to(torch.float64)).item()
        ),
        upper_cap_fraction=float(
            torch.mean(
                (edge_positions >= 1.0 - cap_position_tolerance).to(torch.float64)
            ).item()
        ),
        safe_learning_rate=safe_learning_rate,
    )
