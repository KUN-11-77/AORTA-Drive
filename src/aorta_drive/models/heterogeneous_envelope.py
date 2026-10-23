"""Minimal observation-and-edge scale head for the AORTA-v8.4 probe."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from aorta_drive.models.quotient_field import QuotientFieldPredictor, oriented_grid_gradients


@dataclass(frozen=True)
class HeterogeneousEnvelopeOutput:
    """Zero-gauge potential and canonical right/down edge scales."""

    potential: Tensor
    edge_scales_m: Tensor


@dataclass(frozen=True)
class ScaleLossOutput:
    """Components of the bounded Laplace proper scale loss."""

    total: Tensor
    mean_scaled_absolute_residual: Tensor
    mean_log_scale: Tensor


def bounded_laplace_scale_loss(
    predicted_potential: Tensor,
    label_potential: Tensor,
    edge_scales_m: Tensor,
) -> ScaleLossOutput:
    """Fit scale shape without a standardized-tail term."""

    if predicted_potential.shape != label_potential.shape:
        raise ValueError("predicted and label potentials must have the same shape")
    predicted_edges = oriented_grid_gradients(predicted_potential)
    label_edges = oriented_grid_gradients(label_potential)
    if edge_scales_m.shape != predicted_edges.shape:
        raise ValueError("edge_scales_m must match the canonical edge-gradient matrix")
    if not torch.isfinite(edge_scales_m).all() or torch.any(edge_scales_m <= 0.0):
        raise ValueError("edge scales must be finite and strictly positive")
    residual = torch.abs(predicted_edges - label_edges)
    scaled = (residual / edge_scales_m).mean()
    log_scale = torch.log(edge_scales_m / 1.0).mean()
    return ScaleLossOutput(
        total=scaled + log_scale,
        mean_scaled_absolute_residual=scaled,
        mean_log_scale=log_scale,
    )


class BoundedEdgeScaleHead(nn.Module):
    """Predict positive bounded right/down scales from a frozen spatial map."""

    def __init__(
        self,
        *,
        input_channels: int = 96,
        grid_size: int = 64,
        scale_min_m: float = 2.0**-10,
        scale_max_m: float = 2.0**4,
        initial_scale_m: float = 1.0,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or grid_size <= 1:
            raise ValueError("input_channels must be positive and grid_size must exceed one")
        for name, value in (
            ("scale_min_m", scale_min_m),
            ("scale_max_m", scale_max_m),
            ("initial_scale_m", initial_scale_m),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not scale_min_m < initial_scale_m < scale_max_m:
            raise ValueError("initial scale must lie strictly inside the registered bounds")

        self.input_channels = input_channels
        self.grid_size = grid_size
        self.scale_min_m = scale_min_m
        self.scale_max_m = scale_max_m
        self.initial_scale_m = initial_scale_m
        self.projection = nn.Conv2d(input_channels, 2, kernel_size=1)
        probability = (initial_scale_m - scale_min_m) / (scale_max_m - scale_min_m)
        initial_logit = math.log(probability / (1.0 - probability))
        nn.init.zeros_(self.projection.weight)
        nn.init.constant_(self.projection.bias, initial_logit)

    @property
    def edge_count(self) -> int:
        """Return the canonical number of right/down grid edges."""

        return 2 * self.grid_size * (self.grid_size - 1)

    def forward(self, spatial_features: Tensor) -> Tensor:
        expected = (self.input_channels, self.grid_size, self.grid_size)
        if spatial_features.ndim != 4 or spatial_features.shape[1:] != expected:
            raise ValueError(
                "spatial_features must have shape "
                f"[batch, {self.input_channels}, {self.grid_size}, {self.grid_size}]"
            )
        if not torch.isfinite(spatial_features).all():
            raise ValueError("spatial_features must be finite")
        logits = self.projection(spatial_features)
        scales = self.scale_min_m + (
            self.scale_max_m - self.scale_min_m
        ) * torch.sigmoid(logits)
        horizontal = scales[:, 0, :, :-1].reshape(spatial_features.shape[0], -1)
        vertical = scales[:, 1, :-1, :].reshape(spatial_features.shape[0], -1)
        edges = torch.cat((horizontal, vertical), dim=-1)
        if edges.shape[-1] != self.edge_count:
            raise RuntimeError("edge scale ordering produced an unexpected edge count")
        return edges


class HeterogeneousEnvelopePredictor(nn.Module):
    """Freeze a potential predictor and train only its minimal scale head."""

    def __init__(self, potential_predictor: QuotientFieldPredictor) -> None:
        super().__init__()
        self.potential_predictor = potential_predictor
        self.scale_head = BoundedEdgeScaleHead(
            input_channels=potential_predictor.spatial_output_channels,
            grid_size=potential_predictor.grid_size,
        )
        for parameter in self.potential_predictor.parameters():
            parameter.requires_grad_(False)
        self.potential_predictor.eval()

    def train(self, mode: bool = True) -> HeterogeneousEnvelopePredictor:
        """Set scale-head mode while keeping the potential predictor frozen in eval."""

        super().train(mode)
        self.potential_predictor.eval()
        return self

    def forward(
        self,
        node_slot_features: Tensor,
        node_slot_mask: Tensor,
    ) -> HeterogeneousEnvelopeOutput:
        spatial = self.potential_predictor.spatial_features(
            node_slot_features,
            node_slot_mask,
        )
        potential = self.potential_predictor.potential_from_spatial_features(spatial)
        edge_scales = self.scale_head(spatial.detach())
        return HeterogeneousEnvelopeOutput(
            potential=potential,
            edge_scales_m=edge_scales,
        )
