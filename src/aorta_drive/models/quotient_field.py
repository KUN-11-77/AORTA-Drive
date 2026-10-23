"""Candidate-free masked visual predictor for graph-potential differences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = math.gcd(channels, 16)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, tensor: Tensor) -> Tensor:
        residual = functional.gelu(self.norm1(tensor))
        residual = self.conv1(residual)
        residual = functional.gelu(self.norm2(residual))
        residual = self.conv2(residual)
        return tensor + residual


class QuotientFieldPredictor(nn.Module):
    """Fuse masked camera/history evidence and predict a zero-gauge node field.

    Inputs follow the frozen R004B contract. ``node_slot_features`` has shape
    ``[batch, grid_size**2, 12, input_dimension]`` and ``node_slot_mask`` has
    shape ``[batch, grid_size**2, 12]``. Slots are history-major, with camera
    order ``(left, front, right)`` inside each history offset.

    Invalid feature storage must be finite; its numeric value cannot affect the
    output because invalid slots receive exactly zero fusion weight. Nodes with
    no valid slots use one learned missing-evidence token. The final spatial
    mean is removed, representing the quotient by the common-offset nullspace.
    """

    camera_count = 3
    history_count = 4

    def __init__(
        self,
        *,
        input_dimension: int = 1280,
        token_dimension: int = 64,
        grid_size: int = 64,
        roi_half_extent_m: float = 32.0,
        spatial_widths: Sequence[int] = (96, 128, 192),
    ) -> None:
        super().__init__()
        if input_dimension <= 0 or token_dimension <= 0 or grid_size <= 0:
            raise ValueError("feature dimensions and grid_size must be positive")
        if not math.isfinite(roi_half_extent_m) or roi_half_extent_m <= 0.0:
            raise ValueError("roi_half_extent_m must be finite and positive")
        if len(spatial_widths) != 3 or any(width <= 0 for width in spatial_widths):
            raise ValueError("spatial_widths must contain three positive values")
        if grid_size % 4:
            raise ValueError("grid_size must be divisible by four")

        self.input_dimension = input_dimension
        self.token_dimension = token_dimension
        self.grid_size = grid_size
        self.roi_half_extent_m = roi_half_extent_m
        self.slot_count = self.camera_count * self.history_count

        self.feature_projection = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, token_dimension),
            nn.GELU(),
            nn.Linear(token_dimension, token_dimension),
        )
        self.camera_embedding = nn.Embedding(self.camera_count, token_dimension)
        self.history_embedding = nn.Embedding(self.history_count, token_dimension)
        self.coordinate_projection = nn.Sequential(
            nn.Linear(6, token_dimension),
            nn.GELU(),
            nn.Linear(token_dimension, token_dimension),
        )
        self.evidence_count_projection = nn.Sequential(
            nn.Linear(1, token_dimension),
            nn.GELU(),
            nn.Linear(token_dimension, token_dimension),
        )
        self.attention_score = nn.Sequential(
            nn.Linear(token_dimension, token_dimension),
            nn.Tanh(),
            nn.Linear(token_dimension, 1, bias=False),
        )
        self.missing_evidence_token = nn.Parameter(torch.zeros(token_dimension))
        self.fused_norm = nn.LayerNorm(token_dimension)

        width0, width1, width2 = (int(width) for width in spatial_widths)
        self.spatial_output_channels = width0
        self.spatial_stem = nn.Conv2d(token_dimension, width0, kernel_size=3, padding=1)
        self.encoder0 = nn.Sequential(_ResidualBlock(width0), _ResidualBlock(width0))
        self.down1 = nn.Conv2d(width0, width1, kernel_size=3, stride=2, padding=1)
        self.encoder1 = nn.Sequential(_ResidualBlock(width1), _ResidualBlock(width1))
        self.down2 = nn.Conv2d(width1, width2, kernel_size=3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(_ResidualBlock(width2), _ResidualBlock(width2))
        self.up1 = nn.Conv2d(width2, width1, kernel_size=3, padding=1)
        self.decoder1 = nn.Sequential(_ResidualBlock(width1), _ResidualBlock(width1))
        self.up0 = nn.Conv2d(width1, width0, kernel_size=3, padding=1)
        self.decoder0 = nn.Sequential(_ResidualBlock(width0), _ResidualBlock(width0))
        self.output_head = nn.Sequential(
            nn.GroupNorm(math.gcd(width0, 16), width0),
            nn.GELU(),
            nn.Conv2d(width0, 1, kernel_size=1),
        )

        camera_indices = torch.arange(self.camera_count).repeat(self.history_count)
        history_indices = torch.arange(self.history_count).repeat_interleave(self.camera_count)
        self.register_buffer("slot_camera_indices", camera_indices, persistent=True)
        self.register_buffer("slot_history_indices", history_indices, persistent=True)
        self.register_buffer(
            "normalized_coordinates",
            self._normalized_coordinate_features(grid_size, roi_half_extent_m),
            persistent=True,
        )

    @staticmethod
    def _normalized_coordinate_features(grid_size: int, half_extent: float) -> Tensor:
        resolution = 2.0 * half_extent / grid_size
        centers = torch.linspace(
            -half_extent + 0.5 * resolution,
            half_extent - 0.5 * resolution,
            grid_size,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        x = xx.reshape(-1) / half_extent
        y = yy.reshape(-1) / half_extent
        return torch.stack(
            (x, y, torch.sin(math.pi * x), torch.cos(math.pi * x),
             torch.sin(math.pi * y), torch.cos(math.pi * y)),
            dim=-1,
        )

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
        has_evidence = mask.any(dim=-1, keepdim=True)
        minimum = torch.finfo(logits.dtype).min
        masked_logits = torch.where(mask, logits, torch.full_like(logits, minimum))
        maximum = masked_logits.max(dim=-1, keepdim=True).values
        maximum = torch.where(has_evidence, maximum, torch.zeros_like(maximum))
        unnormalized = torch.exp(logits - maximum) * mask.to(logits.dtype)
        denominator = unnormalized.sum(dim=-1, keepdim=True)
        return unnormalized / denominator.clamp_min(torch.finfo(logits.dtype).tiny)

    def spatial_features(self, node_slot_features: Tensor, node_slot_mask: Tensor) -> Tensor:
        """Return the final spatial decoder map before the potential head."""

        expected_nodes = self.grid_size * self.grid_size
        expected_features = (
            node_slot_features.ndim == 4
            and node_slot_features.shape[1:] == (
                expected_nodes,
                self.slot_count,
                self.input_dimension,
            )
        )
        if not expected_features:
            raise ValueError(
                "node_slot_features must have shape "
                f"[batch, {expected_nodes}, {self.slot_count}, {self.input_dimension}]"
            )
        if node_slot_mask.shape != node_slot_features.shape[:3]:
            raise ValueError("node_slot_mask must match the first three feature dimensions")
        if node_slot_mask.dtype is not torch.bool:
            raise TypeError("node_slot_mask must be boolean")
        if not torch.isfinite(node_slot_features).all():
            raise ValueError("node_slot_features must be finite, including invalid storage")

        compute_dtype = self.missing_evidence_token.dtype
        features = node_slot_features.to(dtype=compute_dtype)
        projected = self.feature_projection(features)
        coordinates = self.normalized_coordinates.to(device=features.device, dtype=compute_dtype)
        coordinate_context = self.coordinate_projection(coordinates)
        slot_context = self.camera_embedding(self.slot_camera_indices)
        slot_context = slot_context + self.history_embedding(self.slot_history_indices)
        logits = self.attention_score(
            projected
            + slot_context[None, None, :, :]
            + coordinate_context[None, :, None, :]
        ).squeeze(-1)
        weights = self._masked_softmax(logits, node_slot_mask)
        fused = torch.sum(weights.unsqueeze(-1) * projected, dim=2)
        has_evidence = node_slot_mask.any(dim=-1)
        missing = self.missing_evidence_token[None, None, :].expand_as(fused)
        fused = torch.where(has_evidence.unsqueeze(-1), fused, missing)
        evidence_fraction = node_slot_mask.to(compute_dtype).mean(dim=-1, keepdim=True)
        fused = self.fused_norm(
            fused
            + coordinate_context[None, :, :]
            + self.evidence_count_projection(evidence_fraction)
        )

        batch = fused.shape[0]
        spatial = fused.transpose(1, 2).reshape(
            batch, self.token_dimension, self.grid_size, self.grid_size
        )
        level0 = self.encoder0(self.spatial_stem(spatial))
        level1 = self.encoder1(self.down1(level0))
        level2 = self.bottleneck(self.down2(level1))
        decoded1 = functional.interpolate(
            level2, size=level1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded1 = self.decoder1(self.up1(decoded1) + level1)
        decoded0 = functional.interpolate(
            decoded1, size=level0.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded0 = self.decoder0(self.up0(decoded0) + level0)
        return cast(Tensor, decoded0)

    def potential_from_spatial_features(self, spatial_features: Tensor) -> Tensor:
        """Map a validated final decoder map to the zero-gauge node potential."""

        expected = (self.spatial_output_channels, self.grid_size, self.grid_size)
        if spatial_features.ndim != 4 or spatial_features.shape[1:] != expected:
            raise ValueError(
                "spatial_features must have shape "
                f"[batch, {self.spatial_output_channels}, {self.grid_size}, {self.grid_size}]"
            )
        if not torch.isfinite(spatial_features).all():
            raise ValueError("spatial_features must be finite")
        potential = self.output_head(spatial_features).squeeze(1)
        return cast(Tensor, potential - potential.mean(dim=(-2, -1), keepdim=True))

    def forward(self, node_slot_features: Tensor, node_slot_mask: Tensor) -> Tensor:
        spatial = self.spatial_features(node_slot_features, node_slot_mask)
        return self.potential_from_spatial_features(spatial)


def oriented_grid_gradients(potential: Tensor) -> Tensor:
    """Return rightward then downward oriented gradients of a node potential."""

    if potential.ndim != 3 or potential.shape[-1] != potential.shape[-2]:
        raise ValueError("potential must have shape [batch, height, width] on a square grid")
    horizontal = potential[:, :, 1:] - potential[:, :, :-1]
    vertical = potential[:, 1:, :] - potential[:, :-1, :]
    return torch.cat((horizontal.reshape(potential.shape[0], -1), vertical.reshape(
        potential.shape[0], -1
    )), dim=-1)


@dataclass(frozen=True)
class QuotientLossOutput:
    total: Tensor
    mean_huber: Tensor
    centered_smooth_max: Tensor
    maximum_residual: Tensor


def quotient_gradient_loss(
    predicted_potential: Tensor,
    label_potential: Tensor,
    *,
    edge_scale: float = 1.0,
    huber_delta_m: float = 0.25,
    tail_temperature_m: float = 0.10,
    tail_weight: float = 0.25,
) -> QuotientLossOutput:
    """Train edge differences while emphasizing the within-anchor residual tail.

    The centered log-sum-exp has the same gradient as smooth maximum but removes
    the parameter-independent ``temperature * log(edge_count)`` offset. It is a
    training surrogate, not a statistical bound. Checkpoint selection still uses
    the exact per-recording-group maximum residual.
    """

    if predicted_potential.shape != label_potential.shape:
        raise ValueError("predicted and label potentials must have the same shape")
    for name, value in (
        ("edge_scale", edge_scale),
        ("huber_delta_m", huber_delta_m),
        ("tail_temperature_m", tail_temperature_m),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(tail_weight) or tail_weight < 0.0:
        raise ValueError("tail_weight must be finite and nonnegative")
    predicted_edges = oriented_grid_gradients(predicted_potential)
    label_edges = oriented_grid_gradients(label_potential)
    residual = torch.abs(predicted_edges - label_edges) / edge_scale
    mean_huber = functional.huber_loss(
        predicted_edges / edge_scale,
        label_edges / edge_scale,
        delta=huber_delta_m,
        reduction="mean",
    )
    edge_count = residual.shape[-1]
    smooth_max = tail_temperature_m * (
        torch.logsumexp(residual / tail_temperature_m, dim=-1)
        - math.log(edge_count)
    )
    centered_smooth_max = smooth_max.mean()
    total = mean_huber + tail_weight * centered_smooth_max
    return QuotientLossOutput(
        total=total,
        mean_huber=mean_huber,
        centered_smooth_max=centered_smooth_max,
        maximum_residual=residual.max(),
    )


@dataclass(frozen=True)
class DevelopmentCheckpoint:
    epoch: int
    session_max_residuals_m: tuple[float, ...]


@dataclass(frozen=True)
class DevelopmentSelection:
    epoch: int
    tail_mean_m: float
    mean_m: float


def select_development_checkpoint(
    candidates: Sequence[DevelopmentCheckpoint],
    *,
    expected_session_count: int = 24,
    tail_fraction: float = 0.20,
) -> DevelopmentSelection:
    """Select by development recording-group tails, then mean, then earliest epoch."""

    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    if expected_session_count <= 0:
        raise ValueError("expected_session_count must be positive")
    if not math.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    seen_epochs: set[int] = set()
    scored: list[DevelopmentSelection] = []
    tail_count = max(1, math.ceil(expected_session_count * tail_fraction))
    for candidate in candidates:
        if candidate.epoch <= 0 or candidate.epoch in seen_epochs:
            raise ValueError("checkpoint epochs must be unique positive integers")
        seen_epochs.add(candidate.epoch)
        values = candidate.session_max_residuals_m
        if len(values) != expected_session_count:
            raise ValueError("every checkpoint must contain every development session")
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("development residuals must be finite and nonnegative")
        ordered = sorted(values, reverse=True)
        scored.append(
            DevelopmentSelection(
                epoch=candidate.epoch,
                tail_mean_m=sum(ordered[:tail_count]) / tail_count,
                mean_m=sum(ordered) / expected_session_count,
            )
        )
    return min(scored, key=lambda item: (item.tail_mean_m, item.mean_m, item.epoch))
