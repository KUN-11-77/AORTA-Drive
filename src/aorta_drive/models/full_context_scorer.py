"""Matched full-context set scorer for the R005E2 recoverability gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


def _trajectory_features(poses: Tensor) -> Tensor:
    """Map ego-frame SE(2) samples to continuous, scale-controlled features."""

    if poses.ndim != 3 or poses.shape[-1] != 3 or poses.shape[1] < 2:
        raise ValueError("poses must have shape [batch, time>=2, 3]")
    if not torch.isfinite(poses).all():
        raise ValueError("trajectory poses must be finite")
    delta_xy = torch.diff(poses[..., :2], dim=1, prepend=poses[:, :1, :2])
    yaw_delta_raw = torch.diff(poses[..., 2], dim=1, prepend=poses[:, :1, 2])
    yaw_delta = torch.atan2(torch.sin(yaw_delta_raw), torch.cos(yaw_delta_raw))
    time = torch.linspace(
        0.0,
        1.0,
        poses.shape[1],
        device=poses.device,
        dtype=poses.dtype,
    ).expand(poses.shape[0], -1)
    return torch.stack(
        (
            poses[..., 0] / 50.0,
            poses[..., 1] / 50.0,
            torch.sin(poses[..., 2]),
            torch.cos(poses[..., 2]),
            delta_xy[..., 0] / 5.0,
            delta_xy[..., 1] / 5.0,
            torch.sin(yaw_delta),
            torch.cos(yaw_delta),
            time,
        ),
        dim=-1,
    )


class _TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dimension: int,
        attention_heads: int,
        layer_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(9, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=attention_heads,
            dim_feedforward=4 * hidden_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layer_count,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dimension)

    def forward(self, poses: Tensor) -> Tensor:
        tokens = self.input_projection(_trajectory_features(poses))
        encoded = self.encoder(tokens)
        return cast(Tensor, self.output_norm(encoded.mean(dim=1)))


class _VisualCrossAttention(nn.Module):
    def __init__(self, hidden_dimension: int, attention_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dimension)
        self.visual_norm = nn.LayerNorm(hidden_dimension)
        self.attention = nn.MultiheadAttention(
            hidden_dimension,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.post_attention_norm = nn.LayerNorm(hidden_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dimension, 4 * hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dimension, hidden_dimension),
            nn.Dropout(dropout),
        )

    def forward(self, candidate_tokens: Tensor, visual_tokens: Tensor, visual_mask: Tensor) -> Tensor:
        attended, _ = self.attention(
            self.query_norm(candidate_tokens),
            self.visual_norm(visual_tokens),
            self.visual_norm(visual_tokens),
            key_padding_mask=~visual_mask,
            need_weights=False,
        )
        residual = candidate_tokens + self.attention_dropout(attended)
        return cast(Tensor, residual + self.feed_forward(self.post_attention_norm(residual)))


class FullContextSetScorer(nn.Module):
    """Predict native-relative contextual margins from a ragged candidate set.

    The candidate axis has no positional embedding.  Consequently, shared
    temporal encoding, set self-attention, visual cross-attention, and the
    shared score head are equivariant to challenger permutations.  Candidate 0
    is the registered native trajectory; subtracting its raw score before the
    bounded output map fixes the native margin to exact zero.
    """

    camera_count = 3
    history_count = 4

    def __init__(
        self,
        *,
        visual_input_dimension: int = 1280,
        hidden_dimension: int = 256,
        attention_heads: int = 8,
        trajectory_layers: int = 2,
        set_layers: int = 2,
        visual_cross_attention_layers: int = 1,
        patch_height: int = 18,
        patch_width: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("visual_input_dimension", visual_input_dimension),
            ("hidden_dimension", hidden_dimension),
            ("attention_heads", attention_heads),
            ("trajectory_layers", trajectory_layers),
            ("set_layers", set_layers),
            ("patch_height", patch_height),
            ("patch_width", patch_width),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if hidden_dimension % attention_heads:
            raise ValueError("hidden_dimension must be divisible by attention_heads")
        if visual_cross_attention_layers != 1:
            raise ValueError("R005E2 freezes exactly one visual cross-attention layer")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.visual_input_dimension = visual_input_dimension
        self.hidden_dimension = hidden_dimension
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.slot_count = self.camera_count * self.history_count

        self.trajectory_encoder = _TrajectoryEncoder(
            hidden_dimension=hidden_dimension,
            attention_heads=attention_heads,
            layer_count=trajectory_layers,
            dropout=dropout,
        )
        self.context_fusion = nn.Sequential(
            nn.LayerNorm(3 * hidden_dimension),
            nn.Linear(3 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.candidate_role_embedding = nn.Embedding(2, hidden_dimension)
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=attention_heads,
            dim_feedforward=4 * hidden_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=set_layers,
            enable_nested_tensor=False,
        )
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_input_dimension),
            nn.Linear(visual_input_dimension, hidden_dimension),
        )
        self.camera_embedding = nn.Embedding(self.camera_count, hidden_dimension)
        self.history_embedding = nn.Embedding(self.history_count, hidden_dimension)
        self.visual_coordinate_projection = nn.Sequential(
            nn.Linear(6, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.visual_cross_attention = _VisualCrossAttention(
            hidden_dimension, attention_heads, dropout
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )

        camera_indices = torch.arange(self.camera_count).repeat(self.history_count)
        history_indices = torch.arange(self.history_count).repeat_interleave(
            self.camera_count
        )
        self.register_buffer("slot_camera_indices", camera_indices, persistent=True)
        self.register_buffer("slot_history_indices", history_indices, persistent=True)
        self.register_buffer(
            "visual_coordinate_features",
            self._coordinate_features(patch_height, patch_width),
            persistent=True,
        )

    @staticmethod
    def _coordinate_features(height: int, width: int) -> Tensor:
        y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack(
            (
                xx,
                yy,
                torch.sin(math.pi * xx),
                torch.cos(math.pi * xx),
                torch.sin(math.pi * yy),
                torch.cos(math.pi * yy),
            ),
            dim=-1,
        ).reshape(height * width, 6)

    def _validate_inputs(
        self,
        candidate_poses: Tensor,
        candidate_mask: Tensor,
        previous_poses: Tensor,
        visual_features: Tensor,
        visual_slot_mask: Tensor,
    ) -> tuple[int, int, int]:
        if candidate_poses.ndim != 4 or candidate_poses.shape[-1] != 3:
            raise ValueError("candidate_poses must have shape [batch, candidate, time, 3]")
        batch, candidates, time, _ = candidate_poses.shape
        if candidates < 1 or time < 2:
            raise ValueError("candidate set and trajectory time must be nonempty")
        if candidate_mask.shape != (batch, candidates) or candidate_mask.dtype is not torch.bool:
            raise ValueError("candidate_mask must be boolean [batch, candidate]")
        if not bool(candidate_mask[:, 0].all()):
            raise ValueError("candidate zero must be the valid native trajectory")
        if previous_poses.ndim != 3 or previous_poses.shape[0] != batch or previous_poses.shape[-1] != 3:
            raise ValueError("previous_poses must have shape [batch, previous_time, 3]")
        expected_visual = (
            batch,
            self.slot_count,
            self.patch_height,
            self.patch_width,
            self.visual_input_dimension,
        )
        if visual_features.shape != expected_visual:
            raise ValueError(f"visual_features must have shape {expected_visual}")
        if visual_slot_mask.shape != (batch, self.slot_count):
            raise ValueError("visual_slot_mask must have shape [batch, 12]")
        if visual_slot_mask.dtype is not torch.bool:
            raise TypeError("visual_slot_mask must be boolean")
        if not bool(visual_slot_mask.any(dim=1).all()):
            raise ValueError("each anchor must contain at least one valid visual slot")
        for name, tensor in (
            ("candidate_poses", candidate_poses),
            ("previous_poses", previous_poses),
            ("visual_features", visual_features),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite, including masked storage")
        return batch, candidates, time

    def _encode_visual(self, features: Tensor, slot_mask: Tensor) -> tuple[Tensor, Tensor]:
        batch = features.shape[0]
        compute_dtype = next(self.parameters()).dtype
        projected = self.visual_projection(features.to(dtype=compute_dtype))
        slot_context = self.camera_embedding(self.slot_camera_indices)
        slot_context = slot_context + self.history_embedding(self.slot_history_indices)
        coordinate_context = self.visual_coordinate_projection(
            self.visual_coordinate_features.to(device=features.device, dtype=compute_dtype)
        ).reshape(self.patch_height, self.patch_width, self.hidden_dimension)
        projected = projected + slot_context[None, :, None, None, :]
        projected = projected + coordinate_context[None, None, :, :, :]
        tokens = projected.reshape(batch, -1, self.hidden_dimension)
        mask = slot_mask[:, :, None, None].expand(
            batch, self.slot_count, self.patch_height, self.patch_width
        )
        return tokens, mask.reshape(batch, -1)

    def forward(
        self,
        candidate_poses: Tensor,
        candidate_mask: Tensor,
        previous_poses: Tensor,
        visual_features: Tensor,
        visual_slot_mask: Tensor,
    ) -> Tensor:
        batch, candidates, time = self._validate_inputs(
            candidate_poses,
            candidate_mask,
            previous_poses,
            visual_features,
            visual_slot_mask,
        )
        compute_dtype = next(self.parameters()).dtype
        candidate_flat = candidate_poses.reshape(batch * candidates, time, 3).to(
            dtype=compute_dtype
        )
        candidate_encoding = self.trajectory_encoder(candidate_flat).reshape(
            batch, candidates, self.hidden_dimension
        )
        previous_encoding = self.trajectory_encoder(
            previous_poses.to(dtype=compute_dtype)
        )[:, None, :].expand(-1, candidates, -1)
        fused = self.context_fusion(
            torch.cat(
                (
                    candidate_encoding,
                    previous_encoding,
                    candidate_encoding - previous_encoding,
                ),
                dim=-1,
            )
        )
        roles = torch.zeros(candidates, dtype=torch.long, device=candidate_poses.device)
        roles[0] = 1
        fused = fused + self.candidate_role_embedding(roles)[None, :, :]
        set_encoded = self.set_encoder(fused, src_key_padding_mask=~candidate_mask)
        visual_tokens, visual_mask = self._encode_visual(visual_features, visual_slot_mask)
        contextual = self.visual_cross_attention(set_encoded, visual_tokens, visual_mask)
        raw_scores = self.score_head(contextual).squeeze(-1)
        relative = torch.tanh(raw_scores - raw_scores[:, :1])
        return torch.where(candidate_mask, relative, torch.zeros_like(relative))


@dataclass(frozen=True)
class ContextMarginLoss:
    total: Tensor
    per_anchor: Tensor
    valid_challenger_count: Tensor


def equal_anchor_challenger_margin_mse(
    predicted_margin: Tensor,
    target_margin: Tensor,
    candidate_mask: Tensor,
) -> ContextMarginLoss:
    """Compute challenger-equal loss inside anchor-equal stochastic batches."""

    if predicted_margin.shape != target_margin.shape or predicted_margin.shape != candidate_mask.shape:
        raise ValueError("prediction, target, and candidate mask shapes must match")
    if predicted_margin.ndim != 2 or candidate_mask.dtype is not torch.bool:
        raise ValueError("margins must be rank two and candidate_mask must be boolean")
    if not bool(candidate_mask[:, 0].all()):
        raise ValueError("candidate zero must be valid native")
    if not torch.isfinite(predicted_margin).all() or not torch.isfinite(target_margin).all():
        raise ValueError("margins must be finite")
    if not torch.equal(predicted_margin[:, 0], torch.zeros_like(predicted_margin[:, 0])):
        raise ValueError("predicted native margin must be exact zero")
    if not torch.equal(target_margin[:, 0], torch.zeros_like(target_margin[:, 0])):
        raise ValueError("target native margin must be exact zero")
    challenger_mask = candidate_mask.clone()
    challenger_mask[:, 0] = False
    count = challenger_mask.sum(dim=1)
    squared = functional.mse_loss(predicted_margin, target_margin, reduction="none")
    summed = (squared * challenger_mask.to(squared.dtype)).sum(dim=1)
    per_anchor = torch.where(
        count > 0,
        summed / count.clamp_min(1).to(summed.dtype),
        torch.zeros_like(summed),
    )
    return ContextMarginLoss(
        total=per_anchor.mean(),
        per_anchor=per_anchor,
        valid_challenger_count=count,
    )
