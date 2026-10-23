"""R004H1 checkpoint-selection and central-signal contract tests."""

from __future__ import annotations

import math
import csv
import hashlib
from pathlib import Path

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.heterogeneous import deterministic_group_roles
from aorta_drive.training.r004h import (
    StagePCheckpointCandidate,
    StageSCheckpointCandidate,
    evidence_only_scale_control,
    pooled_scale_control,
    qualifies_stage_p_signal,
    qualifies_scale_caps,
    scale_cap_fractions,
    select_stage_p_checkpoint,
    select_stage_s_checkpoint,
    uniform_scale_control,
)


ROOT = Path(__file__).parents[2]


def _candidate(epoch: int, mean: float, maximum: float) -> StagePCheckpointCandidate:
    return StagePCheckpointCandidate(
        epoch=epoch,
        group_mean_edge_mae_m=(mean,) * 8,
        group_max_residual_m=(maximum,) * 8,
    )


def test_stage_p_selector_prioritizes_macro_mean_then_tail_then_epoch() -> None:
    selected = select_stage_p_checkpoint(
        (
            _candidate(1, 0.40, 1.0),
            _candidate(2, 0.39, 2.0),
            _candidate(3, 0.39, 1.5),
            _candidate(4, 0.39, 1.5),
        )
    )
    assert selected.epoch == 3
    assert selected.macro_group_mean_edge_mae_m == pytest.approx(0.39)
    assert selected.worst_two_group_max_mean_m == pytest.approx(1.5)


def test_stage_p_selector_uses_exactly_two_worst_group_maxima() -> None:
    maxima = (1.0, 3.0, 2.0, 8.0, 4.0, 5.0, 6.0, 7.0)
    selected = select_stage_p_checkpoint(
        (
            StagePCheckpointCandidate(
                epoch=1,
                group_mean_edge_mae_m=(0.5,) * 8,
                group_max_residual_m=maxima,
            ),
        )
    )
    assert selected.worst_two_group_max_mean_m == pytest.approx(7.5)


@pytest.mark.parametrize(
    "candidate",
    [
        StagePCheckpointCandidate(0, (0.5,) * 8, (1.0,) * 8),
        StagePCheckpointCandidate(1, (0.5,) * 7, (1.0,) * 8),
        StagePCheckpointCandidate(1, (0.5,) * 8, (1.0,) * 7),
        StagePCheckpointCandidate(1, (math.nan,) * 8, (1.0,) * 8),
        StagePCheckpointCandidate(1, (0.5,) * 8, (-1.0,) * 8),
    ],
)
def test_stage_p_selector_rejects_invalid_candidates(
    candidate: StagePCheckpointCandidate,
) -> None:
    with pytest.raises(ValueError):
        select_stage_p_checkpoint((candidate,))


def test_stage_p_signal_requires_strict_finite_improvement_over_zero() -> None:
    assert qualifies_stage_p_signal(learned_mean_m=0.49, zero_mean_m=0.50)
    assert not qualifies_stage_p_signal(learned_mean_m=0.50, zero_mean_m=0.50)
    assert not qualifies_stage_p_signal(learned_mean_m=0.51, zero_mean_m=0.50)
    assert not qualifies_stage_p_signal(learned_mean_m=math.nan, zero_mean_m=0.50)


def test_h1_config_and_role_manifest_are_hash_pinned_and_replayable() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiment/r004h1_stage_p.yaml").read_text(encoding="utf-8")
    )
    role_path = ROOT / "configs/manifests/r004h_group_roles.csv"
    role_hash = hashlib.sha256(role_path.read_bytes()).hexdigest()
    assert role_hash == config["inputs"]["role_manifest_sha256"]
    with role_path.open(encoding="utf-8", newline="") as handle:
        role_rows = list(csv.DictReader(handle))
    with (ROOT / "configs/manifests/r004_recording_group_split.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    train_groups = [row["recording_group"] for row in source_rows if row["split"] == "train"]
    replay = deterministic_group_roles(
        train_groups,
        namespace="AORTA-R004EG-roles-v1",
        fit_count=42,
        selection_count=8,
        proxy_cal_count=16,
    )
    assert tuple(row["recording_group"] for row in role_rows if row["role"] == "fit") == replay.fit
    assert tuple(
        row["recording_group"] for row in role_rows if row["role"] == "selection"
    ) == replay.selection
    assert tuple(
        row["recording_group"] for row in role_rows if row["role"] == "proxy_cal"
    ) == replay.proxy_cal


def test_h1_training_entrypoint_cannot_accept_or_load_an_external_checkpoint() -> None:
    source = (ROOT / "scripts/train_r004h1_stage_p.py").read_text(encoding="utf-8")
    config = yaml.safe_load(
        (ROOT / "configs/experiment/r004h1_stage_p.yaml").read_text(encoding="utf-8")
    )
    assert "--checkpoint" not in source
    assert "torch.load(" not in source
    assert config["model"]["external_checkpoint_load"] == "prohibited"


def test_stage_s_selector_uses_macro_proper_loss_then_earliest_epoch() -> None:
    selected = select_stage_s_checkpoint(
        (
            StageSCheckpointCandidate(1, (0.4,) * 8),
            StageSCheckpointCandidate(2, (0.3,) * 8),
            StageSCheckpointCandidate(3, (0.3,) * 8),
        )
    )
    assert selected.epoch == 2
    assert selected.macro_group_mean_proper_loss == pytest.approx(0.3)


def test_stage_s_selector_allows_negative_proper_loss_but_rejects_nonfinite() -> None:
    selected = select_stage_s_checkpoint((StageSCheckpointCandidate(1, (-0.2,) * 8),))
    assert selected.macro_group_mean_proper_loss == pytest.approx(-0.2)
    with pytest.raises(ValueError, match="finite"):
        select_stage_s_checkpoint((StageSCheckpointCandidate(1, (math.inf,) * 8),))


def test_scale_cap_gate_uses_registered_sigmoid_position_and_strict_collapse_rule() -> None:
    scales = torch.tensor([[2.0**-10, 1.0, 2.0**4]])
    lower, upper = scale_cap_fractions(scales)
    assert lower == pytest.approx(1 / 3)
    assert upper == pytest.approx(1 / 3)
    assert qualifies_scale_caps(lower, upper)
    assert not qualifies_scale_caps(0.95, 0.0)
    assert not qualifies_scale_caps(0.0, 0.95)


def test_pooled_and_uniform_controls_preserve_shape_and_registered_gauges() -> None:
    learned = torch.tensor([[0.5, 1.0, 2.0], [2.0, 2.0, 2.0]])
    pooled = pooled_scale_control(learned)
    uniform = uniform_scale_control(learned)
    torch.testing.assert_close(pooled[0], torch.ones(3))
    torch.testing.assert_close(pooled[1], torch.full((3,), 2.0))
    torch.testing.assert_close(uniform, torch.ones_like(learned))


def test_evidence_only_control_uses_right_then_down_order_and_unit_geometric_gauge() -> None:
    mask = torch.zeros(1, 9, 4, dtype=torch.bool)
    mask[:, 0, :] = True
    mask[:, 1, :2] = True
    mask[:, 3, :1] = True
    scales = evidence_only_scale_control(mask, grid_size=3)
    assert scales.shape == (1, 12)
    node = mask.to(torch.float64).mean(dim=-1).reshape(1, 3, 3)
    right = 0.5 * (node[:, :, :-1] + node[:, :, 1:])
    down = 0.5 * (node[:, :-1, :] + node[:, 1:, :])
    raw = torch.rsqrt(torch.cat((right.reshape(1, -1), down.reshape(1, -1)), dim=-1).clamp_min(0.25))
    expected = raw / torch.exp(torch.log(raw).mean(dim=-1, keepdim=True))
    torch.testing.assert_close(scales, expected.to(torch.float32))
    assert torch.exp(torch.log(scales.to(torch.float64)).mean()).item() == pytest.approx(
        1.0, abs=1e-7
    )


def test_h2_config_freezes_proper_loss_caps_controls_and_h1_checkpoint() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiment/r004h2_scale_fit.yaml").read_text(encoding="utf-8")
    )
    assert config["predecessor"]["selected_checkpoint_sha256"] == (
        "8b5110b312341f1516067c55d6bb5a9cdec9b230de3837f451f993d5dbc0906b"
    )
    assert config["model"]["stage_p_trainable_parameters"] == 0
    assert config["model"]["scale_head_trainable_parameters"] == 194
    assert config["objective"]["group_tail_term"] == "prohibited"
    assert config["cap_collapse"]["cap_position_tolerance"] == 1.0e-4
    assert config["cap_collapse"]["collapse_fraction"] == 0.95
    assert set(config["controls"]) == {"learned", "pooled", "uniform", "evidence_only", "proxy_threshold_or_task_metric_used"}
