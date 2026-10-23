"""R004H1 checkpoint-selection and central-signal contract tests."""

from __future__ import annotations

import math
import csv
import hashlib
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.heterogeneous import deterministic_group_roles
from aorta_drive.training.r004h import (
    StagePCheckpointCandidate,
    qualifies_stage_p_signal,
    select_stage_p_checkpoint,
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
