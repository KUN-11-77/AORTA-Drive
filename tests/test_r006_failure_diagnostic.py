from __future__ import annotations

import numpy as np
import pytest
import torch

from aorta_drive.vision.r006_failure_diagnostic import (
    R006V0F_RANKS,
    R006V0F_SEEDS,
    audit_r006v0f_fold_result,
    minimum_qualifying_rank,
    normalized_projector_overlap,
    route_failure_decomposition,
    summarize_drift_denominators,
)
from aorta_drive.vision.r006_cache import sha256_file
from aorta_drive.vision.r006_protocol import summarize_basis_coverage


def test_projector_overlap_is_rotation_invariant() -> None:
    rng = np.random.default_rng(17)
    basis, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    assert normalized_projector_overlap(basis, basis @ rotation) == pytest.approx(1.0)


def test_minimum_qualifying_rank_uses_registered_order() -> None:
    assert minimum_qualifying_rank({4: False, 8: True, 16: False, 32: True}) == 8
    assert minimum_qualifying_rank({4: False, 8: False, 16: False, 32: False}) is None


def _seed_map(rank: int | None) -> dict[int, int | None]:
    return {seed: rank for seed in R006V0F_SEEDS}


def test_route_prioritizes_qualification_instability() -> None:
    rows = {fold: _seed_map(8) for fold in range(6)}
    route = route_failure_decomposition(rows, qualification_flip_folds=[1, 4])
    assert route.status == "RANK_ESTIMATION_UNSTABLE"
    assert route.unstable_fold_count == 2


def test_route_detects_stable_fold_heterogeneous_dimension() -> None:
    rows = {fold: _seed_map(4 if fold < 3 else 32) for fold in range(6)}
    route = route_failure_decomposition(rows, qualification_flip_folds=[])
    assert route.status == "STABLE_FOLD_HETEROGENEOUS_EFFECTIVE_DIMENSION"
    assert route.stable_fold_count == 6


def test_route_is_inconclusive_when_stable_folds_include_no_interface() -> None:
    rows = {fold: _seed_map(None if fold == 5 else 8) for fold in range(6)}
    route = route_failure_decomposition(rows, qualification_flip_folds=[])
    assert route.status == "INCONCLUSIVE_DIRECTION_OR_THRESHOLD_STRUCTURE"


def test_drift_denominator_summary_exposes_near_zero_mass() -> None:
    summary = summarize_drift_denominators([0.0, 1e-13, 1e-6, 2.0], epsilon=1e-12)
    assert summary.sample_count == 4
    assert summary.near_zero_count == 2
    assert summary.near_zero_fraction == pytest.approx(0.5)
    assert summary.minimum_positive == pytest.approx(1e-6)
    assert summary.median == pytest.approx((1e-13 + 1e-6) / 2.0)


def test_drift_denominator_summary_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        summarize_drift_denominators([], epsilon=1e-12)
    with pytest.raises(ValueError):
        summarize_drift_denominators([1.0, -1.0], epsilon=1e-12)


def test_fold_audit_recomputes_every_rank_seed_and_detects_hash_tamper(
    tmp_path,
) -> None:
    discovery = [f"discovery-{index}" for index in range(28)]
    selection = [f"selection-{index}" for index in range(7)]
    outer = [f"outer-{index}" for index in range(7)]
    group_ids = [
        group
        for group in selection
        for _anchor in range(16)
        for _condition in ("low_exposure", "low_contrast", "cool_cast")
    ]
    totals = torch.ones(336, dtype=torch.float64)
    uncovered = torch.full((336,), 0.05, dtype=torch.float64)
    candidates = {}
    for rank in R006V0F_RANKS:
        summary = summarize_basis_coverage(
            totals,
            uncovered,
            group_ids,
            rank=rank,
            epsilon=1e-12,
            mean_threshold=0.90,
            worst_group_threshold=0.50,
        )
        rank_rows = {}
        for seed in R006V0F_SEEDS:
            basis_path = tmp_path / "bases" / f"rank_{rank}_seed_{seed}.npy"
            basis_path.parent.mkdir(exist_ok=True)
            np.save(basis_path, np.eye(256, rank, dtype=np.float32), allow_pickle=False)
            samples = [
                {
                    "recording_group": group,
                    "anchor_index": anchor,
                    "scene_token": f"{group}-{anchor}",
                    "condition": condition,
                    "total_task_drift_sq": 1.0,
                    "uncovered_task_drift_sq": 0.05,
                }
                for group in selection
                for anchor in range(16)
                for condition in ("low_exposure", "low_contrast", "cool_cast")
            ]
            rank_rows[str(seed)] = {
                "basis_file": str(basis_path.relative_to(tmp_path)),
                "basis_sha256": sha256_file(basis_path),
                "basis_shape": [256, rank],
                "discovery_objective": 0.05,
                "solver_steps": 3,
                "solver_initialization": f"random_{seed}",
                "discovery_sample_count": 1344,
                "selection_sample_count": 336,
                "selection_summary": {
                    "rank": rank,
                    "group_count": 7,
                    "group_equal_mean": summary.group_equal_mean,
                    "worst_group_mean": summary.worst_group_mean,
                    "qualifies": summary.qualifies,
                    "group_means": summary.group_means,
                },
                "selection_samples": samples,
            }
        candidates[str(rank)] = rank_rows
    result = {
        "artifact": "R006V0F_FOLD_FAILURE_DECOMPOSITION_V1",
        "status": "PASS_DIAGNOSTIC_MATERIALIZED",
        "mechanical_status": "PASS",
        "scientific_metrics_emitted": True,
        "outer_confirmation_metrics_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
        "registered_layer": "trajectory_query_secondary",
        "registered_ranks": list(R006V0F_RANKS),
        "registered_seeds": list(R006V0F_SEEDS),
        "streaming_batch_size": 128,
        "conditions": ["low_exposure", "low_contrast", "cool_cast"],
        "roles": {
            "basis_discovery_groups": discovery,
            "layer_rank_selection_groups": selection,
            "outer_confirmation_groups": outer,
        },
        "candidates": candidates,
    }
    expected_fold = {
        "basis_discovery_groups": discovery,
        "layer_rank_selection_groups": selection,
        "outer_confirmation_groups": outer,
    }
    checks = audit_r006v0f_fold_result(
        result, result_dir=tmp_path, expected_fold=expected_fold
    )
    assert all(row["pass"] for row in checks)

    result["candidates"]["8"][str(R006V0F_SEEDS[0])]["basis_sha256"] = "tampered"
    tampered = audit_r006v0f_fold_result(
        result, result_dir=tmp_path, expected_fold=expected_fold
    )
    assert any(
        row["check"] == f"rank8/seed{R006V0F_SEEDS[0]}: basis hash" and not row["pass"]
        for row in tampered
    )
