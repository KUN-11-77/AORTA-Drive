from __future__ import annotations

from pathlib import Path

import numpy as np

from aorta_drive.vision.r006_basis_audit import audit_fold_basis_result


def _groups(prefix: str, count: int) -> list[str]:
    return [f"{prefix}-{index}" for index in range(count)]


def test_audit_recomputes_stop_decision_and_rejects_tampering(tmp_path: Path) -> None:
    discovery = _groups("discovery", 28)
    selection = _groups("selection", 7)
    outer = _groups("outer", 7)
    candidates = {}
    for layer in ("planner_memory", "trajectory_query_secondary"):
        ranks = {}
        for rank in (4, 8, 16, 32):
            basis = np.eye(256, rank, dtype=np.float32)
            basis_path = tmp_path / "bases" / f"{layer}_{rank}.npy"
            basis_path.parent.mkdir(exist_ok=True)
            np.save(basis_path, basis, allow_pickle=False)
            from aorta_drive.vision.r006_cache import sha256_file

            samples = [
                {
                    "recording_group": group,
                    "anchor_index": anchor,
                    "scene_token": f"{group}-{anchor}",
                    "condition": condition,
                    "total_task_drift_sq": 1.0,
                    "uncovered_task_drift_sq": 0.8,
                }
                for group in selection
                for anchor in range(16)
                for condition in ("low_exposure", "low_contrast", "cool_cast")
            ]
            ranks[str(rank)] = {
                "basis_file": str(basis_path.relative_to(tmp_path)),
                "basis_sha256": sha256_file(basis_path),
                "basis_shape": [256, rank],
                "discovery_objective": 1.0,
                "solver_steps": 1,
                "solver_initialization": "test",
                "discovery_sample_count": 1344,
                "selection_sample_count": 336,
                "selection_summary": {
                    "rank": rank,
                    "group_count": 7,
                    "group_equal_mean": 0.2000000000008,
                    "worst_group_mean": 0.2000000000008,
                    "qualifies": False,
                    "group_means": {group: 0.2000000000008 for group in selection},
                },
                "selection_samples": samples,
            }
        candidates[layer] = ranks
    result = {
        "artifact": "R006V0_FOLD_BASIS_SELECTION_V1",
        "status": "STOP_NO_QUALIFIED_INTERFACE",
        "mechanical_status": "PASS",
        "scientific_metrics_emitted": True,
        "outer_confirmation_metrics_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
        "roles": {
            "basis_discovery_groups": discovery,
            "layer_rank_selection_groups": selection,
            "outer_confirmation_groups": outer,
        },
        "conditions": ["low_exposure", "low_contrast", "cool_cast"],
        "registered_layers": ["planner_memory", "trajectory_query_secondary"],
        "registered_ranks": [4, 8, 16, 32],
        "streaming_batch_size": 128,
        "output_metric": "W=I_(2T)/T",
        "candidates": candidates,
        "selected_interface": None,
    }
    fold = {
        "basis_discovery_groups": discovery,
        "layer_rank_selection_groups": selection,
        "outer_confirmation_groups": outer,
    }
    checks, _ = audit_fold_basis_result(result, result_dir=tmp_path, expected_fold=fold)
    assert all(bool(row["pass"]) for row in checks)

    result["candidates"]["planner_memory"]["4"]["selection_summary"][
        "group_equal_mean"
    ] = 0.99
    checks, _ = audit_fold_basis_result(result, result_dir=tmp_path, expected_fold=fold)
    assert not all(bool(row["pass"]) for row in checks)
