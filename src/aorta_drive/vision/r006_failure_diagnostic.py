"""Pure analysis and audit primitives for the R006V0F failure decomposition."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .r006_cache import sha256_file
from .r006_protocol import summarize_basis_coverage


R006V0F_RANKS = (4, 8, 16, 32)
R006V0F_SEEDS = tuple(range(2026102701, 2026102709))
R006V0F_CONDITIONS = ("low_exposure", "low_contrast", "cool_cast")


@dataclass(frozen=True)
class FailureDecompositionRoute:
    status: str
    unstable_fold_count: int
    stable_fold_count: int
    stable_fold_modes: dict[int, int | None]
    per_fold_mode_counts: dict[int, dict[str, int]]


def normalized_projector_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Return ``tr(P_left P_right) / rank`` without materializing projectors."""

    if left.ndim != 2 or right.shape != left.shape or left.shape[1] <= 0:
        raise ValueError("basis arrays must have the same nonempty [C,r] shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("basis arrays must be finite")
    rank = left.shape[1]
    return float(np.sum((left.T @ right) ** 2) / rank)


def minimum_qualifying_rank(qualifies: Mapping[int, bool]) -> int | None:
    """Apply the frozen smallest-qualifying-rank rule to one seed and fold."""

    if set(qualifies) != set(R006V0F_RANKS):
        raise ValueError("qualification mapping must contain the registered ranks")
    return next((rank for rank in R006V0F_RANKS if qualifies[rank]), None)


def route_failure_decomposition(
    seed_minimum_ranks: Mapping[int, Mapping[int, int | None]],
    qualification_flip_folds: Sequence[int],
) -> FailureDecompositionRoute:
    """Route the post-failure study using its preregistered repeated-seed rules."""

    if set(seed_minimum_ranks) != set(range(6)):
        raise ValueError("all six folds are required")
    expected_seeds = set(R006V0F_SEEDS)
    mode_counts: dict[int, dict[str, int]] = {}
    stable_modes: dict[int, int | None] = {}
    for fold in range(6):
        values = seed_minimum_ranks[fold]
        if set(values) != expected_seeds:
            raise ValueError("every fold must contain all eight registered seeds")
        counts = Counter(values.values())
        mode_counts[fold] = {
            "none" if rank is None else str(rank): count
            for rank, count in sorted(
                counts.items(), key=lambda item: (-1 if item[0] is None else item[0])
            )
        }
        highest = max(counts.values())
        modes = [rank for rank, count in counts.items() if count == highest]
        if highest >= 7 and len(modes) == 1:
            stable_modes[fold] = modes[0]

    unstable_folds = sorted(set(int(value) for value in qualification_flip_folds))
    if len(unstable_folds) >= 2:
        status = "RANK_ESTIMATION_UNSTABLE"
    else:
        nonnull_stable = [rank for rank in stable_modes.values() if rank is not None]
        if (
            len(stable_modes) >= 5
            and len(nonnull_stable) == len(stable_modes)
            and len(set(nonnull_stable)) > 1
        ):
            status = "STABLE_FOLD_HETEROGENEOUS_EFFECTIVE_DIMENSION"
        else:
            status = "INCONCLUSIVE_DIRECTION_OR_THRESHOLD_STRUCTURE"
    return FailureDecompositionRoute(
        status=status,
        unstable_fold_count=len(unstable_folds),
        stable_fold_count=len(stable_modes),
        stable_fold_modes=stable_modes,
        per_fold_mode_counts=mode_counts,
    )


def route_as_dict(route: FailureDecompositionRoute) -> dict[str, Any]:
    """Return a JSON-compatible representation of a diagnostic route."""

    return asdict(route)


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return (
        math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance
    )


def _expected_sample_keys(groups: Sequence[str]) -> set[tuple[str, int, str]]:
    return {
        (str(group), anchor, condition)
        for group in groups
        for anchor in range(16)
        for condition in R006V0F_CONDITIONS
    }


def audit_r006v0f_fold_result(
    result: Mapping[str, Any],
    *,
    result_dir: Path,
    expected_fold: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Recompute one diagnostic fold without trusting stored summaries."""

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool) -> None:
        checks.append({"check": name, "pass": bool(passed)})

    check("artifact", result.get("artifact") == "R006V0F_FOLD_FAILURE_DECOMPOSITION_V1")
    check("status", result.get("status") == "PASS_DIAGNOSTIC_MATERIALIZED")
    check("mechanical status", result.get("mechanical_status") == "PASS")
    check("science emitted", result.get("scientific_metrics_emitted") is True)
    check(
        "outer metrics absent",
        result.get("outer_confirmation_metrics_emitted") is False,
    )
    check(
        "sealed roles untouched",
        result.get("development_calibration_test_navdream_target_accessed") is False,
    )
    check(
        "registered layer",
        result.get("registered_layer") == "trajectory_query_secondary",
    )
    check(
        "registered ranks", tuple(result.get("registered_ranks", [])) == R006V0F_RANKS
    )
    check(
        "registered seeds", tuple(result.get("registered_seeds", [])) == R006V0F_SEEDS
    )
    check("fixed batch size", result.get("streaming_batch_size") == 128)

    roles = result.get("roles", {})
    discovery = list(roles.get("basis_discovery_groups", []))
    selection = list(roles.get("layer_rank_selection_groups", []))
    outer = list(roles.get("outer_confirmation_groups", []))
    for key in (
        "basis_discovery_groups",
        "layer_rank_selection_groups",
        "outer_confirmation_groups",
    ):
        check(f"role exact: {key}", roles.get(key) == expected_fold.get(key))
    check(
        "role cardinalities", (len(discovery), len(selection), len(outer)) == (28, 7, 7)
    )
    check(
        "role disjointness",
        not set(discovery) & set(selection)
        and not (set(discovery) | set(selection)) & set(outer),
    )
    check("condition roster", tuple(result.get("conditions", [])) == R006V0F_CONDITIONS)

    candidates = result.get("candidates", {})
    check(
        "candidate rank roster",
        set(candidates) == {str(rank) for rank in R006V0F_RANKS},
    )
    expected_keys = _expected_sample_keys(selection)
    for rank in R006V0F_RANKS:
        seeds = candidates.get(str(rank), {})
        check(
            f"rank{rank}: seed roster",
            set(seeds) == {str(seed) for seed in R006V0F_SEEDS},
        )
        for seed in R006V0F_SEEDS:
            row = seeds.get(str(seed), {})
            prefix = f"rank{rank}/seed{seed}"
            basis_path = result_dir / str(row.get("basis_file", "__missing__"))
            check(f"{prefix}: basis exists", basis_path.is_file())
            if basis_path.is_file():
                basis = np.load(basis_path, allow_pickle=False)
                check(f"{prefix}: basis shape", basis.shape == (256, rank))
                check(f"{prefix}: basis finite", bool(np.isfinite(basis).all()))
                if basis.shape == (256, rank) and np.isfinite(basis).all():
                    gram = basis.astype(np.float64).T @ basis.astype(np.float64)
                    check(
                        f"{prefix}: basis orthonormal",
                        float(np.max(np.abs(gram - np.eye(rank)))) <= 2e-5,
                    )
                check(
                    f"{prefix}: basis hash",
                    sha256_file(basis_path) == row.get("basis_sha256"),
                )
            check(f"{prefix}: recorded shape", row.get("basis_shape") == [256, rank])
            check(
                f"{prefix}: discovery count", row.get("discovery_sample_count") == 1344
            )
            check(
                f"{prefix}: selection count", row.get("selection_sample_count") == 336
            )
            check(
                f"{prefix}: random-only initialization",
                row.get("solver_initialization") == f"random_{seed}",
            )
            check(
                f"{prefix}: solver finite",
                math.isfinite(float(row.get("discovery_objective", math.nan)))
                and int(row.get("solver_steps", -1)) >= 0,
            )

            samples = row.get("selection_samples", [])
            observed_keys = {
                (
                    str(item.get("recording_group")),
                    int(item.get("anchor_index", -1)),
                    str(item.get("condition")),
                )
                for item in samples
            }
            check(f"{prefix}: sample count", len(samples) == 336)
            check(
                f"{prefix}: sample roster exact",
                observed_keys == expected_keys and len(observed_keys) == len(samples),
            )
            totals = torch.as_tensor(
                [item.get("total_task_drift_sq", math.nan) for item in samples],
                dtype=torch.float64,
            )
            uncovered = torch.as_tensor(
                [item.get("uncovered_task_drift_sq", math.nan) for item in samples],
                dtype=torch.float64,
            )
            valid = bool(
                len(samples) == 336
                and torch.isfinite(totals).all()
                and torch.isfinite(uncovered).all()
                and torch.all(totals >= 0)
                and torch.all(uncovered >= 0)
            )
            check(f"{prefix}: finite nonnegative samples", valid)
            if not valid:
                continue
            summary = summarize_basis_coverage(
                totals,
                uncovered,
                [str(item["recording_group"]) for item in samples],
                rank=rank,
                epsilon=1e-12,
                mean_threshold=0.90,
                worst_group_threshold=0.50,
            )
            recorded = row.get("selection_summary", {})
            check(f"{prefix}: summary rank", recorded.get("rank") == rank)
            check(f"{prefix}: summary group count", recorded.get("group_count") == 7)
            check(
                f"{prefix}: group-equal mean",
                _close(
                    float(recorded.get("group_equal_mean", math.nan)),
                    summary.group_equal_mean,
                ),
            )
            check(
                f"{prefix}: worst-group mean",
                _close(
                    float(recorded.get("worst_group_mean", math.nan)),
                    summary.worst_group_mean,
                ),
            )
            check(
                f"{prefix}: qualification",
                recorded.get("qualifies") is summary.qualifies,
            )
            recorded_means = recorded.get("group_means", {})
            check(
                f"{prefix}: group means",
                set(recorded_means) == set(summary.group_means)
                and all(
                    _close(float(recorded_means[group]), value)
                    for group, value in summary.group_means.items()
                ),
            )
    return checks
