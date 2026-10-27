"""Independent mechanical audit primitives for R006V0 fold selection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .r006_cache import sha256_file
from .r006_protocol import (
    CoverageSummary,
    select_registered_interface,
    selection_as_dict,
    summarize_basis_coverage,
)


LAYERS = ("planner_memory", "trajectory_query_secondary")
RANKS = (4, 8, 16, 32)
CONDITIONS = ("low_exposure", "low_contrast", "cool_cast")


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance


def _selection_matches(
    recorded: object,
    expected: Mapping[str, Any] | None,
) -> bool:
    """Compare the discrete decision exactly and its statistics numerically.

    JSON serialization and an independent float64 reduction may differ by one
    unit in the last place.  Such a difference cannot change the registered
    layer/rank decision and must not invalidate an otherwise exact replay.
    """

    if expected is None:
        return recorded is None
    if not isinstance(recorded, Mapping):
        return False
    return bool(
        recorded.get("layer") == expected.get("layer")
        and recorded.get("rank") == expected.get("rank")
        and recorded.get("rule") == expected.get("rule")
        and _close(
            float(recorded.get("group_equal_mean", math.nan)),
            float(expected.get("group_equal_mean", math.nan)),
        )
        and _close(
            float(recorded.get("worst_group_mean", math.nan)),
            float(expected.get("worst_group_mean", math.nan)),
        )
    )


def _expected_sample_keys(groups: Sequence[str]) -> set[tuple[str, int, str]]:
    return {
        (str(group), anchor, condition)
        for group in groups
        for anchor in range(16)
        for condition in CONDITIONS
    }


def audit_fold_basis_result(
    result: Mapping[str, Any],
    *,
    result_dir: Path,
    expected_fold: Mapping[str, Any],
) -> tuple[list[dict[str, object]], dict[str, dict[int, CoverageSummary]]]:
    """Recompute all fold-selection decisions without trusting the runner summary."""

    checks: list[dict[str, object]] = []

    def check(name: str, value: bool) -> None:
        checks.append({"check": name, "pass": bool(value)})

    check("artifact", result.get("artifact") == "R006V0_FOLD_BASIS_SELECTION_V1")
    check("mechanical status", result.get("mechanical_status") == "PASS")
    check("scientific metrics declared", result.get("scientific_metrics_emitted") is True)
    check(
        "outer confirmation metrics absent",
        result.get("outer_confirmation_metrics_emitted") is False,
    )
    check(
        "sealed roles untouched",
        result.get("development_calibration_test_navdream_target_accessed") is False,
    )
    check("registered layers", tuple(result.get("registered_layers", [])) == LAYERS)
    check("registered ranks", tuple(result.get("registered_ranks", [])) == RANKS)
    check("fixed batch size", result.get("streaming_batch_size") == 128)
    check("fixed output metric", result.get("output_metric") == "W=I_(2T)/T")

    roles = result.get("roles", {})
    role_mapping = {
        "basis_discovery_groups": "basis_discovery_groups",
        "layer_rank_selection_groups": "layer_rank_selection_groups",
        "outer_confirmation_groups": "outer_confirmation_groups",
    }
    for result_key, fold_key in role_mapping.items():
        check(
            f"role exact: {result_key}",
            roles.get(result_key) == expected_fold.get(fold_key),
        )
    discovery = set(roles.get("basis_discovery_groups", []))
    selection = list(roles.get("layer_rank_selection_groups", []))
    outer = set(roles.get("outer_confirmation_groups", []))
    check("role cardinalities", (len(discovery), len(selection), len(outer)) == (28, 7, 7))
    check(
        "role disjointness",
        not discovery & set(selection) and not (discovery | set(selection)) & outer,
    )
    check("condition roster", tuple(result.get("conditions", [])) == CONDITIONS)

    expected_keys = _expected_sample_keys(selection)
    candidates = result.get("candidates", {})
    check("candidate layer roster", tuple(candidates) == LAYERS)
    summaries: dict[str, dict[int, CoverageSummary]] = {}
    for layer in LAYERS:
        rows = candidates.get(layer, {})
        check(f"{layer}: rank roster", set(rows) == {str(rank) for rank in RANKS})
        summaries[layer] = {}
        for rank in RANKS:
            row = rows.get(str(rank), {})
            prefix = f"{layer}/rank{rank}"
            basis_path = result_dir / str(row.get("basis_file", "__missing__"))
            check(f"{prefix}: basis exists", basis_path.is_file())
            if basis_path.is_file():
                basis = np.load(basis_path, allow_pickle=False)
                check(f"{prefix}: basis shape", basis.shape == (256, rank))
                check(f"{prefix}: basis finite", bool(np.isfinite(basis).all()))
                if basis.shape == (256, rank) and np.isfinite(basis).all():
                    gram = basis.astype(np.float64).T @ basis.astype(np.float64)
                    error = float(np.max(np.abs(gram - np.eye(rank))))
                    check(f"{prefix}: basis orthonormal", error <= 2e-5)
                check(f"{prefix}: basis hash", sha256_file(basis_path) == row.get("basis_sha256"))
            check(f"{prefix}: recorded basis shape", row.get("basis_shape") == [256, rank])
            check(f"{prefix}: discovery count", row.get("discovery_sample_count") == 1344)
            check(f"{prefix}: selection count", row.get("selection_sample_count") == 336)
            check(
                f"{prefix}: solver finite",
                math.isfinite(float(row.get("discovery_objective", math.nan)))
                and int(row.get("solver_steps", -1)) >= 0
                and bool(row.get("solver_initialization")),
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
            group_ids = [str(item.get("recording_group")) for item in samples]
            valid_values = bool(
                torch.isfinite(totals).all()
                and torch.isfinite(uncovered).all()
                and torch.all(totals >= 0)
                and torch.all(uncovered >= 0)
            )
            check(f"{prefix}: finite nonnegative samples", valid_values)
            if not valid_values or len(samples) != 336:
                continue
            recomputed = summarize_basis_coverage(
                totals,
                uncovered,
                group_ids,
                rank=rank,
                epsilon=1e-12,
                mean_threshold=0.90,
                worst_group_threshold=0.50,
            )
            summaries[layer][rank] = recomputed
            recorded = row.get("selection_summary", {})
            check(f"{prefix}: summary rank", recorded.get("rank") == rank)
            check(f"{prefix}: summary group count", recorded.get("group_count") == 7)
            check(
                f"{prefix}: group-equal mean",
                _close(
                    float(recorded.get("group_equal_mean", math.nan)),
                    recomputed.group_equal_mean,
                ),
            )
            check(
                f"{prefix}: worst-group mean",
                _close(
                    float(recorded.get("worst_group_mean", math.nan)),
                    recomputed.worst_group_mean,
                ),
            )
            check(f"{prefix}: qualification", recorded.get("qualifies") is recomputed.qualifies)
            recorded_means = recorded.get("group_means", {})
            check(f"{prefix}: group roster", set(recorded_means) == set(selection))
            check(
                f"{prefix}: group means",
                set(recorded_means) == set(recomputed.group_means)
                and all(
                    _close(float(recorded_means[group]), value)
                    for group, value in recomputed.group_means.items()
                ),
            )

    if all(set(summaries.get(layer, {})) == set(RANKS) for layer in LAYERS):
        try:
            expected_selection = selection_as_dict(select_registered_interface(summaries))
            expected_status = "PASS_SELECTED_INTERFACE"
        except ValueError as error:
            if str(error) != "no registered layer/rank satisfies coverage gates":
                raise
            expected_selection = None
            expected_status = "STOP_NO_QUALIFIED_INTERFACE"
        check(
            "selection decision exact",
            _selection_matches(result.get("selected_interface"), expected_selection),
        )
        check("status exact", result.get("status") == expected_status)
    else:
        check("all summaries recomputable", False)
    return checks, summaries
