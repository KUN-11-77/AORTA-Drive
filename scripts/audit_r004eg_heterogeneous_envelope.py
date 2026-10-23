"""Run the registered R004E-G heterogeneous-envelope synthetic audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.certification.calibration import split_conformal_threshold
from aorta_drive.certification.graph import UndirectedEdge, robust_set_min_lower_bound
from aorta_drive.certification.heterogeneous import (
    GroupRoles,
    deterministic_group_roles,
    outward_abs_edge_residuals,
    outward_edge_budgets,
    outward_session_max_score,
    outward_standardized_scores,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_train_groups(path: Path, split: str) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = [row["recording_group"] for row in rows if row["split"] == split]
    if len(groups) != len(set(groups)):
        raise ValueError("recording group manifest contains duplicate train groups")
    return groups


def _role_payload(roles: GroupRoles, exposure: str) -> dict[str, list[dict[str, str]]]:
    def records(role: str, groups: tuple[str, ...]) -> list[dict[str, str]]:
        return [
            {"recording_group": group, "role": role, "historical_exposure": exposure}
            for group in groups
        ]

    return {
        "fit": records("fit", roles.fit),
        "selection": records("selection", roles.selection),
        "proxy_cal": records("proxy_cal", roles.proxy_cal),
    }


def _rank_coverage(n: int, alpha: float) -> tuple[int, int]:
    successes = 0
    for new_rank in range(n + 1):
        calibration = [float(rank) for rank in range(n + 1) if rank != new_rank]
        threshold = split_conformal_threshold(calibration, alpha)
        successes += int(float(new_rank) <= threshold)
    return successes, n + 1


def _fraction_arithmetic_audit(
    *,
    seed: int,
    residual_cases: int,
    budget_cases: int,
    scale_min: float,
    scale_max: float,
) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    residual_passed = 0
    for _ in range(residual_cases):
        values = rng.normal(size=4) * 2.0 ** int(rng.integers(-30, 31))
        scale = float(2.0 ** int(rng.integers(-9, 5)))
        potentials = np.array([[values[0], values[1]]])
        labels = np.array([[values[2], values[3]]])
        upper = outward_abs_edge_residuals(potentials, labels, [0], [1])
        score = outward_standardized_scores(
            upper,
            np.array([[scale]]),
            scale_min=scale_min,
            scale_max=scale_max,
        )
        exact_residual = abs(
            (Fraction.from_float(float(values[1])) - Fraction.from_float(float(values[0])))
            - (Fraction.from_float(float(values[3])) - Fraction.from_float(float(values[2])))
        )
        exact_score = exact_residual / Fraction.from_float(scale)
        if Fraction.from_float(float(upper[0, 0])) < exact_residual:
            raise AssertionError("four-term residual enclosure was optimistic")
        if Fraction.from_float(float(score[0, 0])) < exact_score:
            raise AssertionError("standardized score was optimistic")
        residual_passed += 1

    budget_passed = 0
    for _ in range(budget_cases):
        threshold = float(2.0 ** int(rng.integers(-30, 31)) * rng.uniform(0.5, 1.0))
        scale = float(2.0 ** int(rng.integers(-9, 5)) * rng.uniform(0.5, 1.0))
        scale = min(max(scale, scale_min), scale_max)
        budget = outward_edge_budgets(
            threshold,
            [scale],
            scale_min=scale_min,
            scale_max=scale_max,
        )[0]
        exact = Fraction.from_float(threshold) * Fraction.from_float(scale)
        if Fraction.from_float(float(budget)) < exact:
            raise AssertionError("edge budget multiplication was optimistic")
        budget_passed += 1
    return {
        "randomized_residual_cases_passed": residual_passed,
        "randomized_budget_cases_passed": budget_passed,
    }


def _cancellation_audit(scale_min: float, scale_max: float) -> dict[str, float]:
    potentials = np.array([[-2.0**-53, 1.0]])
    labels = np.array([[0.0, 1.0]])
    upper = outward_abs_edge_residuals(potentials, labels, [0], [1])
    score = outward_standardized_scores(
        upper,
        np.array([[1.0]]),
        scale_min=scale_min,
        scale_max=scale_max,
    )
    naive = abs(
        (potentials[0, 1] - potentials[0, 0])
        - (labels[0, 1] - labels[0, 0])
    )
    exact = 2.0**-53
    if naive != 0.0 or float(upper[0, 0]) < exact or float(score[0, 0]) < exact:
        raise AssertionError("registered cancellation counterexample was not enclosed")
    return {
        "exact_residual": exact,
        "naive_residual": float(naive),
        "outward_residual_upper": float(upper[0, 0]),
        "outward_score_upper": float(score[0, 0]),
    }


def _gauge_audit(
    scale_min: float,
    scale_max: float,
    factor: float,
    max_ulp: int,
) -> dict[str, Any]:
    potentials = np.array([[0.0, 0.25, 0.75]])
    labels = np.zeros_like(potentials)
    sources = np.array([0, 1])
    targets = np.array([1, 2])
    scales = np.array([[0.5, 1.0]])
    score = outward_session_max_score(
        potentials,
        labels,
        sources,
        targets,
        scales,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    rescaled_score = outward_session_max_score(
        potentials,
        labels,
        sources,
        targets,
        scales * factor,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    budgets = outward_edge_budgets(
        score,
        scales,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    rescaled = outward_edge_budgets(
        rescaled_score,
        scales * factor,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    ulp_differences = [
        abs(float(left) - float(right)) / max(math.ulp(float(left)), math.ulp(float(right)))
        for left, right in zip(budgets[0], rescaled[0], strict=True)
    ]
    if max(ulp_differences) > max_ulp:
        raise AssertionError("interior power-of-two scale gauge exceeded ULP tolerance")
    return {
        "score": score,
        "rescaled_score": rescaled_score,
        "budgets": budgets[0].tolist(),
        "rescaled_budgets": rescaled[0].tolist(),
        "ulp_differences": ulp_differences,
    }


def _graph_audit(scale_min: float, scale_max: float) -> dict[str, float]:
    budgets = outward_edge_budgets(
        0.75,
        [0.5, 2.0],
        scale_min=scale_min,
        scale_max=scale_max,
    )
    values = [0.2, -0.1, 0.7]
    edges = [
        UndirectedEdge(0, 1, float(budgets[0])),
        UndirectedEdge(1, 2, float(budgets[1])),
    ]
    computed = robust_set_min_lower_bound(values, edges, (2,), (0,))
    exact = (
        Fraction.from_float(values[2])
        - Fraction.from_float(values[0])
        - Fraction.from_float(float(budgets[0]))
        - Fraction.from_float(float(budgets[1]))
    )
    if Fraction.from_float(computed) > exact:
        raise AssertionError("graph lower bound was optimistic")
    return {"computed_lower_bound": computed, "exact_lower_bound": float(exact)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    if _sha256(args.manifest) != frozen["recording_group_manifest_sha256"]:
        raise ValueError("recording group manifest hash disagrees with registration")
    groups = _load_train_groups(args.manifest, str(frozen["allowed_manifest_split"]))
    role_config = config["roles"]
    roles = deterministic_group_roles(
        groups,
        namespace=str(role_config["namespace"]),
        fit_count=int(role_config["fit_count"]),
        selection_count=int(role_config["selection_count"]),
        proxy_cal_count=int(role_config["proxy_cal_count"]),
    )
    scale_config = config["scale_contract"]
    scale_min = float(scale_config["minimum"])
    scale_max = float(scale_config["maximum"])
    oracle = config["oracle"]
    arithmetic = _fraction_arithmetic_audit(
        seed=int(oracle["random_seed"]),
        residual_cases=int(oracle["randomized_four_term_cases"]),
        budget_cases=int(oracle["randomized_budget_cases"]),
        scale_min=scale_min,
        scale_max=scale_max,
    )
    alpha = float(config["conformal_contract"]["proxy_alpha"])
    rank_successes, rank_total = _rank_coverage(len(roles.proxy_cal), alpha)
    if rank_successes != 16 or rank_total != 17:
        raise AssertionError("joint exchangeable rank enumeration disagrees with registration")

    result = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "decision": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": _sha256(args.config),
        "manifest_sha256": _sha256(args.manifest),
        "data_access": "recording_group_ID_metadata_only",
        "roles": _role_payload(roles, str(role_config["historical_exposure_label"])),
        "role_counts": {
            "fit": len(roles.fit),
            "selection": len(roles.selection),
            "proxy_cal": len(roles.proxy_cal),
        },
        "proxy_rank_oracle": {
            "successes": rank_successes,
            "total_joint_ranks": rank_total,
            "empirical_fraction": rank_successes / rank_total,
            "coverage_claim_authorized": False,
        },
        "arithmetic": arithmetic,
        "cancellation": _cancellation_audit(scale_min, scale_max),
        "gauge": _gauge_audit(
            scale_min,
            scale_max,
            float(oracle["gauge_rescaling_factor"]),
            int(oracle["gauge_max_ulp_difference"]),
        ),
        "graph": _graph_audit(scale_min, scale_max),
        "tail_loss_counterexample": {
            "absolute_residual": 2.0,
            "tail_weight": 0.5,
            "proper_stationary_scale": 2.0,
            "tail_augmented_stationary_scale": 3.0,
            "tail_term_allowed_in_stage_s": False,
        },
        "claims": {
            "primitive_protocol_qualified": True,
            "learned_uncertainty_established": False,
            "historical_proxy_coverage_established": False,
            "gate_b_passed": False,
            "algorithmic_novelty_established": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
