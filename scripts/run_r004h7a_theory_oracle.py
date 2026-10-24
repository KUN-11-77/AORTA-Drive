"""Run the registered data-free R004H7-A theory oracle."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.theory.decision_quotient import (
    cut_space_projection,
    decision_error_eta,
    incidence_matrix,
    recover_zero_mean_potential,
    selection_comparison,
    set_min_contrasts,
    set_min_oscillation_bounds,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subsets(node_count: int) -> list[tuple[int, ...]]:
    return [
        subset
        for size in range(1, node_count + 1)
        for subset in itertools.combinations(range(node_count), size)
    ]


def run_oracle() -> dict[str, int]:
    """Execute every registered finite oracle and return case counts."""

    potentials = [
        np.asarray(values, dtype=np.float64)
        for values in itertools.product((-1.0, 0.0, 1.0), repeat=3)
    ]
    subsets = _subsets(3)
    offset_cases = 0
    oscillation_cases = 0
    regret_cases = 0

    for first in potentials:
        for native in subsets:
            for challenger in subsets:
                two_candidate_panel = (native, challenger)
                base = set_min_contrasts(first, two_candidate_panel)
                for offset in (-2.0, 0.5, 3.0):
                    offset_cases += 1
                    if not np.array_equal(
                        base,
                        set_min_contrasts(first + offset, two_candidate_panel),
                    ):
                        raise AssertionError("common-offset invariance failed")
                for second in potentials:
                    oscillation_cases += 1
                    error = np.abs(
                        set_min_contrasts(first, two_candidate_panel)
                        - set_min_contrasts(second, two_candidate_panel)
                    )
                    bound = set_min_oscillation_bounds(
                        first,
                        second,
                        two_candidate_panel,
                    )
                    if not np.all(error <= np.nextafter(bound, np.inf)):
                        raise AssertionError("set-min oscillation bound failed")

    for teacher in potentials:
        for predicted in potentials:
            for native in subsets:
                for first_challenger in subsets:
                    for second_challenger in subsets:
                        regret_cases += 1
                        three_candidate_panel = (
                            native,
                            first_challenger,
                            second_challenger,
                        )
                        teacher_margin = set_min_contrasts(teacher, three_candidate_panel)
                        predicted_margin = set_min_contrasts(
                            predicted,
                            three_candidate_panel,
                        )
                        comparison = selection_comparison(teacher_margin, predicted_margin)
                        eta = decision_error_eta(teacher_margin, predicted_margin)
                        if comparison.regret > np.nextafter(2.0 * eta, np.inf):
                            raise AssertionError("two-eta regret bound failed")

    incidence = incidence_matrix(3, ((0, 1), (1, 2), (0, 2)))
    projection_cases = 0
    for potential in potentials:
        projection_cases += 1
        recovered = recover_zero_mean_potential(incidence @ potential, incidence)
        expected = potential - np.mean(potential)
        if not np.allclose(recovered, expected, rtol=0.0, atol=1e-12):
            raise AssertionError("potential recovery failed")
    raw_edge = np.asarray((1.0, -2.0, 0.25), dtype=np.float64)
    residual = raw_edge - cut_space_projection(raw_edge, incidence)
    if not np.allclose(incidence.T @ residual, 0.0, rtol=0.0, atol=1e-12):
        raise AssertionError("cut-space orthogonality failed")

    nonlinear_panel = ((0,), (1, 2))
    first = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    second = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    if (
        set_min_contrasts(first + second, nonlinear_panel)[0]
        == set_min_contrasts(first, nonlinear_panel)[0]
        + set_min_contrasts(second, nonlinear_panel)[0]
    ):
        raise AssertionError("registered global nonlinearity counterexample failed")

    native_tie = selection_comparison((0.0, -1.0), (0.0, -1.0))
    challenger_tie = selection_comparison((1.0, 1.0), (1.0, 1.0))
    if native_tie.teacher_index != 0 or challenger_tie.teacher_index != 1:
        raise AssertionError("deterministic tie policy failed")

    return {
        "offset_invariance_cases": offset_cases,
        "oscillation_bound_cases": oscillation_cases,
        "two_eta_regret_cases": regret_cases,
        "integrable_projection_cases": projection_cases,
        "nonlinearity_counterexamples": 1,
        "tie_policy_cases": 2,
        "failure_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["experiment_id"] != "R004H7-decision-quotient-structure-necessity-v1":
        raise ValueError("unexpected experiment_id")
    if config["H7A"]["data_access"] != "none":
        raise ValueError("H7A must remain data-free")

    result: dict[str, Any] = {
        "experiment_id": "R004H7A-decision-quotient-theory-v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "config_sha256": _sha256(args.config),
        "script_sha256": _sha256(Path(__file__)),
        "data_access": "none",
        "oracle_counts": run_oracle(),
        "claim_boundary": (
            "finite-graph theory qualification only; no visual, driving, OOD, or safety evidence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
