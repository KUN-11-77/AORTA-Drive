"""Fail-closed tests for the R004H4P5D payload-free result audit."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from audit_r004h4p5d_result import audit


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_diagnosis_config_parses_and_pins_current_geometry() -> None:
    config = yaml.safe_load(
        (
            ROOT / "configs/experiment/r004h4p5d_disk_resolution_diagnosis.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["implementation"]["geometry_source_sha256"] == _sha256(
        ROOT / "src/aorta_drive/certification/geometry.py"
    )
    assert config["predecessor"]["raw_result_sha256"] == _sha256(
        ROOT / "experiments/results/R004H4P5I/result.json"
    )
    assert config["decision"]["prevalence_performance_coverage_or_gate_b_claim"] == (
        "prohibited"
    )


def _result() -> dict[str, object]:
    amplitudes = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
    rows = []
    for index in range(8):
        valid = index < 6
        rows.append(
            {
                "log_name": f"log-{index}",
                "scene_token": f"scene-{index}",
                "execution_deterministic_replay_bit_exact": True,
                "candidate_ledger": [
                    {
                        "amplitude_m": amplitude,
                        "predecessor_geometry_valid": valid,
                        "predecessor_bidirectional_zero": valid,
                        "maximum_paired_separation_upper_m": 1.5,
                        "no_witness_threshold_lower_m": 4.49,
                        "barrier_guarantees_zero_witnesses": True,
                        "barrier_explains_predecessor_zero": valid,
                    }
                    for amplitude in amplitudes
                ],
            }
        )
    return {
        "experiment_id": "R004H4P5D-disk-resolution-diagnosis-v1",
        "status": "COMPLETED",
        "decision": "RETIRE_DISK_NONINCLUSION_AND_QUALIFY_ORIENTED_RECTANGLE",
        "implementation_revision": "revision",
        "config_sha256": "config",
        "elapsed_seconds": 1.0,
        "implementation_failures": [],
        "diagnostic_counts": {
            "predecessor_bidirectional_zero_candidates": 36,
            "zero_candidates_explained_by_barrier": 36,
        },
        "maximum_paired_separation_upper_m": 1.5,
        "minimum_no_witness_threshold_lower_m": 4.49,
        "sentinel_ledger": rows,
        "claims": {
            "disk_certificate_resolution_barrier_verified": True,
            "physical_opportunity_absent": False,
            "prevalence_estimated": False,
            "method_performance_evaluated": False,
            "coverage_established": False,
            "gate_b_passed": False,
            "development_access_authorized": False,
        },
    }


def test_registered_result_passes() -> None:
    audited = audit(
        _result(), config_sha256="config", expected_implementation_revision="revision"
    )
    assert audited["status"] == "PASS"


def test_barrier_boolean_corruption_fails() -> None:
    result = deepcopy(_result())
    result["sentinel_ledger"][0]["candidate_ledger"][0][
        "barrier_guarantees_zero_witnesses"
    ] = False
    audited = audit(
        result, config_sha256="config", expected_implementation_revision="revision"
    )
    assert audited["status"] == "FAIL"


def test_forbidden_physical_absence_claim_fails() -> None:
    result = deepcopy(_result())
    result["claims"]["physical_opportunity_absent"] = True
    audited = audit(
        result, config_sha256="config", expected_implementation_revision="revision"
    )
    assert audited["status"] == "FAIL"


def test_nonfinite_separation_fails() -> None:
    result = deepcopy(_result())
    result["sentinel_ledger"][0]["candidate_ledger"][0][
        "maximum_paired_separation_upper_m"
    ] = float("nan")
    audited = audit(
        result, config_sha256="config", expected_implementation_revision="revision"
    )
    assert audited["status"] == "FAIL"
