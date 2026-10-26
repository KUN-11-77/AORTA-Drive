from __future__ import annotations

from copy import deepcopy

from audit_r005e3c_gradient_mass import audit


FROZEN_DATA = {
    "bundle_manifest_sha256": "bundle",
    "visual_manifest_sha256": "visual",
    "crossfit_manifest_sha256": "crossfit",
    "mismatch_manifest_sha256": "mismatch",
    "r005e3b_primary_predictions_sha256": "predictions",
}


def _category(count: int, *, utility: float, gradient: float) -> dict[str, object]:
    return {
        "count": count,
        "utility_sum": utility,
        "utility_abs_sum": abs(utility),
        "probability_sum": 1.0,
        "score_gradient_sum": gradient,
        "score_gradient_abs_sum": abs(gradient),
        "score_gradient_abs_mean": abs(gradient) / count,
    }


def _score() -> dict[str, object]:
    return {
        "anchor_count": 560,
        "valid_challenger_count": 2,
        "opportunity_anchor_count": 1,
        "opportunity_anchor_fraction": 1.0 / 560.0,
        "positive_challenger_fraction": 0.5,
        "positive": _category(1, utility=0.2, gradient=-0.1),
        "nonpositive": _category(1, utility=-0.3, gradient=0.1),
        "positive_raise_direction_fraction": 1.0,
        "nonpositive_lower_direction_fraction": 1.0,
    }


def _result() -> dict[str, object]:
    score = _score()
    return {
        "schema": "aorta.r005e3b.gradient_mass_diagnostic.v1",
        "status": "COMPLETED",
        "completed_at_utc": "now",
        "outer_fold": 0,
        "seed": 20261025,
        "temperature": 0.125,
        "initial_model_state_sha256": "initial",
        "weights_updated": False,
        "model_state_unchanged": True,
        "development_calibration_test_navdream_target_accessed": False,
        "input_sha256": {
            "bundle_manifest_sha256": "bundle",
            "visual_manifest_sha256": "visual",
            "crossfit_manifest_sha256": "crossfit",
            "mismatch_manifest_sha256": "mismatch",
            "r005e3b_primary_predictions_sha256": "predictions",
            "config": "config",
        },
        "runner_sha256": "runner",
        "code_commit": "commit",
        "physical_gpu": {"physical_index": 0, "uuid": "same"},
        "initial_score_diagnostics": deepcopy(score),
        "final_score_diagnostics": deepcopy(score),
        "initial_parameter_gradient_decomposition": {
            "anchor_count": 560,
            "positive_parameter_gradient_norm": 2.0,
            "nonpositive_parameter_gradient_norm": 4.0,
            "nonpositive_to_positive_norm_ratio": 2.0,
            "positive_nonpositive_cosine": -0.5,
            "combined_parameter_gradient_norm": 3.4641016151377544,
        },
    }


def _audit(primary: dict[str, object], replay: dict[str, object]) -> dict[str, object]:
    return audit(
        primary,
        replay,
        config_sha256="config",
        runner_sha256="runner",
        expected_commit="commit",
        frozen_data=FROZEN_DATA,
        frozen_initialization_sha256="initial",
        recomputed_final=_score(),
    )


def test_audit_accepts_complete_exact_replay_with_different_timestamp() -> None:
    primary = _result()
    replay = deepcopy(primary)
    replay["completed_at_utc"] = "later"
    result = _audit(primary, replay)
    assert result["status"] == "PASS"


def test_audit_rejects_scientific_drift() -> None:
    primary = _result()
    replay = deepcopy(primary)
    cast_score = replay["final_score_diagnostics"]
    assert isinstance(cast_score, dict)
    cast_score["opportunity_anchor_count"] = 2
    assert _audit(primary, replay)["status"] == "FAIL"


def test_audit_rejects_missing_scientific_payload_even_if_replay_matches() -> None:
    primary = _result()
    for key in (
        "initial_score_diagnostics",
        "final_score_diagnostics",
        "initial_parameter_gradient_decomposition",
    ):
        primary.pop(key)
    replay = deepcopy(primary)
    assert _audit(primary, replay)["status"] == "FAIL"


def test_audit_rejects_jointly_tampered_frozen_contract() -> None:
    primary = _result()
    primary["outer_fold"] = 99
    primary["seed"] = 1
    primary["temperature"] = 999.0
    input_sha = primary["input_sha256"]
    assert isinstance(input_sha, dict)
    input_sha["bundle_manifest_sha256"] = "not-a-hash"
    replay = deepcopy(primary)
    assert _audit(primary, replay)["status"] == "FAIL"


def test_audit_rejects_impossible_count_arithmetic() -> None:
    primary = _result()
    initial = primary["initial_score_diagnostics"]
    assert isinstance(initial, dict)
    positive = initial["positive"]
    assert isinstance(positive, dict)
    positive["count"] = -100
    replay = deepcopy(primary)
    assert _audit(primary, replay)["status"] == "FAIL"
