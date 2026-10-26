from audit_r005e3c_gradient_mass import audit


def _result() -> dict[str, object]:
    return {
        "schema": "aorta.r005e3b.gradient_mass_diagnostic.v1",
        "status": "COMPLETED",
        "completed_at_utc": "now",
        "weights_updated": False,
        "model_state_unchanged": True,
        "development_calibration_test_navdream_target_accessed": False,
        "input_sha256": {"config": "config"},
        "runner_sha256": "runner",
        "code_commit": "commit",
        "physical_gpu": {"uuid": "same"},
        "value": 1.0,
    }


def test_audit_accepts_exact_replay_with_different_timestamp() -> None:
    primary = _result()
    replay = _result()
    replay["completed_at_utc"] = "later"
    result = audit(
        primary,
        replay,
        config_sha256="config",
        runner_sha256="runner",
        expected_commit="commit",
    )
    assert result["status"] == "PASS"


def test_audit_rejects_scientific_drift() -> None:
    primary = _result()
    replay = _result()
    replay["value"] = 2.0
    result = audit(
        primary,
        replay,
        config_sha256="config",
        runner_sha256="runner",
        expected_commit="commit",
    )
    assert result["status"] == "FAIL"
