import numpy as np

from aorta_drive.vision.r006_failure_diagnostic import audit_smoke_replay


def _basis(rank: int = 4) -> np.ndarray:
    value = np.zeros((256, rank), dtype=np.float32)
    value[:rank] = np.eye(rank, dtype=np.float32)
    return value


def test_smoke_replay_accepts_exact_finite_orthonormal_basis() -> None:
    basis = _basis()
    result = audit_smoke_replay(basis, basis.copy(), expected_rank=4)
    assert result["status"] == "PASS"
    assert result["exact_primary_replay"] is True


def test_smoke_replay_rejects_nonexact_or_nonfinite_basis() -> None:
    primary = _basis()
    replay = primary.copy()
    replay[10, 0] = np.nan
    result = audit_smoke_replay(primary, replay, expected_rank=4)
    assert result["status"] == "FAIL"
    assert result["replay_finite"] is False
