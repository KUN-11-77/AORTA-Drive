"""Unit tests for the forward96 exact-teacher qualification."""

from __future__ import annotations

import pytest

from audit_r004h7bs4b0a_forward96_labels import _validate_manifest
from build_r004h7bs4b0a_forward96_labels import forward96_grid


def _manifest() -> dict[str, object]:
    records = [
        {
            "cache_id": f"c/{index}",
            "node_count": 6144,
        }
        for index in range(164)
    ]
    from audit_r004h7bs4b0a_forward96_labels import _logical_hash

    return {
        "status": "completed",
        "records": records,
        "logical_manifest_sha256": _logical_hash(records),
        "protocol": {
            "grid": {
                "x_support_m": [-32.0, 64.0],
                "y_support_m": [-32.0, 32.0],
                "resolution_m": 1.0,
                "rows": 64,
                "columns": 96,
                "nodes": 6144,
                "undirected_edges": 12128,
            }
        },
    }


def test_forward96_grid_is_full_horizon_asymmetric_domain() -> None:
    grid = forward96_grid()
    assert grid.x_min == -32.0
    assert grid.x_max == 64.0
    assert grid.y_min == -32.0
    assert grid.y_max == 32.0
    assert grid.rows == 64
    assert grid.columns == 96
    assert grid.centers().shape == (6144, 2)


def test_manifest_validator_accepts_exact_contract() -> None:
    assert len(_validate_manifest(_manifest())) == 164


def test_manifest_validator_rejects_wrong_grid() -> None:
    payload = _manifest()
    payload["protocol"]["grid"]["columns"] = 64  # type: ignore[index]
    with pytest.raises(ValueError, match="grid protocol"):
        _validate_manifest(payload)
