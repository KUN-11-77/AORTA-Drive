from __future__ import annotations

import pytest

from build_r004h7bs4b0a2_fit_forward96_labels import forward96_grid, select_fit_records


def _payloads() -> tuple[dict[str, object], dict[str, object]]:
    anchors = [
        {
            "log_name": f"log-{index}",
            "scene_token": f"token-{index}",
            "recording_group": f"group-{index // 16}",
        }
        for index in range(672)
    ]
    metrics = [
        {
            "log_name": f"log-{index}",
            "scene_token": f"token-{index}",
            "recording_group": f"group-{index // 16}",
            "split": "train",
        }
        for index in range(672)
    ]
    metrics.extend(
        {
            "log_name": f"extra-{index}",
            "scene_token": f"extra-token-{index}",
            "recording_group": f"extra-group-{index}",
            "split": "development",
        }
        for index in range(768)
    )
    return {"anchor_ledgers": anchors}, {"records": metrics}


def test_forward96_grid_is_exact() -> None:
    grid = forward96_grid()
    assert (grid.rows, grid.columns, grid.resolution) == (64, 96, 1.0)
    assert (grid.x_min, grid.x_max, grid.y_min, grid.y_max) == (-32.0, 64.0, -32.0, 32.0)


def test_exact_fit_join_accepts_672_train_anchors() -> None:
    anchors, metrics = _payloads()
    selected = select_fit_records(anchors, metrics)
    assert len(selected) == 672
    assert len({record["recording_group"] for record in selected}) == 42


def test_exact_fit_join_rejects_role_pollution() -> None:
    anchors, metrics = _payloads()
    metrics["records"][0]["split"] = "development"  # type: ignore[index]
    with pytest.raises(ValueError, match="non-train"):
        select_fit_records(anchors, metrics)
