"""Controlled checks for the result-independent R004 NAVSIM source index."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pytest


def _load_script_module():
    path = Path(__file__).parents[1] / "scripts" / "build_r004_source_index.py"
    spec = importlib.util.spec_from_file_location("build_r004_source_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INDEX = _load_script_module()


def _write_filter(path: Path, log_names: list[str], tokens: list[str]) -> None:
    lines = [
        "num_history_frames: 2",
        "num_future_frames: 1",
        "frame_interval: 1",
        "has_route: true",
        "log_names:",
        *[f"  - '{name}'" for name in log_names],
        "tokens:",
        *[f"  - '{token}'" for token in tokens],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_log(path: Path, tokens: list[str], route: bool = True) -> None:
    frames = [{"token": token, "roadblock_ids": ["road"] if route else []} for token in tokens]
    with path.open("wb") as stream:
        pickle.dump(frames, stream)


def test_build_index_reproduces_scene_filter_window_semantics(tmp_path: Path) -> None:
    log_name = "2021.06.14.18.33.41_veh-35_00001_00010"
    config = tmp_path / "filter.yaml"
    log_root = tmp_path / "logs"
    log_root.mkdir()
    _write_filter(config, [log_name], ["t1", "t2", "t3"])
    _write_log(log_root / f"{log_name}.pkl", ["t0", "t1", "t2", "t3", "t4"])

    payload = INDEX.build_index(config, log_root)

    assert payload["official_log_count"] == 1
    assert payload["official_token_count"] == 3
    assert payload["records"][0]["tokens"] == ["t1", "t2", "t3"]


def test_build_index_rejects_missing_or_unreachable_official_token(tmp_path: Path) -> None:
    log_name = "2021.06.14.18.33.41_veh-35_00001_00010"
    config = tmp_path / "filter.yaml"
    log_root = tmp_path / "logs"
    log_root.mkdir()
    _write_filter(config, [log_name], ["does-not-exist"])
    _write_log(log_root / f"{log_name}.pkl", ["t0", "t1", "t2"])

    with pytest.raises(ValueError, match="source token mismatch"):
        INDEX.build_index(config, log_root)
