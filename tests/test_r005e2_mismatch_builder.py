from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from build_r005e2_mismatch_manifest import _resolve_panel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resolve_panel_uses_exact_hash_across_base_and_replacement(tmp_path: Path) -> None:
    base = tmp_path / "base"
    replacement = tmp_path / "replacement"
    base.mkdir()
    replacement.mkdir()
    (base / "scene.npz").write_bytes(b"stale")
    expected = replacement / "scene.npz"
    expected.write_bytes(b"registered replacement")
    path, root_index = _resolve_panel(
        [base, replacement],
        scene_token="scene",
        expected_sha256=_sha256(expected),
    )
    assert path == expected
    assert root_index == 1


def test_resolve_panel_rejects_missing_or_wrong_hash(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "scene.npz").write_bytes(b"wrong")
    with pytest.raises(FileNotFoundError, match="wrong-hash"):
        _resolve_panel(
            [root],
            scene_token="scene",
            expected_sha256=hashlib.sha256(b"expected").hexdigest(),
        )


def test_resolve_panel_accepts_identical_exact_copies_deterministically(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        (root / "scene.npz").write_bytes(b"identical")
    path, root_index = _resolve_panel(
        [first, second],
        scene_token="scene",
        expected_sha256=_sha256(first / "scene.npz"),
    )
    assert path == first / "scene.npz"
    assert root_index == 0
