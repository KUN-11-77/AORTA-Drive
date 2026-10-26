from __future__ import annotations

import csv
import subprocess

import pytest
import torch

from aorta_drive.vision.ltf_causal_runtime import apply_controlled_appearance
from aorta_drive.vision.r006_cache import (
    STAGE_SITE_SEMANTICS,
    load_source_fit_rows,
    validate_group_shard,
)

from scripts.build_r006v0_activation_cache import git_revision


def test_controlled_appearances_are_deterministic_and_bounded():
    image = torch.linspace(0, 1, 24).reshape(1, 3, 2, 4)
    for condition in ("low_exposure", "low_contrast", "cool_cast"):
        first = apply_controlled_appearance(image, condition)
        second = apply_controlled_appearance(image, condition)
        assert torch.equal(first, second)
        assert torch.all((0 <= first) & (first <= 1))
        assert not torch.equal(first, image)


def test_load_source_fit_rows_requires_exact_group_structure(tmp_path):
    roles = tmp_path / "roles.csv"
    time_set = tmp_path / "time.csv"
    groups = [f"g{index:02d}" for index in range(42)]
    with roles.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["recording_group", "role"])
        writer.writeheader()
        writer.writerows({"recording_group": group, "role": "fit"} for group in groups)
    with time_set.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["recording_group", "anchor_index", "scene_token", "log_name"],
        )
        writer.writeheader()
        for group in reversed(groups):
            for anchor in reversed(range(16)):
                writer.writerow(
                    {
                        "recording_group": group,
                        "anchor_index": anchor,
                        "scene_token": f"{group}_{anchor}",
                        "log_name": group,
                    }
                )
    rows = load_source_fit_rows(roles, time_set)
    assert len(rows) == 672
    assert rows[0]["recording_group"] == "g00"
    assert int(rows[0]["anchor_index"]) == 0


def test_validate_group_shard_rejects_missing_condition():
    activations = {
        "c3_planner_memory": torch.zeros(1, 65, 256),
        "c4_trajectory_query": torch.zeros(1, 1, 256),
    }
    conditions = {
        condition: {"trajectory": torch.zeros(1, 8, 3), "activations": activations}
        for condition in ("clean", "low_exposure", "low_contrast", "cool_cast")
    }
    payload = {
        "artifact": "R006V0_SOURCE_FIT_ACTIVATION_SHARD_V1",
        "role": "fit",
        "recording_group": "g",
        "anchors": [
            {
                "recording_group": "g",
                "anchor_index": index,
                "scene_token": f"t{index}",
                "conditions": conditions,
            }
            for index in range(16)
        ],
    }
    assert validate_group_shard(payload)["c4_trajectory_query"] == [1, 1, 256]
    assert STAGE_SITE_SEMANTICS["c3_planner_memory"]["frozen_site_indices"] == [64]
    del payload["anchors"][0]["conditions"]["cool_cast"]
    with pytest.raises(ValueError, match="condition roster"):
        validate_group_shard(payload)


def test_git_revision_distinguishes_execution_subtree_from_unrelated_dirty_files(
    tmp_path,
):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True
    )
    navsim = tmp_path / "navsim"
    navsim.mkdir()
    (navsim / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("clean\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "navsim", "notes.md"], check=True
    )
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "fixture"], check=True)

    (tmp_path / "notes.md").write_text("dirty but unrelated\n", encoding="utf-8")
    revision = git_revision(tmp_path)
    assert revision["dirty"] is True
    assert revision["execution_subtree"] == "navsim"
    assert revision["execution_subtree_dirty"] is False
    assert revision["execution_subtree_tree"]

    (navsim / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert git_revision(tmp_path)["execution_subtree_dirty"] is True
