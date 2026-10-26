"""Source-fit activation-cache contracts for R006V0.

This module contains only role filtering, deterministic row ordering and
mechanical cache validation.  It does not compute mediation statistics.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .ltf_causal_runtime import CONTROLLED_APPEARANCES


QUALIFIED_STAGES = ("c3_planner_memory", "c4_trajectory_query")
STAGE_SITE_SEMANTICS = {
    "c3_planner_memory": {
        "sites": 65,
        "channels": 256,
        "repairable_site_indices": list(range(64)),
        "frozen_site_indices": [64],
        "meaning": "8x8_bev_memory_tokens_plus_one_nonvisual_status_token",
    },
    "c4_trajectory_query": {
        "sites": 1,
        "channels": 256,
        "repairable_site_indices": [0],
        "frozen_site_indices": [],
        "meaning": "single_trajectory_query_token_after_transformer_decoder",
    },
}
EXPECTED_GROUPS = 42
EXPECTED_ANCHORS_PER_GROUP = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_fit_rows(
    group_roles_path: Path, time_set_path: Path
) -> list[dict[str, str]]:
    """Return exactly the registered 42x16 source-fit anchors."""

    with group_roles_path.open(newline="", encoding="utf-8") as stream:
        role_rows = list(csv.DictReader(stream))
    fit_groups = {
        row["recording_group"] for row in role_rows if row["role"] == "fit"
    }
    if len(fit_groups) != EXPECTED_GROUPS:
        raise RuntimeError(
            f"expected {EXPECTED_GROUPS} source-fit groups, found {len(fit_groups)}"
        )
    with time_set_path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["recording_group"] in fit_groups
        ]
    rows.sort(
        key=lambda row: (
            row["recording_group"],
            int(row["anchor_index"]),
            row["scene_token"],
        )
    )
    counts = Counter(row["recording_group"] for row in rows)
    if set(counts) != fit_groups or any(
        value != EXPECTED_ANCHORS_PER_GROUP for value in counts.values()
    ):
        raise RuntimeError(f"invalid source-fit anchor counts: {dict(counts)}")
    tokens = [row["scene_token"] for row in rows]
    if len(tokens) != len(set(tokens)):
        raise RuntimeError("source-fit scene tokens are not unique")
    return rows


def validate_group_shard(payload: dict[str, Any]) -> dict[str, list[int]]:
    """Validate one cache shard and return its frozen stage schema."""

    if payload.get("artifact") != "R006V0_SOURCE_FIT_ACTIVATION_SHARD_V1":
        raise ValueError("unexpected activation shard artifact")
    if payload.get("role") != "fit":
        raise ValueError("activation shard is not source-fit")
    anchors = payload.get("anchors", [])
    if len(anchors) != EXPECTED_ANCHORS_PER_GROUP:
        raise ValueError("activation shard must contain 16 anchors")
    group = payload.get("recording_group")
    indices = []
    tokens = set()
    schema: dict[str, list[int]] = {}
    for anchor in anchors:
        if anchor.get("recording_group") != group:
            raise ValueError("cross-group anchor found in shard")
        indices.append(int(anchor["anchor_index"]))
        token = anchor["scene_token"]
        if token in tokens:
            raise ValueError("duplicate token in shard")
        tokens.add(token)
        conditions = anchor.get("conditions", {})
        if set(conditions) != {"clean", *CONTROLLED_APPEARANCES}:
            raise ValueError("condition roster mismatch")
        for condition_payload in conditions.values():
            trajectory = condition_payload.get("trajectory")
            if not isinstance(trajectory, torch.Tensor):
                raise ValueError("trajectory is not a tensor")
            if not torch.isfinite(trajectory).all():
                raise ValueError("non-finite trajectory")
            activations = condition_payload.get("activations", {})
            if set(activations) != set(QUALIFIED_STAGES):
                raise ValueError("qualified stage roster mismatch")
            for stage, value in activations.items():
                if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                    raise ValueError(f"invalid activation at {stage}")
                shape = list(value.shape)
                expected = STAGE_SITE_SEMANTICS[stage]
                if shape != [1, expected["sites"], expected["channels"]]:
                    raise ValueError(f"registered site semantics disagree at {stage}")
                if stage in schema and schema[stage] != shape:
                    raise ValueError(f"activation schema changed at {stage}")
                schema.setdefault(stage, shape)
    if sorted(indices) != list(range(EXPECTED_ANCHORS_PER_GROUP)):
        raise ValueError("anchor indices are not exactly 0..15")
    return schema
