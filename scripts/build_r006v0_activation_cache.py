#!/usr/bin/env python
"""Build the sealed source-fit activation cache for formal R006V0.

The program emits no mediation, restoration or effect metrics.  It only runs
the frozen planner on clean and preregistered controlled appearances, stores
the two mechanically qualified causal interfaces, and writes a hash manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aorta_drive.vision.ltf_causal_runtime import (
    CONTROLLED_APPEARANCES,
    LTFCausalRuntime,
    apply_controlled_appearance,
    build_frozen_ltf,
    controlled_sensor_config,
    load_ltf_bindings,
    make_features,
)
from aorta_drive.vision.r006_cache import (
    QUALIFIED_STAGES,
    load_source_fit_rows,
    sha256_file,
    validate_group_shard,
)


def git_revision(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(path), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    return {
        "root": str(path.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--planner-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sensors", type=Path, required=True)
    parser.add_argument("--group-roles", type=Path, required=True)
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--limit-groups", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("an explicitly selected CUDA device is required")

    rows = load_source_fit_rows(args.group_roles, args.time_set)
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[row["recording_group"]].append(row)
    group_names = sorted(by_group)
    if args.limit_groups is not None:
        if args.limit_groups <= 0 or args.limit_groups > len(group_names):
            raise ValueError("limit-groups must lie in [1,42]")
        group_names = group_names[: args.limit_groups]
    selected = [row for group in group_names for row in by_group[group]]

    bindings = load_ltf_bindings(args.planner_root)
    model, config = build_frozen_ltf(bindings, args.checkpoint, device)
    runtime = LTFCausalRuntime(model)
    builder = bindings.TransfuserFeatureBuilder(config)
    loader = bindings.SceneLoader(
        data_path=args.metadata,
        original_sensor_path=args.sensors,
        scene_filter=bindings.SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            frame_interval=1,
            has_route=True,
            log_names=sorted({row["log_name"] for row in selected}),
            tokens=sorted({row["scene_token"] for row in selected}),
        ),
        sensor_config=controlled_sensor_config(bindings),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_records = []
    global_schema: dict[str, list[int]] = {}
    for group_index, group in enumerate(group_names, start=1):
        anchor_payloads = []
        for row in by_group[group]:
            agent_input = loader.get_agent_input_from_token(row["scene_token"])
            clean_features = make_features(builder, agent_input, config, device)
            condition_payloads = {}
            for condition in ("clean", *CONTROLLED_APPEARANCES):
                features = {key: value.clone() for key, value in clean_features.items()}
                if condition != "clean":
                    features["image"] = apply_controlled_appearance(
                        features["image"], condition
                    )
                with torch.no_grad():
                    trajectory, captures = runtime.forward(features)
                condition_payloads[condition] = {
                    "trajectory": trajectory.detach().float().cpu().contiguous(),
                    "activations": {
                        stage: captures[stage].detach().float().cpu().contiguous()
                        for stage in QUALIFIED_STAGES
                    },
                }
            anchor_payloads.append(
                {
                    "recording_group": group,
                    "anchor_index": int(row["anchor_index"]),
                    "scene_token": row["scene_token"],
                    "log_name": row["log_name"],
                    "conditions": condition_payloads,
                }
            )
        shard = {
            "artifact": "R006V0_SOURCE_FIT_ACTIVATION_SHARD_V1",
            "role": "fit",
            "recording_group": group,
            "anchors": anchor_payloads,
        }
        schema = validate_group_shard(shard)
        if global_schema and schema != global_schema:
            raise RuntimeError("cross-group activation schema changed")
        global_schema = schema
        shard_path = args.output_dir / f"group_{group_index:02d}.pt"
        torch.save(shard, shard_path)
        shard_records.append(
            {
                "recording_group": group,
                "file": shard_path.name,
                "sha256": sha256_file(shard_path),
                "anchors": len(anchor_payloads),
            }
        )
        print(
            json.dumps(
                {
                    "status": "RUNNING",
                    "completed_groups": group_index,
                    "total_groups": len(group_names),
                }
            ),
            flush=True,
        )

    manifest = {
        "artifact": "R006V0_SOURCE_FIT_ACTIVATION_CACHE_MANIFEST_V1",
        "status": "PASS",
        "scientific_metrics_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
        "complete_formal_cache": args.limit_groups is None,
        "groups": len(group_names),
        "anchors": len(selected),
        "conditions": ["clean", *CONTROLLED_APPEARANCES],
        "qualified_stages": list(QUALIFIED_STAGES),
        "schema": global_schema,
        "shards": shard_records,
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "planner_source": git_revision(args.planner_root),
            "group_roles_sha256": sha256_file(args.group_roles),
            "time_set_sha256": sha256_file(args.time_set),
            "runner_sha256": sha256_file(Path(__file__)),
            "runtime_sha256": sha256_file(
                Path(__file__).parents[1]
                / "src/aorta_drive/vision/ltf_causal_runtime.py"
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "seed": args.seed,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
