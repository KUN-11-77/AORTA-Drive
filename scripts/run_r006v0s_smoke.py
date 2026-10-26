#!/usr/bin/env python
"""Mechanical R006V0S smoke for frozen LTF activation interfaces.

This program reads only source-fit anchors. It does not estimate a scientific
effect, select a layer/rank, or access development, calibration, test, NavDream,
or target payloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


STAGES = (
    "c1_early_visual",
    "c2_fused_bev",
    "c3_planner_memory",
    "c4_trajectory_query",
)
ALLOWED_CONDITIONS = ("low_exposure", "low_contrast", "cool_cast")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_revision(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(path), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        return {
            "root": str(path),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_dir = path / ".git"
        try:
            head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                commit = (git_dir / head[5:]).read_text(encoding="utf-8").strip()
            else:
                commit = head
        except OSError:
            commit = None
        return {"root": str(path), "commit": commit, "dirty": None}


def bind_complete_early_visual_cut(prototype: Any) -> str:
    """Bind C1 to the first encoder return layer, before any image/BEV fusion.

    The legacy probe used the final image return layer.  In TransFuser that is
    not a causal cut because earlier fusion blocks have already copied visual
    information into the latent BEV stream.  Patching only that final image
    tensor therefore cannot restore the clean computation.  The first return
    layer lies before the first fusion and intercepts every downstream visual
    path while remaining a learned visual representation.
    """

    def first_image_module(model):
        encoder = model._backbone.image_encoder
        names = list(encoder.return_layers.keys())
        if not names:
            raise RuntimeError("image encoder has no return layers")
        modules = dict(encoder.named_modules())
        name = names[0]
        if name not in modules:
            raise RuntimeError(f"cannot resolve first image return layer {name}")
        return name, modules[name]

    prototype.final_image_module = first_image_module
    return "first_image_encoder_return_layer_before_any_cross_modal_fusion"


def select_fit_anchors(
    group_roles_path: Path, time_set_path: Path, count: int
) -> list[dict[str, str]]:
    with group_roles_path.open(newline="", encoding="utf-8") as stream:
        role_rows = list(csv.DictReader(stream))
    fit_groups = {
        row["recording_group"] for row in role_rows if row["role"] == "fit"
    }
    if len(fit_groups) != 42:
        raise RuntimeError(f"expected 42 fit groups, found {len(fit_groups)}")
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
    if len(rows) != 42 * 16:
        raise RuntimeError(f"expected 672 source-fit anchors, found {len(rows)}")
    # Use distinct groups so the smoke covers more than one log binding.
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row["recording_group"] in seen:
            continue
        selected.append(row)
        seen.add(row["recording_group"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError("not enough distinct fit groups for smoke")
    return selected


def apply_condition(image: torch.Tensor, condition: str) -> torch.Tensor:
    if condition not in ALLOWED_CONDITIONS:
        raise ValueError(f"unsupported condition {condition}")
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"expected [1,3,H,W] image, got {tuple(image.shape)}")
    if condition == "low_exposure":
        output = image * 0.50
    elif condition == "low_contrast":
        output = 0.50 + 0.55 * (image - 0.50)
    else:
        gains = image.new_tensor((0.85, 0.95, 1.10)).view(1, 3, 1, 1)
        output = image * gains
    return torch.clamp(output, 0.0, 1.0)


def trajectory_rms(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64)[..., :2] - np.asarray(
        right, dtype=np.float64
    )[..., :2]
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=-1))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrumentation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--planner-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sensors", type=Path, required=True)
    parser.add_argument("--group-roles", type=Path, required=True)
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=ALLOWED_CONDITIONS, default="low_exposure")
    parser.add_argument("--anchors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tail-rms-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    required = (
        args.instrumentation,
        args.checkpoint,
        args.planner_root,
        args.metadata,
        args.sensors,
        args.group_roles,
        args.time_set,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.anchors not in (1, 2):
        raise ValueError("R006V0S is restricted to one or two anchors")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R006V0S requires an explicitly selected CUDA device")

    prototype = load_module(args.instrumentation, "r006v0s_instrumentation")
    c1_cut_definition = bind_complete_early_visual_cut(prototype)
    imported_model_path = Path(inspect.getfile(prototype.TransfuserModel)).resolve()
    imported_builder_path = Path(
        inspect.getfile(prototype.TransfuserFeatureBuilder)
    ).resolve()
    expected_model_path = (
        args.planner_root / "navsim/agents/transfuser/transfuser_model.py"
    ).resolve()
    expected_builder_path = (
        args.planner_root / "navsim/agents/transfuser/transfuser_features.py"
    ).resolve()
    if imported_model_path != expected_model_path:
        raise RuntimeError(
            f"planner model source mismatch: imported {imported_model_path}, "
            f"expected {expected_model_path}"
        )
    if imported_builder_path != expected_builder_path:
        raise RuntimeError(
            f"feature builder source mismatch: imported {imported_builder_path}, "
            f"expected {expected_builder_path}"
        )
    helper = prototype.load_module(
        prototype.NAVDREAM_PATH, "r006v0s_sensor_config_helper"
    )
    model, config = prototype.build_frozen_model(
        args.checkpoint, device, latent=True
    )
    model.requires_grad_(False)
    for parameter in model.parameters():
        if parameter.requires_grad:
            raise RuntimeError("frozen planner contains trainable parameters")
    builder = prototype.TransfuserFeatureBuilder(config)
    selected = select_fit_anchors(args.group_roles, args.time_set, args.anchors)
    loader = prototype.SceneLoader(
        data_path=args.metadata,
        original_sensor_path=args.sensors,
        scene_filter=prototype.SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            frame_interval=1,
            has_route=True,
            log_names=sorted({row["log_name"] for row in selected}),
            tokens=sorted({row["scene_token"] for row in selected}),
        ),
        sensor_config=helper.sensor_config(),
    )

    anchor_reports: list[dict[str, Any]] = []
    schema: dict[str, dict[str, Any]] = {}
    all_pass = True
    for row in selected:
        token = row["scene_token"]
        agent_input = loader.get_agent_input_from_token(token)
        clean_features = prototype.make_features(builder, agent_input, config, device)
        shifted_features = {
            key: value.clone() for key, value in clean_features.items()
        }
        shifted_features["image"] = apply_condition(
            shifted_features["image"], args.condition
        )
        prototype.assert_non_image_equal(clean_features, shifted_features)

        clean_trajectory, clean_activations = prototype.run_forward(
            model, clean_features
        )
        clean_replay, replay_activations = prototype.run_forward(
            model, clean_features
        )
        shifted_trajectory, shifted_activations = prototype.run_forward(
            model, shifted_features
        )
        clean_replay_exact = bool(np.array_equal(clean_trajectory, clean_replay))
        activation_replay_exact = all(
            torch.equal(clean_activations[stage], replay_activations[stage])
            for stage in STAGES
        )
        if not clean_replay_exact or not activation_replay_exact:
            all_pass = False

        stage_reports: list[dict[str, Any]] = []
        for stage in STAGES:
            clean_value = clean_activations[stage]
            shifted_value = shifted_activations[stage]
            schema.setdefault(
                stage,
                {
                    "shape": list(clean_value.shape),
                    "dtype": str(clean_value.dtype),
                    "spatial_channel_convention": (
                        "BCHW_channels_shared" if clean_value.ndim == 4 else "BPC_channels_last"
                    ),
                },
            )
            if list(clean_value.shape) != schema[stage]["shape"]:
                raise RuntimeError(f"activation schema changed at {stage}")

            identity_trajectory, _ = prototype.run_forward(
                model,
                shifted_features,
                patch_stage=stage,
                donor=shifted_activations,
            )
            restored_trajectory, _ = prototype.run_forward(
                model,
                shifted_features,
                patch_stage=stage,
                donor=clean_activations,
            )
            reverse_trajectory, _ = prototype.run_forward(
                model,
                clean_features,
                patch_stage=stage,
                donor=shifted_activations,
            )
            identity_exact = bool(
                np.array_equal(identity_trajectory, shifted_trajectory)
            )
            restoration_rms = trajectory_rms(
                restored_trajectory, clean_trajectory
            )
            reverse_damage_rms = trajectory_rms(
                reverse_trajectory, clean_trajectory
            )
            finite = bool(
                np.isfinite(restored_trajectory).all()
                and np.isfinite(reverse_trajectory).all()
                and torch.isfinite(clean_value).all()
                and torch.isfinite(shifted_value).all()
            )
            stage_pass = (
                identity_exact
                and finite
                and restoration_rms <= args.tail_rms_tolerance
            )
            all_pass = all_pass and stage_pass
            stage_reports.append(
                {
                    "stage": stage,
                    "identity_exact": identity_exact,
                    "full_patch_restoration_rms_m": restoration_rms,
                    "reverse_damage_rms_m": reverse_damage_rms,
                    "finite": finite,
                    "pass": stage_pass,
                }
            )
        anchor_reports.append(
            {
                "recording_group": row["recording_group"],
                "token": token,
                "log_name": row["log_name"],
                "role": "fit",
                "clean_replay_exact": clean_replay_exact,
                "activation_replay_exact": activation_replay_exact,
                "clean_shift_trajectory_rms_m": trajectory_rms(
                    shifted_trajectory, clean_trajectory
                ),
                "stages": stage_reports,
            }
        )

    schema_payload = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    preprocess_path = args.planner_root / "navsim/agents/transfuser/transfuser_features.py"
    report = {
        "artifact": "R006V0S_FROZEN_LTF_INTERFACE_SMOKE_V1",
        "status": "PASS" if all_pass else "FAIL",
        "scientific_metrics_emitted": False,
        "development_calibration_test_navdream_target_accessed": False,
        "appearance_condition": args.condition,
        "appearance_application_point": "after_official_stitch_resize_before_frozen_backbone",
        "c1_cut_definition": c1_cut_definition,
        "anchors": len(selected),
        "tail_rms_tolerance": args.tail_rms_tolerance,
        "device": str(device),
        "seed": args.seed,
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "instrumentation": str(args.instrumentation),
            "instrumentation_sha256": sha256_file(args.instrumentation),
            "runner_sha256": sha256_file(Path(__file__)),
            "group_roles_sha256": sha256_file(args.group_roles),
            "time_set_sha256": sha256_file(args.time_set),
            "preprocess_sha256": sha256_file(preprocess_path),
            "imported_model_path": str(imported_model_path),
            "imported_model_sha256": sha256_file(imported_model_path),
            "imported_builder_path": str(imported_builder_path),
            "imported_builder_sha256": sha256_file(imported_builder_path),
            "planner_source": git_revision(args.planner_root),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "hook_schema": schema,
        "hook_schema_sha256": hashlib.sha256(schema_payload.encode()).hexdigest(),
        "anchor_reports": anchor_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
