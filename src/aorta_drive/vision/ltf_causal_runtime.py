"""Project-owned causal intervention runtime for the frozen NAVSIM LTF planner.

NAVSIM is imported lazily so this module remains unit-testable without the
server checkout.  All visual interventions are explicit tensor replacements;
the frozen planner parameters are never modified.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


STAGES = (
    "c1_early_visual",
    "c2_fused_bev",
    "c3_planner_memory",
    "c4_trajectory_query",
)


@dataclass(frozen=True)
class LTFBindings:
    planner_root: Path
    TransfuserConfig: type
    TransfuserFeatureBuilder: type
    TransfuserModel: type
    SceneFilter: type
    SensorConfig: type
    SceneLoader: type
    TrajectorySampling: type


def load_ltf_bindings(planner_root: Path) -> LTFBindings:
    """Import NAVSIM from exactly ``planner_root`` and reject source aliasing."""

    root = planner_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    sys.path.insert(0, str(root))
    config_module = importlib.import_module(
        "navsim.agents.transfuser.transfuser_config"
    )
    feature_module = importlib.import_module(
        "navsim.agents.transfuser.transfuser_features"
    )
    model_module = importlib.import_module("navsim.agents.transfuser.transfuser_model")
    dataclass_module = importlib.import_module("navsim.common.dataclasses")
    loader_module = importlib.import_module("navsim.common.dataloader")
    trajectory_module = importlib.import_module(
        "nuplan.planning.simulation.trajectory.trajectory_sampling"
    )
    imported_model = Path(model_module.__file__).resolve()
    expected_model = root / "navsim/agents/transfuser/transfuser_model.py"
    if imported_model != expected_model:
        raise RuntimeError(
            f"NAVSIM source aliasing: imported {imported_model}, expected {expected_model}"
        )
    return LTFBindings(
        planner_root=root,
        TransfuserConfig=config_module.TransfuserConfig,
        TransfuserFeatureBuilder=feature_module.TransfuserFeatureBuilder,
        TransfuserModel=model_module.TransfuserModel,
        SceneFilter=dataclass_module.SceneFilter,
        SensorConfig=dataclass_module.SensorConfig,
        SceneLoader=loader_module.SceneLoader,
        TrajectorySampling=trajectory_module.TrajectorySampling,
    )


def controlled_sensor_config(bindings: LTFBindings):
    current = [3]
    return bindings.SensorConfig(
        cam_f0=current,
        cam_l0=current,
        cam_l1=False,
        cam_l2=False,
        cam_r0=current,
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=current,
    )


def build_frozen_ltf(
    bindings: LTFBindings, checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, Any]:
    """Build the released latent TransFuser and strictly restore its weights."""

    import timm

    config = bindings.TransfuserConfig(bkb_path="", latent=True, normalization=False)
    original_create_model = timm.create_model

    def architecture_only(*args, **kwargs):
        kwargs["pretrained"] = False
        kwargs.pop("pretrained_cfg_overlay", None)
        return original_create_model(*args, **kwargs)

    timm.create_model = architecture_only
    try:
        model = bindings.TransfuserModel(
            bindings.TrajectorySampling(time_horizon=4.0, interval_length=0.5),
            config,
        )
    finally:
        timm.create_model = original_create_model
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    prefix = "agent._transfuser_model."
    state = {
        key[len(prefix) :]: value
        for key, value in payload["state_dict"].items()
        if key.startswith(prefix)
    }
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    wrong_shape = sorted(
        key
        for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or wrong_shape:
        raise RuntimeError(
            {"missing": missing[:10], "unexpected": unexpected[:10], "shape": wrong_shape[:10]}
        )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval(), config


def compatible_features(builder: Any, agent_input: Any, config: Any) -> dict[str, torch.Tensor]:
    status = torch.concatenate(
        [
            torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
            torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
        ]
    )
    result = {"image": builder._get_camera_feature(agent_input), "status": status}
    if not config.latent:
        raise NotImplementedError("R006 currently qualifies only the released latent LTF")
    return result


def make_features(
    builder: Any, agent_input: Any, config: Any, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in compatible_features(builder, agent_input, config).items()
    }


def activation_to_npc(value: torch.Tensor) -> torch.Tensor:
    """Convert one activation to the registered [N=1, spatial, channel] form."""

    if value.shape[0] != 1:
        raise ValueError("R006 intervention runtime expects one anchor per forward")
    if value.ndim == 4:
        return value.permute(0, 2, 3, 1).reshape(1, -1, value.shape[1])
    if value.ndim == 3:
        return value
    raise ValueError(f"unsupported activation shape {tuple(value.shape)}")


def npc_to_activation(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if reference.ndim == 4:
        _, channels, height, width = reference.shape
        if value.shape != (1, height * width, channels):
            raise ValueError("NPC tensor does not match BCHW reference")
        return value.reshape(1, height, width, channels).permute(0, 3, 1, 2)
    if reference.ndim == 3 and value.shape == reference.shape:
        return value
    raise ValueError("NPC tensor does not match reference")


def low_rank_coefficient_patch(
    clean: torch.Tensor,
    shifted: torch.Tensor,
    basis: torch.Tensor,
    dose: float = 1.0,
) -> torch.Tensor:
    """Patch only clean-minus-shifted coefficients in a shared channel basis."""

    clean_npc = activation_to_npc(clean)
    shifted_npc = activation_to_npc(shifted)
    basis = basis.to(device=shifted.device, dtype=shifted.dtype)
    if basis.ndim != 2 or basis.shape[0] != shifted_npc.shape[-1]:
        raise ValueError("basis shape disagrees with activation channels")
    delta = clean_npc - shifted_npc
    correction = (delta @ basis) @ basis.transpose(0, 1)
    return npc_to_activation(shifted_npc + float(dose) * correction, shifted)


class LTFCausalRuntime:
    """Capture and replace the four preregistered frozen-LTF interfaces."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def _stage_module(self, stage: str) -> tuple[torch.nn.Module, bool]:
        if stage == "c1_early_visual":
            encoder = self.model._backbone.image_encoder
            names = list(encoder.return_layers.keys())
            modules = dict(encoder.named_modules())
            return modules[names[0]], False
        if stage == "c2_fused_bev":
            return self.model._backbone, False
        if stage == "c3_planner_memory":
            return self.model._tf_decoder, True
        if stage == "c4_trajectory_query":
            return self.model._trajectory_head, True
        raise ValueError(stage)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        *,
        patch_stage: str | None = None,
        patch: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        capture: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        captures: dict[str, torch.Tensor] = {}
        handles = []

        def replacement(stage: str, current: torch.Tensor) -> torch.Tensor:
            if capture:
                captures[stage] = current.detach().float().cpu().clone()
            if patch_stage != stage:
                return current
            if patch is None:
                raise ValueError("patch_stage requires a patch")
            value = patch(current) if callable(patch) else patch
            if value.shape != current.shape:
                raise ValueError(f"patch shape mismatch at {stage}")
            return value.to(device=current.device, dtype=current.dtype)

        for stage in STAGES:
            module, is_pre = self._stage_module(stage)
            if stage == "c2_fused_bev":
                def c2_hook(_module, _inputs, output, stage=stage):
                    values = list(output)
                    values[1] = replacement(stage, values[1])
                    return tuple(values)

                handles.append(module.register_forward_hook(c2_hook))
            elif is_pre:
                index = 1 if stage == "c3_planner_memory" else 0

                def pre_hook(_module, inputs, stage=stage, index=index):
                    values = list(inputs)
                    values[index] = replacement(stage, values[index])
                    return tuple(values)

                handles.append(module.register_forward_pre_hook(pre_hook))
            else:
                def post_hook(_module, _inputs, output, stage=stage):
                    return replacement(stage, output)

                handles.append(module.register_forward_hook(post_hook))
        try:
            output = self.model(features)["trajectory"]
        finally:
            for handle in handles:
                handle.remove()
        if capture and set(captures) != set(STAGES):
            raise RuntimeError(f"missing captures: {set(STAGES) - set(captures)}")
        return output, captures

    def trajectory_xy_from_stage(
        self,
        features: dict[str, torch.Tensor],
        stage: str,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        trajectory, _ = self.forward(
            features, patch_stage=stage, patch=activation, capture=False
        )
        return trajectory[..., :2].reshape(trajectory.shape[0], -1)


def trajectory_rms(left: torch.Tensor | np.ndarray, right: torch.Tensor | np.ndarray) -> float:
    left_value = torch.as_tensor(left, dtype=torch.float64)[..., :2]
    right_value = torch.as_tensor(right, dtype=torch.float64)[..., :2]
    difference = left_value - right_value
    return float(torch.sqrt(torch.mean(torch.sum(difference * difference, dim=-1))))
