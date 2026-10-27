"""Topology-faithful intervention runtime for Constant Eye TransFuser-DINO.

The runtime intentionally depends only on the public module attributes observed
in the frozen source audit.  It does not import the external NAVSIM checkout and
therefore remains unit-testable.  A real checkpoint is still required before
any scientific use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, cast

import torch


CONSTANT_EYE_DINO_ONLY_STAGES = (
    "raw_dino_tokens",
    "projected_dino_tokens",
    "decoder_keyval",
    "trajectory_query",
)


@dataclass(frozen=True)
class StageBinding:
    """A complete-cut hook location in the audited DINO-only topology."""

    module: torch.nn.Module
    mode: str
    input_index: int | None = None


class ConstantEyeTransfuserRuntime:
    """Capture or replace complete cuts in a TransFuser-DINO forward pass."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        for name in ("_dino_proj", "_tf_decoder", "_trajectory_head"):
            if not isinstance(getattr(model, name, None), torch.nn.Module):
                raise TypeError(f"model is missing audited module {name}")
        self._dino_proj = cast(torch.nn.Module, getattr(model, "_dino_proj"))
        self._tf_decoder = cast(torch.nn.Module, getattr(model, "_tf_decoder"))
        self._trajectory_head = cast(torch.nn.Module, getattr(model, "_trajectory_head"))

    def _binding(self, stage: str) -> StageBinding:
        if stage == "raw_dino_tokens":
            return StageBinding(self._dino_proj, "pre", 0)
        if stage == "projected_dino_tokens":
            return StageBinding(self._dino_proj, "post")
        if stage == "decoder_keyval":
            return StageBinding(self._tf_decoder, "pre", 1)
        if stage == "trajectory_query":
            return StageBinding(self._trajectory_head, "pre", 0)
        raise ValueError(f"unsupported Constant Eye stage {stage}")

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        *,
        patch_stage: str | None = None,
        patch: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        capture: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one deterministic forward with at most one complete-cut patch."""

        if patch_stage is not None and patch_stage not in CONSTANT_EYE_DINO_ONLY_STAGES:
            raise ValueError(f"unsupported patch stage {patch_stage}")
        if patch_stage is not None and patch is None:
            raise ValueError("patch_stage requires a patch")
        captures: dict[str, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def replace(stage: str, current: torch.Tensor) -> torch.Tensor:
            if capture:
                captures[stage] = current.detach().float().cpu().clone()
            if stage != patch_stage:
                return current
            assert patch is not None
            value = patch(current) if callable(patch) else patch
            if value.shape != current.shape:
                raise ValueError(f"patch shape mismatch at {stage}")
            return value.to(device=current.device, dtype=current.dtype)

        for stage in CONSTANT_EYE_DINO_ONLY_STAGES:
            binding = self._binding(stage)
            if binding.mode == "post":

                def post_hook(
                    _module: torch.nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    stage: str = stage,
                ) -> torch.Tensor:
                    if not isinstance(output, torch.Tensor):
                        raise TypeError(f"expected tensor output at {stage}")
                    return replace(stage, output)

                handles.append(binding.module.register_forward_hook(post_hook))
            else:
                assert binding.input_index is not None
                index = binding.input_index

                def pre_hook(
                    _module: torch.nn.Module,
                    inputs: tuple[Any, ...],
                    stage: str = stage,
                    index: int = index,
                ) -> tuple[Any, ...]:
                    values = list(inputs)
                    if index >= len(values) or not isinstance(values[index], torch.Tensor):
                        raise TypeError(f"expected tensor input {index} at {stage}")
                    values[index] = replace(stage, values[index])
                    return tuple(values)

                handles.append(binding.module.register_forward_pre_hook(pre_hook))
        try:
            result = self.model(dict(features))
            if not isinstance(result, Mapping) or not isinstance(
                result.get("trajectory"), torch.Tensor
            ):
                raise TypeError("Constant Eye model must return a tensor trajectory")
            trajectory = cast(torch.Tensor, result["trajectory"])
        finally:
            for handle in handles:
                handle.remove()
        if capture and set(captures) != set(CONSTANT_EYE_DINO_ONLY_STAGES):
            missing = sorted(set(CONSTANT_EYE_DINO_ONLY_STAGES) - set(captures))
            raise RuntimeError(f"missing Constant Eye captures: {missing}")
        return trajectory, captures

