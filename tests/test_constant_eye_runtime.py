from __future__ import annotations

from typing import cast

import torch

from aorta_drive.vision.constant_eye_runtime import (
    CONSTANT_EYE_DINO_ONLY_STAGES,
    ConstantEyeTransfuserRuntime,
)


class _ToyDecoder(torch.nn.Module):
    def forward(self, query: torch.Tensor, keyval: torch.Tensor) -> torch.Tensor:
        return query + keyval.mean(dim=1, keepdim=True)


class _ToyHead(torch.nn.Module):
    def forward(self, query: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"trajectory": query[..., :2]}


class _ToyConstantEye(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._dino_proj: torch.nn.Linear = torch.nn.Linear(3, 4, bias=False)
        self._tf_decoder: _ToyDecoder = _ToyDecoder()
        self._trajectory_head: _ToyHead = _ToyHead()
        with torch.no_grad():
            self._dino_proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                )
            )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        projected = self._dino_proj(features["dino_feature"])
        query = torch.ones(projected.shape[0], 1, 4)
        query = self._tf_decoder(query, projected)
        # Call the module rather than ``forward`` directly so PyTorch hooks run.
        # The cast only narrows ``nn.Module.__call__`` for strict static typing.
        return cast(dict[str, torch.Tensor], self._trajectory_head(query))


def _features(value: float) -> dict[str, torch.Tensor]:
    return {"dino_feature": torch.full((1, 5, 3), value)}


def test_runtime_captures_audited_dino_only_order() -> None:
    runtime = ConstantEyeTransfuserRuntime(_ToyConstantEye().eval())
    output, captures = runtime.forward(_features(2.0))
    assert output.shape == (1, 1, 2)
    assert tuple(captures) == CONSTANT_EYE_DINO_ONLY_STAGES
    assert captures["raw_dino_tokens"].shape == (1, 5, 3)
    assert captures["projected_dino_tokens"].shape == (1, 5, 4)
    assert captures["decoder_keyval"].shape == (1, 5, 4)
    assert captures["trajectory_query"].shape == (1, 1, 4)


def test_identity_patch_is_exact_at_every_complete_cut() -> None:
    runtime = ConstantEyeTransfuserRuntime(_ToyConstantEye().eval())
    shifted_output, shifted = runtime.forward(_features(2.0))
    for stage in CONSTANT_EYE_DINO_ONLY_STAGES:
        identity, _ = runtime.forward(
            _features(2.0), patch_stage=stage, patch=shifted[stage]
        )
        torch.testing.assert_close(identity, shifted_output, rtol=0.0, atol=0.0)


def test_clean_patch_at_every_complete_cut_recovers_clean_output() -> None:
    runtime = ConstantEyeTransfuserRuntime(_ToyConstantEye().eval())
    clean_output, clean = runtime.forward(_features(1.0))
    shifted_output, _ = runtime.forward(_features(2.0))
    assert not torch.equal(clean_output, shifted_output)
    for stage in CONSTANT_EYE_DINO_ONLY_STAGES:
        patched, _ = runtime.forward(
            _features(2.0), patch_stage=stage, patch=clean[stage]
        )
        torch.testing.assert_close(patched, clean_output, rtol=0.0, atol=0.0)


def test_runtime_rejects_wrong_shape_and_missing_topology() -> None:
    runtime = ConstantEyeTransfuserRuntime(_ToyConstantEye().eval())
    try:
        runtime.forward(
            _features(2.0),
            patch_stage="raw_dino_tokens",
            patch=torch.zeros(1, 1, 3),
        )
    except ValueError as error:
        assert "shape mismatch" in str(error)
    else:
        raise AssertionError("wrong-shape patch was accepted")

    class _Missing(torch.nn.Module):
        pass

    try:
        ConstantEyeTransfuserRuntime(_Missing())
    except TypeError as error:
        assert "missing audited module" in str(error)
    else:
        raise AssertionError("missing topology was accepted")

