from __future__ import annotations

import torch

from aorta_drive.vision.ltf_causal_runtime import (
    activation_to_npc,
    low_rank_coefficient_patch,
    npc_to_activation,
)


def test_activation_npc_roundtrip_for_bchw_and_bpc() -> None:
    bchw = torch.arange(1 * 3 * 2 * 4, dtype=torch.float32).reshape(1, 3, 2, 4)
    bpc = torch.arange(1 * 5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    torch.testing.assert_close(npc_to_activation(activation_to_npc(bchw), bchw), bchw)
    torch.testing.assert_close(npc_to_activation(activation_to_npc(bpc), bpc), bpc)


def test_low_rank_patch_changes_only_registered_channel_subspace() -> None:
    shifted = torch.zeros(1, 3, 2, dtype=torch.float64)
    clean = shifted.clone()
    clean[..., 0] = 2.0
    clean[..., 1] = 7.0
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    patched = low_rank_coefficient_patch(clean, shifted, basis, dose=0.25)
    torch.testing.assert_close(
        patched[..., 0], torch.full((1, 3), 0.5, dtype=torch.float64)
    )
    torch.testing.assert_close(patched[..., 1], torch.zeros(1, 3, dtype=torch.float64))


def test_low_rank_patch_never_changes_frozen_site() -> None:
    shifted = torch.zeros(1, 3, 2, dtype=torch.float64)
    clean = torch.ones(1, 3, 2, dtype=torch.float64)
    basis = torch.eye(2, dtype=torch.float64)
    site_mask = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    patched = low_rank_coefficient_patch(
        clean,
        shifted,
        basis,
        site_mask=site_mask,
    )
    torch.testing.assert_close(patched[:, :2], clean[:, :2])
    torch.testing.assert_close(patched[:, 2], shifted[:, 2])
