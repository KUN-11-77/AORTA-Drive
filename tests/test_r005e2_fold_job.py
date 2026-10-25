from __future__ import annotations

import numpy as np
import torch

from run_r005e2_fold_job import _group_equal_mse, _schedule


def test_group_equal_mse_preserves_zero_challenger_anchor() -> None:
    prediction = np.zeros((32, 7), dtype=np.float32)
    target = np.zeros((32, 7), dtype=np.float32)
    mask = np.zeros((32, 7), dtype=np.bool_)
    mask[:, 0] = True
    mask[0, 1] = True
    target[0, 1] = 2.0
    indices = np.arange(32, dtype=np.int64)
    groups = {index: ("g0" if index < 16 else "g1") for index in range(32)}
    # The single challenger error is averaged over all 16 anchors in g0, then
    # g0/g1 are equal-weighted: 4 / 16 / 2.
    assert _group_equal_mse(prediction, target, mask, indices, groups) == 0.125


def test_schedule_warms_then_reaches_zero() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=3.0e-4)
    scheduler = _schedule(optimizer, total_steps=20)
    values: list[float] = []
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        values.append(float(optimizer.param_groups[0]["lr"]))
    assert all(np.isfinite(values))
    assert values[-1] == 0.0
    assert max(values) <= 3.0e-4
