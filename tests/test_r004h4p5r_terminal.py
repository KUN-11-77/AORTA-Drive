from __future__ import annotations

import pytest

from run_r004h4p5r_oriented_terminal import _phase_grid


@pytest.mark.parametrize("resolution", [1.0, 0.5, 0.25, 0.125])
def test_canonical_geometry_grid_preserves_frozen_support(resolution: float) -> None:
    grid = _phase_grid(resolution, 0.0, 0.0)
    assert grid.x_support == (-32.0, 32.0)
    assert grid.y_support == (-32.0, 32.0)


def test_half_phase_partition_covers_but_does_not_redefine_support() -> None:
    grid = _phase_grid(0.25, 0.5, 0.5)
    assert grid.x_min == -32.125
    assert grid.y_min == -32.125
    assert grid.x_max == 32.125
    assert grid.y_max == 32.125

