"""Mechanical contracts for the topology-conditioned R007 DINO audit.

This module contains no scientific threshold and never interprets an effect as
beneficial.  It validates ordered DINO-only stage exchanges and, only for true
hybrid systems, a native-path x DINO-path factorial exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt


CELL_NAMES = ("shift_shift", "shift_clean", "clean_shift", "clean_clean")
Array = npt.NDArray[Any]
Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FactorialCell:
    """One output and the immutable identities of its two visual paths."""

    output: Array
    native_fingerprint: str
    dino_fingerprint: str


@dataclass(frozen=True)
class FactorialEffects:
    """Exact vector-valued effects for a 2x2 finite intervention."""

    dino_repair_at_shifted_native: Float64Array
    native_repair_at_shifted_dino: Float64Array
    joint_repair: Float64Array
    interaction: Float64Array
    closure_residual: Float64Array


@dataclass(frozen=True)
class FactorialAudit:
    """Mechanical audit result without a scientific mechanism label."""

    output_shape: tuple[int, ...]
    finite: bool
    path_isolation: bool
    closure_exact: bool
    max_abs_closure_residual: float


@dataclass(frozen=True)
class SequentialCut:
    """One topology-ordered DINO-only interchange result."""

    stage: str
    clean_fingerprint: str
    shifted_fingerprint: str
    shifted_output: Array
    identity_output: Array
    clean_patch_output: Array


@dataclass(frozen=True)
class SequentialAudit:
    """Mechanical result for a DINO-only ordered interface."""

    stages: tuple[str, ...]
    output_shape: tuple[int, ...]
    identity_exact: bool
    full_cut_exact: bool
    finite: bool


def _as_float64(value: Array) -> Float64Array:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError("factorial outputs must be non-empty")
    return array


def validate_factorial_cells(cells: Mapping[str, FactorialCell]) -> tuple[int, ...]:
    """Validate complete cells, shape/finite values, and path isolation.

    Fingerprints are content identities computed by the real runner.  Equal
    fingerprints certify that the path declared fixed was not silently rebuilt
    from a different image or cache entry.
    """

    if set(cells) != set(CELL_NAMES):
        missing = sorted(set(CELL_NAMES) - set(cells))
        extra = sorted(set(cells) - set(CELL_NAMES))
        raise ValueError(f"factorial cells disagree: missing={missing}, extra={extra}")

    arrays = {name: _as_float64(cells[name].output) for name in CELL_NAMES}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"factorial output shapes disagree: {sorted(shapes)}")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("factorial outputs must all be finite")

    ss = cells["shift_shift"]
    sc = cells["shift_clean"]
    cs = cells["clean_shift"]
    cc = cells["clean_clean"]
    if not (
        ss.native_fingerprint == sc.native_fingerprint
        and cs.native_fingerprint == cc.native_fingerprint
        and ss.dino_fingerprint == cs.dino_fingerprint
        and sc.dino_fingerprint == cc.dino_fingerprint
    ):
        raise ValueError("factorial path isolation fingerprints disagree")
    if ss.native_fingerprint == cs.native_fingerprint:
        raise ValueError("clean and shifted native paths unexpectedly alias")
    if ss.dino_fingerprint == sc.dino_fingerprint:
        raise ValueError("clean and shifted DINO paths unexpectedly alias")
    return next(iter(shapes))


def factorial_effects(cells: Mapping[str, FactorialCell]) -> FactorialEffects:
    """Compute the exact non-additive 2x2 finite-difference decomposition."""

    validate_factorial_cells(cells)
    ss = _as_float64(cells["shift_shift"].output)
    sc = _as_float64(cells["shift_clean"].output)
    cs = _as_float64(cells["clean_shift"].output)
    cc = _as_float64(cells["clean_clean"].output)

    dino = sc - ss
    native = cs - ss
    joint = cc - ss
    interaction = cc - sc - cs + ss
    closure = joint - (dino + native + interaction)
    return FactorialEffects(
        dino_repair_at_shifted_native=dino,
        native_repair_at_shifted_dino=native,
        joint_repair=joint,
        interaction=interaction,
        closure_residual=closure,
    )


def audit_factorial_cells(
    cells: Mapping[str, FactorialCell], *, atol: float = 0.0
) -> FactorialAudit:
    """Return fail-closed mechanical checks for one factorial payload."""

    if atol < 0:
        raise ValueError("atol must be non-negative")
    shape = validate_factorial_cells(cells)
    effects = factorial_effects(cells)
    maximum = float(np.max(np.abs(effects.closure_residual)))
    return FactorialAudit(
        output_shape=shape,
        finite=True,
        path_isolation=True,
        closure_exact=maximum <= atol,
        max_abs_closure_residual=maximum,
    )


def validate_derangement(mapping: Mapping[str, str]) -> None:
    """Reject incomplete, non-bijective, or fixed-point mismatch donors."""

    keys = set(mapping)
    values = set(mapping.values())
    if not keys:
        raise ValueError("donor mapping must be non-empty")
    if keys != values:
        raise ValueError("donor mapping must be a permutation of the groups")
    fixed = sorted(group for group, donor in mapping.items() if group == donor)
    if fixed:
        raise ValueError(f"donor mapping contains fixed points: {fixed}")


def require_exact_identity(reference: Array, identity: Array) -> None:
    """Require byte-level numerical identity for an identity intervention."""

    left = np.asarray(reference)
    right = np.asarray(identity)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("identity intervention shape or dtype changed")
    if not np.array_equal(left, right, equal_nan=False):
        raise ValueError("identity intervention is not exact")


def audit_sequential_cuts(
    cuts: Mapping[str, SequentialCut],
    *,
    expected_order: tuple[str, ...],
    clean_output: Array,
) -> SequentialAudit:
    """Validate a DINO-only chain without inventing a native visual path.

    ``expected_order[0]`` must be the complete raw-token cut.  Replacing that
    cut with its clean value must exactly reproduce ``clean_output``.  Later
    cuts are mechanically qualified but receive no scientific interpretation.
    """

    if not expected_order or len(expected_order) != len(set(expected_order)):
        raise ValueError("expected_order must contain unique stages")
    if set(cuts) != set(expected_order):
        missing = sorted(set(expected_order) - set(cuts))
        extra = sorted(set(cuts) - set(expected_order))
        raise ValueError(f"sequential cuts disagree: missing={missing}, extra={extra}")
    if tuple(cuts) != expected_order:
        raise ValueError("sequential cut insertion order disagrees with topology")

    clean = _as_float64(clean_output)
    shape = clean.shape
    for name in expected_order:
        cut = cuts[name]
        if cut.stage != name:
            raise ValueError(f"sequential cut stage label disagrees at {name}")
        if cut.clean_fingerprint == cut.shifted_fingerprint:
            raise ValueError(f"clean and shifted stage values alias at {name}")
        arrays = (
            _as_float64(cut.shifted_output),
            _as_float64(cut.identity_output),
            _as_float64(cut.clean_patch_output),
        )
        if any(array.shape != shape for array in arrays):
            raise ValueError(f"sequential output shape changed at {name}")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError(f"sequential output is non-finite at {name}")
        require_exact_identity(cut.shifted_output, cut.identity_output)

    first = cuts[expected_order[0]]
    require_exact_identity(clean_output, first.clean_patch_output)
    return SequentialAudit(
        stages=expected_order,
        output_shape=shape,
        identity_exact=True,
        full_cut_exact=True,
        finite=True,
    )

