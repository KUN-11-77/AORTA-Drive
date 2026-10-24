"""Exact finite-graph primitives for the R004H7 decision quotient.

These functions deliberately operate on finite arrays and frozen node sets. They
do not implement a continuous swept-footprint certificate or a driving-safety
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
CandidatePanel = Sequence[Sequence[int] | IntArray]


@dataclass(frozen=True)
class DecisionComparison:
    """Deterministic teacher-versus-prediction selection diagnostics."""

    teacher_index: int
    predicted_index: int
    regret: float
    flip: bool
    exact_teacher_tie: bool


def _as_potential(values: Sequence[float] | FloatArray) -> FloatArray:
    potential = np.asarray(values, dtype=np.float64)
    if potential.ndim != 1 or potential.size == 0:
        raise ValueError("potential must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(potential)):
        raise ValueError("potential must contain only finite values")
    return potential


def _validated_panel(panel: CandidatePanel, *, node_count: int) -> tuple[IntArray, ...]:
    if len(panel) < 2:
        raise ValueError("panel must contain native plus at least one challenger")
    result: list[IntArray] = []
    for candidate in panel:
        nodes = np.asarray(candidate, dtype=np.int64)
        if nodes.ndim != 1 or nodes.size == 0:
            raise ValueError("every candidate node set must be non-empty and one-dimensional")
        if np.any(nodes < 0) or np.any(nodes >= node_count):
            raise ValueError("candidate node index is outside the potential")
        if np.unique(nodes).size != nodes.size:
            raise ValueError("candidate node sets must not contain duplicate nodes")
        result.append(nodes)
    return tuple(result)


def set_min_contrasts(
    potential: Sequence[float] | FloatArray,
    panel: CandidatePanel,
) -> FloatArray:
    """Return exact challenger-minus-native set-min contrasts.

    Candidate zero is native. The function is exactly invariant to a common
    additive offset of the potential.
    """

    values = _as_potential(potential)
    node_sets = _validated_panel(panel, node_count=values.size)
    native_min = float(np.min(values[node_sets[0]]))
    return np.asarray(
        [float(np.min(values[nodes])) - native_min for nodes in node_sets[1:]],
        dtype=np.float64,
    )


def set_min_oscillation_bounds(
    first: Sequence[float] | FloatArray,
    second: Sequence[float] | FloatArray,
    panel: CandidatePanel,
) -> FloatArray:
    """Return the registered deterministic contrast-error upper bounds.

    For challenger j the bound is the oscillation of first minus second on
    the union of native and challenger nodes.
    """

    left = _as_potential(first)
    right = _as_potential(second)
    if left.shape != right.shape:
        raise ValueError("potentials must have identical shape")
    node_sets = _validated_panel(panel, node_count=left.size)
    error = left - right
    native = node_sets[0]
    bounds: list[float] = []
    for challenger in node_sets[1:]:
        union = np.union1d(native, challenger)
        local = error[union]
        bounds.append(float(np.max(local) - np.min(local)))
    return np.asarray(bounds, dtype=np.float64)


def decision_error_eta(
    teacher_contrasts: Sequence[float] | FloatArray,
    predicted_contrasts: Sequence[float] | FloatArray,
) -> float:
    """Return maximum absolute native-relative contrast error."""

    teacher = _as_potential(teacher_contrasts)
    predicted = _as_potential(predicted_contrasts)
    if teacher.shape != predicted.shape:
        raise ValueError("teacher and predicted contrasts must have identical shape")
    return float(np.max(np.abs(predicted - teacher)))


def selection_comparison(
    teacher_contrasts: Sequence[float] | FloatArray,
    predicted_contrasts: Sequence[float] | FloatArray,
) -> DecisionComparison:
    """Compare deterministic selections after prepending native margin zero.

    numpy.argmax chooses the first occurrence, which freezes the tie policy
    to native-first and then the registered challenger order.
    """

    teacher = _as_potential(teacher_contrasts)
    predicted = _as_potential(predicted_contrasts)
    if teacher.shape != predicted.shape:
        raise ValueError("teacher and predicted contrasts must have identical shape")
    teacher_with_native = np.concatenate((np.zeros(1, dtype=np.float64), teacher))
    predicted_with_native = np.concatenate((np.zeros(1, dtype=np.float64), predicted))
    teacher_index = int(np.argmax(teacher_with_native))
    predicted_index = int(np.argmax(predicted_with_native))
    sorted_teacher = np.sort(teacher_with_native)
    exact_teacher_tie = bool(
        sorted_teacher.size >= 2 and sorted_teacher[-1] == sorted_teacher[-2]
    )
    regret = float(
        teacher_with_native[teacher_index] - teacher_with_native[predicted_index]
    )
    return DecisionComparison(
        teacher_index=teacher_index,
        predicted_index=predicted_index,
        regret=regret,
        flip=teacher_index != predicted_index,
        exact_teacher_tie=exact_teacher_tie,
    )


def incidence_matrix(node_count: int, edges: Sequence[tuple[int, int]]) -> FloatArray:
    """Construct a fixed-orientation incidence matrix."""

    if node_count < 2:
        raise ValueError("node_count must be at least two")
    if len(edges) == 0:
        raise ValueError("edges must be non-empty")
    matrix = np.zeros((len(edges), node_count), dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    for row, (source, target) in enumerate(edges):
        if source == target:
            raise ValueError("self edges are not allowed")
        if source < 0 or target < 0 or source >= node_count or target >= node_count:
            raise ValueError("edge endpoint is outside the graph")
        canonical = (min(source, target), max(source, target))
        if canonical in seen:
            raise ValueError("duplicate undirected edges are not allowed")
        seen.add(canonical)
        matrix[row, source] = -1.0
        matrix[row, target] = 1.0
    if np.linalg.matrix_rank(matrix) != node_count - 1:
        raise ValueError("graph must be connected")
    return matrix


def recover_zero_mean_potential(
    edge_values: Sequence[float] | FloatArray,
    incidence: FloatArray,
) -> FloatArray:
    """Recover the zero-mean least-squares potential using a KKT solve."""

    matrix = np.asarray(incidence, dtype=np.float64)
    values = _as_potential(edge_values)
    if matrix.ndim != 2 or matrix.shape[0] != values.size:
        raise ValueError("incidence rows must match edge_values")
    node_count = matrix.shape[1]
    if node_count < 2 or np.linalg.matrix_rank(matrix) != node_count - 1:
        raise ValueError("incidence must describe a connected graph")
    laplacian = matrix.T @ matrix
    kkt = np.block(
        [
            [laplacian, np.ones((node_count, 1), dtype=np.float64)],
            [np.ones((1, node_count), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)],
        ]
    )
    rhs = np.concatenate((matrix.T @ values, np.zeros(1, dtype=np.float64)))
    solution = np.linalg.solve(kkt, rhs)[:node_count]
    if not np.all(np.isfinite(solution)):
        raise FloatingPointError("potential recovery produced non-finite values")
    return np.asarray(solution, dtype=np.float64)


def cut_space_projection(
    edge_values: Sequence[float] | FloatArray,
    incidence: FloatArray,
) -> FloatArray:
    """Orthogonally project arbitrary edge values onto im(D)."""

    matrix = np.asarray(incidence, dtype=np.float64)
    potential = recover_zero_mean_potential(edge_values, matrix)
    return np.asarray(matrix @ potential, dtype=np.float64)
