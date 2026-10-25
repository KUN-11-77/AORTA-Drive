"""Outcome-free geometry matching for the R005E2 visual control.

The matcher deliberately accepts geometry descriptors and group identities only.
It cannot consume utility labels, DINO features, images, or model predictions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


_QUANTILES = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)


def anchor_key(group: str, anchor_index: int) -> str:
    """Return the canonical identifier for one repeated anchor."""

    if not group:
        raise ValueError("group must be nonempty")
    if anchor_index < 0:
        raise ValueError("anchor_index must be nonnegative")
    return f"{group}::{anchor_index:02d}"


def geometry_descriptor_names(sample_count: int = 9) -> tuple[str, ...]:
    """Return the frozen descriptor schema used by geometry-only matching."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    names: list[str] = []
    for index in range(sample_count):
        names.extend(
            (
                f"native_sample_{index}_x_m",
                f"native_sample_{index}_y_m",
                f"native_sample_{index}_sin_yaw",
                f"native_sample_{index}_cos_yaw",
            )
        )
    names.extend(f"native_step_distance_q{int(q * 100):03d}_m" for q in _QUANTILES)
    names.extend(f"native_abs_dyaw_q{int(q * 100):03d}_rad" for q in _QUANTILES)
    names.extend(
        (
            "native_path_length_m",
            "native_terminal_displacement_m",
            "native_terminal_sin_yaw",
            "native_terminal_cos_yaw",
            "eligible_challenger_count",
        )
    )
    return tuple(names)


def native_geometry_descriptor(
    realized_local_poses: np.ndarray[Any, np.dtype[np.float64]],
    challenger_eligible: np.ndarray[Any, np.dtype[np.bool_]],
    *,
    sample_count: int = 9,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Describe native geometry without labels, pixels, or learned features.

    The first realized trajectory is the frozen native member of the registered
    panel.  Challenger poses are intentionally excluded: the final scalar only
    records legal-domain cardinality, preventing score-derived panel structure
    from entering the visual donor rule.
    """

    poses = np.asarray(realized_local_poses, dtype=np.float64)
    eligible = np.asarray(challenger_eligible, dtype=np.bool_)
    if poses.ndim != 3 or poses.shape[0] != 7 or poses.shape[2] != 3:
        raise ValueError("realized_local_poses must have shape [7, time, 3]")
    if poses.shape[1] < sample_count:
        raise ValueError("trajectory has fewer samples than the descriptor")
    if eligible.shape != (6,):
        raise ValueError("challenger_eligible must have shape [6]")
    if not np.isfinite(poses).all():
        raise ValueError("trajectory geometry must be finite")

    native = poses[0]
    sample_indices = np.rint(
        np.linspace(0, native.shape[0] - 1, sample_count)
    ).astype(np.int64)
    sampled = native[sample_indices]
    sampled_features = np.column_stack(
        (
            sampled[:, 0],
            sampled[:, 1],
            np.sin(sampled[:, 2]),
            np.cos(sampled[:, 2]),
        )
    ).reshape(-1)
    step_distance = np.linalg.norm(np.diff(native[:, :2], axis=0), axis=1)
    absolute_yaw_step = np.abs(np.diff(np.unwrap(native[:, 2])))
    terminal_displacement = float(np.linalg.norm(native[-1, :2] - native[0, :2]))
    tail = np.asarray(
        (
            float(step_distance.sum()),
            terminal_displacement,
            float(np.sin(native[-1, 2])),
            float(np.cos(native[-1, 2])),
            float(eligible.sum()),
        ),
        dtype=np.float64,
    )
    descriptor: np.ndarray[Any, np.dtype[np.float64]] = np.concatenate(
        (
            sampled_features,
            np.quantile(step_distance, _QUANTILES),
            np.quantile(absolute_yaw_step, _QUANTILES),
            tail,
        )
    ).astype(np.float64, copy=False)
    if descriptor.shape != (len(geometry_descriptor_names(sample_count)),):
        raise AssertionError("geometry descriptor schema drift")
    if not np.isfinite(descriptor).all():
        raise ValueError("geometry descriptor must be finite")
    return descriptor


def descriptor_sha256(descriptor: np.ndarray[Any, np.dtype[np.float64]]) -> str:
    """Hash one descriptor after canonical little-endian float64 conversion."""

    canonical = np.asarray(descriptor, dtype="<f8")
    if canonical.ndim != 1 or not np.isfinite(canonical).all():
        raise ValueError("descriptor must be a finite vector")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _pair_tie_break(recipient: str, donor: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}\0{recipient}\0{donor}".encode()).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64)


def globally_match_geometry(
    recipients: Sequence[str],
    donors: Sequence[str],
    *,
    descriptor_by_key: Mapping[str, np.ndarray[Any, np.dtype[np.float64]]],
    group_by_key: Mapping[str, str],
    tie_salt: str,
    require_donor_permutation: bool,
    standard_deviation_floor: float = 1.0e-8,
) -> dict[str, Any]:
    """Solve a deterministic, group-deranged, without-replacement assignment."""

    recipient_keys = sorted(str(value) for value in recipients)
    donor_keys = sorted(str(value) for value in donors)
    if len(recipient_keys) != len(set(recipient_keys)):
        raise ValueError("recipient keys must be unique")
    if len(donor_keys) != len(set(donor_keys)):
        raise ValueError("donor keys must be unique")
    if not recipient_keys or len(recipient_keys) > len(donor_keys):
        raise ValueError("donor pool must cover the nonempty recipient set")
    if require_donor_permutation and len(recipient_keys) != len(donor_keys):
        raise ValueError("a donor permutation requires equal set sizes")
    if standard_deviation_floor <= 0.0:
        raise ValueError("standard_deviation_floor must be positive")

    try:
        recipient_matrix = np.stack([descriptor_by_key[key] for key in recipient_keys])
        donor_matrix = np.stack([descriptor_by_key[key] for key in donor_keys])
    except KeyError as error:
        raise ValueError(f"missing geometry descriptor: {error}") from error
    if recipient_matrix.shape[1:] != donor_matrix.shape[1:] or recipient_matrix.ndim != 2:
        raise ValueError("recipient and donor descriptor shapes differ")
    if not np.isfinite(recipient_matrix).all() or not np.isfinite(donor_matrix).all():
        raise ValueError("geometry descriptors must be finite")

    mean = donor_matrix.mean(axis=0)
    standard_deviation = donor_matrix.std(axis=0)
    active = standard_deviation > standard_deviation_floor
    if not bool(active.any()):
        raise ValueError("donor geometry has no varying descriptor dimension")
    recipient_z = (recipient_matrix[:, active] - mean[active]) / standard_deviation[active]
    donor_z = (donor_matrix[:, active] - mean[active]) / standard_deviation[active]
    delta = recipient_z[:, None, :] - donor_z[None, :, :]
    base_cost = np.mean(np.square(delta), axis=2)
    invalid = np.asarray(
        [
            [group_by_key[recipient] == group_by_key[donor] for donor in donor_keys]
            for recipient in recipient_keys
        ],
        dtype=np.bool_,
    )
    if bool(np.any(np.all(invalid, axis=1))):
        raise ValueError("at least one recipient has no cross-group donor")
    finite_max = float(np.max(base_cost))
    invalid_cost = max(1.0, finite_max) * 1.0e9
    cost = np.where(invalid, invalid_cost, base_cost)
    tie = np.asarray(
        [
            [_pair_tie_break(recipient, donor, tie_salt) for donor in donor_keys]
            for recipient in recipient_keys
        ],
        dtype=np.float64,
    )
    cost = cost + tie * 1.0e-12
    row_indices, column_indices = linear_sum_assignment(cost)
    if len(row_indices) != len(recipient_keys):
        raise ValueError("assignment did not cover every recipient")

    rows: list[dict[str, Any]] = []
    for row_index, column_index in sorted(
        zip(row_indices.tolist(), column_indices.tolist()), key=lambda pair: pair[0]
    ):
        if invalid[row_index, column_index]:
            raise ValueError("optimizer selected a forbidden same-group donor")
        recipient = recipient_keys[row_index]
        donor = donor_keys[column_index]
        rows.append(
            {
                "recipient": recipient,
                "donor": donor,
                "standardized_squared_distance": float(base_cost[row_index, column_index]),
            }
        )
    selected_donors = [str(row["donor"]) for row in rows]
    if len(selected_donors) != len(set(selected_donors)):
        raise AssertionError("without-replacement assignment reused a donor")
    if require_donor_permutation and set(selected_donors) != set(donor_keys):
        raise AssertionError("training donor marginal is not an exact permutation")
    distances = np.asarray(
        [float(row["standardized_squared_distance"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "matching_method": "global_linear_sum_assignment",
        "matching_inputs": "native_geometry_and_legal_domain_cardinality_only",
        "labels_images_dino_or_model_outputs_used": False,
        "recipient_count": len(recipient_keys),
        "donor_pool_count": len(donor_keys),
        "selected_donor_count": len(selected_donors),
        "selected_donors_unique": True,
        "donor_marginal_exact_permutation": require_donor_permutation,
        "active_descriptor_dimension_count": int(active.sum()),
        "standard_deviation_floor": standard_deviation_floor,
        "distance_mean": float(distances.mean()),
        "distance_p95": float(np.quantile(distances, 0.95)),
        "distance_max": float(distances.max()),
        "rows": rows,
    }


def audit_geometry_matching(
    assignment: Mapping[str, Any],
    *,
    expected_recipients: Sequence[str],
    allowed_donors: Sequence[str],
    group_by_key: Mapping[str, str],
    require_donor_permutation: bool,
) -> dict[str, Any]:
    """Fail closed on leakage, omission, reuse, or same-group visual donors."""

    if assignment.get("labels_images_dino_or_model_outputs_used") is not False:
        raise ValueError("matching is not outcome/content free")
    rows = [dict(row) for row in assignment.get("rows", [])]
    recipients = [str(row["recipient"]) for row in rows]
    donors = [str(row["donor"]) for row in rows]
    if sorted(recipients) != sorted(str(value) for value in expected_recipients):
        raise ValueError("assignment recipient roster mismatch")
    if len(recipients) != len(set(recipients)):
        raise ValueError("a recipient appears more than once")
    if len(donors) != len(set(donors)):
        raise ValueError("a donor is reused")
    allowed = set(str(value) for value in allowed_donors)
    if not set(donors) <= allowed:
        raise ValueError("assignment uses a donor outside its allowed training pool")
    for recipient, donor in zip(recipients, donors):
        if group_by_key[recipient] == group_by_key[donor]:
            raise ValueError("assignment contains a same-group donor")
    if require_donor_permutation and set(donors) != allowed:
        raise ValueError("training donor marginal is not an exact permutation")
    distances = np.asarray(
        [float(row["standardized_squared_distance"]) for row in rows]
    )
    if not np.isfinite(distances).all() or bool(np.any(distances < 0.0)):
        raise ValueError("assignment distances must be finite and nonnegative")
    return {
        "status": "PASS",
        "recipient_count": len(recipients),
        "donor_count": len(donors),
        "same_group_count": 0,
        "outside_allowed_pool_count": 0,
        "donor_reuse_count": 0,
        "donor_marginal_exact_permutation": require_donor_permutation,
        "labels_images_dino_or_model_outputs_used": False,
    }
