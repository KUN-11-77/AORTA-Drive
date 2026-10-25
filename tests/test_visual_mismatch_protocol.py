from __future__ import annotations

import copy

import numpy as np
import pytest

from aorta_drive.protocol.visual_mismatch import (
    anchor_key,
    audit_geometry_matching,
    descriptor_sha256,
    geometry_descriptor_names,
    globally_match_geometry,
    native_geometry_descriptor,
)


def _panel(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 4.0, 41)
    native = np.column_stack(
        (3.0 * time, offset + 0.2 * np.sin(time), 0.03 * time)
    )
    poses = np.stack([native + np.asarray([0.0, 0.1 * index, 0.0]) for index in range(7)])
    return poses.astype(np.float64), np.asarray([True, True, False, True, False, True])


def test_native_geometry_descriptor_is_finite_and_excludes_challenger_poses() -> None:
    poses, eligible = _panel()
    first = native_geometry_descriptor(poses, eligible)
    changed = poses.copy()
    changed[1:, :, :] += 10_000.0
    second = native_geometry_descriptor(changed, eligible)
    assert first.shape == (len(geometry_descriptor_names()),)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert descriptor_sha256(first) == descriptor_sha256(second)


def test_descriptor_changes_with_native_geometry_or_domain_cardinality() -> None:
    poses, eligible = _panel()
    base = native_geometry_descriptor(poses, eligible)
    changed_native = poses.copy()
    changed_native[0, :, 1] += np.linspace(0.0, 2.0, 41)
    changed_eligible = eligible.copy()
    changed_eligible[2] = True
    assert not np.array_equal(base, native_geometry_descriptor(changed_native, eligible))
    assert not np.array_equal(base, native_geometry_descriptor(poses, changed_eligible))


def _matching_inputs() -> tuple[list[str], dict[str, str], dict[str, np.ndarray]]:
    keys: list[str] = []
    groups: dict[str, str] = {}
    descriptors: dict[str, np.ndarray] = {}
    for group_index in range(4):
        group = f"group-{group_index}"
        for anchor_index in range(2):
            key = anchor_key(group, anchor_index)
            keys.append(key)
            groups[key] = group
            descriptors[key] = np.asarray(
                [group_index + anchor_index * 0.1, group_index**2 + anchor_index],
                dtype=np.float64,
            )
    return keys, groups, descriptors


def test_global_assignment_is_cross_group_unique_and_deterministic() -> None:
    keys, groups, descriptors = _matching_inputs()
    first = globally_match_geometry(
        keys,
        keys,
        descriptor_by_key=descriptors,
        group_by_key=groups,
        tie_salt="test",
        require_donor_permutation=True,
    )
    second = globally_match_geometry(
        list(reversed(keys)),
        list(reversed(keys)),
        descriptor_by_key=descriptors,
        group_by_key=groups,
        tie_salt="test",
        require_donor_permutation=True,
    )
    assert first == second
    assert {row["donor"] for row in first["rows"]} == set(keys)
    audit = audit_geometry_matching(
        first,
        expected_recipients=keys,
        allowed_donors=keys,
        group_by_key=groups,
        require_donor_permutation=True,
    )
    assert audit["status"] == "PASS"
    assert audit["same_group_count"] == 0


def test_rectangular_assignment_uses_training_donors_without_replacement() -> None:
    keys, groups, descriptors = _matching_inputs()
    recipients = keys[:2]
    donors = keys[2:]
    assignment = globally_match_geometry(
        recipients,
        donors,
        descriptor_by_key=descriptors,
        group_by_key=groups,
        tie_salt="rectangular",
        require_donor_permutation=False,
    )
    assert len(assignment["rows"]) == 2
    assert len({row["donor"] for row in assignment["rows"]}) == 2
    audit_geometry_matching(
        assignment,
        expected_recipients=recipients,
        allowed_donors=donors,
        group_by_key=groups,
        require_donor_permutation=False,
    )


@pytest.mark.parametrize("mutation", ["same_group", "reuse", "outside", "content_flag"])
def test_assignment_audit_fails_closed(mutation: str) -> None:
    keys, groups, descriptors = _matching_inputs()
    assignment = globally_match_geometry(
        keys,
        keys,
        descriptor_by_key=descriptors,
        group_by_key=groups,
        tie_salt="test",
        require_donor_permutation=True,
    )
    broken = copy.deepcopy(assignment)
    if mutation == "same_group":
        recipient = broken["rows"][0]["recipient"]
        broken["rows"][0]["donor"] = recipient
    elif mutation == "reuse":
        broken["rows"][1]["donor"] = broken["rows"][0]["donor"]
    elif mutation == "outside":
        outside = anchor_key("outside", 0)
        groups[outside] = "outside"
        broken["rows"][0]["donor"] = outside
    else:
        broken["labels_images_dino_or_model_outputs_used"] = True
    with pytest.raises(ValueError):
        audit_geometry_matching(
            broken,
            expected_recipients=keys,
            allowed_donors=keys,
            group_by_key=groups,
            require_donor_permutation=True,
        )


@pytest.mark.parametrize("bad_shape", [(6, 41, 3), (7, 41, 2), (7, 8, 3)])
def test_descriptor_rejects_wrong_panel_shape(bad_shape: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        native_geometry_descriptor(
            np.zeros(bad_shape, dtype=np.float64),
            np.ones(6, dtype=np.bool_),
        )
