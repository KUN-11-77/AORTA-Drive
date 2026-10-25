from __future__ import annotations

import numpy as np

from aorta_drive.protocol.visual_mismatch import globally_match_sensitivity


def test_disjoint_geometry_excludes_primary_edges() -> None:
    keys = ["a::00", "b::00", "c::00", "d::00"]
    groups = {key: key[0] for key in keys}
    descriptors = {
        key: np.asarray([float(index), float(index * index)], dtype=np.float64)
        for index, key in enumerate(keys)
    }
    primary = [
        ("a::00", "b::00"),
        ("b::00", "a::00"),
        ("c::00", "d::00"),
        ("d::00", "c::00"),
    ]
    result = globally_match_sensitivity(
        keys,
        keys,
        descriptor_by_key=descriptors,
        group_by_key=groups,
        mode="disjoint_geometry",
        salt="test-disjoint",
        require_donor_permutation=True,
        forbidden_edges=primary,
    )
    edges = {(row["recipient"], row["donor"]) for row in result["rows"]}
    assert edges.isdisjoint(primary)
    assert len({row["donor"] for row in result["rows"]}) == len(keys)


def test_hash_random_is_deterministic_and_geometry_independent_for_selection() -> None:
    keys = ["a::00", "b::00", "c::00", "d::00"]
    groups = {key: key[0] for key in keys}
    first_descriptors = {
        key: np.asarray([float(index), 0.0]) for index, key in enumerate(keys)
    }
    second_descriptors = {
        key: np.asarray([float(index * 100), float(index)])
        for index, key in enumerate(keys)
    }
    kwargs = dict(
        recipients=keys,
        donors=keys,
        group_by_key=groups,
        mode="hash_random",
        salt="fixed-hash-random",
        require_donor_permutation=True,
    )
    first = globally_match_sensitivity(descriptor_by_key=first_descriptors, **kwargs)
    second = globally_match_sensitivity(descriptor_by_key=second_descriptors, **kwargs)
    first_edges = [(row["recipient"], row["donor"]) for row in first["rows"]]
    second_edges = [(row["recipient"], row["donor"]) for row in second["rows"]]
    assert first_edges == second_edges
