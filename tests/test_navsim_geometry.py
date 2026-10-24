"""Dependency-free tests for NAVSIM vehicle-geometry compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml  # type: ignore[import-untyped]

from aorta_drive.evaluation.navsim_geometry import (
    audit_official_corners,
    rear_axle_states_to_corners,
    vehicle_geometry_contract,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Parameters:
    width: float = 2.297
    length: float = 5.176
    rear_axle_to_center: float = 1.461


def test_completed_realized_geometry_config_preserves_registered_source_hashes() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/experiment/r004h4p5i_realized_geometry_interface.yaml"
        ).read_text(encoding="utf-8")
    )
    implementation = config["implementation"]
    assert implementation["navsim_geometry_source_sha256"] == (
        "0f0703a9f3bd692a45f5307f1e91ccd1404dbad455c53007ada34a81852aa1f4"
    )
    assert implementation["geometry_source_sha256"] == (
        "3c47d87a79d2e08af12a97000565d05dfcbef56dbf60022945e88c1aa8749025"
    )
    assert implementation["opportunity_source_sha256"] == (
        "7b433037944677c4af87fef3c961c419ab26ae33ac2b24ff5489338b0623333b"
    )
    assert config["blocking"]["experimental_unit"] == "recording_group"
    assert config["decision"]["prevalence_performance_coverage_or_gate_b_claim"] == (
        "prohibited"
    )


def test_official_vehicle_parameters_recover_frozen_rear_axle_extents() -> None:
    contract = vehicle_geometry_contract(_Parameters())
    assert contract.body.front_extent == 4.049
    assert contract.body.rear_overhang == 1.127
    assert contract.body.half_width == 1.1485


def test_rear_axle_corners_follow_official_enum_order() -> None:
    states = np.zeros((1, 1, 3), dtype=np.float64)
    corners = rear_axle_states_to_corners(
        states,
        contract=vehicle_geometry_contract(_Parameters()),
        x_index=0,
        y_index=1,
        heading_index=2,
    )
    expected = np.array(
        [[[4.049, 1.1485], [-1.127, 1.1485], [-1.127, -1.1485], [4.049, -1.1485]]]
    )
    assert np.array_equal(corners[0], expected)


def test_corner_audit_accepts_roundoff_and_rejects_convention_change() -> None:
    analytic = np.full((1, 2, 4, 2), 1_000_000.0, dtype=np.float64)
    official = analytic.copy()
    official[0, 0, 0, 0] = np.nextafter(official[0, 0, 0, 0], np.inf)
    assert audit_official_corners(analytic, official).verified
    official[0, 0, 0, 0] += 1e-3
    assert not audit_official_corners(analytic, official).verified


def test_corner_audit_rejects_nonfinite_input() -> None:
    analytic = np.zeros((1, 1, 4, 2), dtype=np.float64)
    official = analytic.copy()
    official[0, 0, 0, 0] = np.nan
    try:
        audit_official_corners(analytic, official)
    except ValueError:
        return
    raise AssertionError("non-finite official corners must fail closed")
