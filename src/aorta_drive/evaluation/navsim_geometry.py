"""Auditable bridge between official NAVSIM states and proof geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from aorta_drive.certification.geometry import VehicleRectangle


@dataclass(frozen=True)
class VehicleGeometryContract:
    """Rear-axle body convention derived from official vehicle parameters."""

    body: VehicleRectangle
    width_m: float
    length_m: float
    rear_axle_to_center_m: float


@dataclass(frozen=True)
class CornerCompatibilityAudit:
    """Measured agreement with NAVSIM's official corner preprocessing."""

    verified: bool
    maximum_coordinate_disagreement_m: float
    tolerance_m: float
    global_coordinate_scale_m: float


def vehicle_geometry_contract(parameters: Any) -> VehicleGeometryContract:
    """Derive the exact rear-axle rectangle used by the official scorer."""

    width = float(parameters.width)
    length = float(parameters.length)
    rear_axle_to_center = float(parameters.rear_axle_to_center)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (width, length, rear_axle_to_center)
    ):
        raise ValueError("official vehicle parameters must be finite and positive")
    half_length = length / 2.0
    rear_overhang = half_length - rear_axle_to_center
    front_extent = half_length + rear_axle_to_center
    if rear_overhang <= 0.0:
        raise ValueError("rear axle lies outside the official vehicle rectangle")
    return VehicleGeometryContract(
        body=VehicleRectangle(
            rear_overhang=rear_overhang,
            front_extent=front_extent,
            half_width=width / 2.0,
        ),
        width_m=width,
        length_m=length,
        rear_axle_to_center_m=rear_axle_to_center,
    )


def rear_axle_states_to_corners(
    states: npt.ArrayLike,
    *,
    contract: VehicleGeometryContract,
    x_index: int,
    y_index: int,
    heading_index: int,
) -> npt.NDArray[np.float64]:
    """Construct corners in official enum order: FL, RL, RR, FR.

    This implementation uses the analytic left vector ``(-sin, cos)`` rather
    than NAVSIM's ``(cos(yaw+pi/2), sin(yaw+pi/2))`` expression so the audit is
    not a byte-for-byte copy of the official preprocessing implementation.
    """

    array = np.asarray(states, dtype=np.float64)
    if array.ndim != 3 or not np.all(np.isfinite(array)):
        raise ValueError("states must be a finite [batch,time,state] array")
    if min(x_index, y_index, heading_index) < 0 or max(
        x_index, y_index, heading_index
    ) >= array.shape[2]:
        raise ValueError("state pose indices lie outside the state array")
    heading = array[..., heading_index]
    cosine = np.cos(heading)
    sine = np.sin(heading)
    forward = np.stack((cosine, sine), axis=-1)
    left = np.stack((-sine, cosine), axis=-1)
    rear_axle = np.stack((array[..., x_index], array[..., y_index]), axis=-1)
    center = rear_axle + contract.rear_axle_to_center_m * forward
    half_length = contract.length_m / 2.0
    half_width = contract.width_m / 2.0
    front = half_length * forward
    rear = -half_length * forward
    lateral = half_width * left
    return np.asarray(
        np.stack(
            (
                center + front + lateral,
                center + rear + lateral,
                center + rear - lateral,
                center + front - lateral,
            ),
            axis=2,
        ),
        dtype=np.float64,
    )


def audit_official_corners(
    analytic_corners: npt.ArrayLike,
    official_coords: npt.ArrayLike,
    *,
    ulp_multiplier: int = 32,
) -> CornerCompatibilityAudit:
    """Check corner conventions within a frozen global-coordinate ULP budget."""

    analytic = np.asarray(analytic_corners, dtype=np.float64)
    official = np.asarray(official_coords, dtype=np.float64)
    if analytic.ndim != 4 or analytic.shape[-2:] != (4, 2):
        raise ValueError("analytic corners must have shape [batch,time,4,2]")
    if official.shape != analytic.shape:
        raise ValueError("official and analytic corner arrays must align")
    if not np.all(np.isfinite(analytic)) or not np.all(np.isfinite(official)):
        raise ValueError("corner arrays must be finite")
    if not isinstance(ulp_multiplier, int) or ulp_multiplier <= 0:
        raise ValueError("ULP multiplier must be a positive integer")
    scale = max(1.0, float(np.max(np.abs(official))))
    spacing = float(np.spacing(scale))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("global coordinate scale has no finite positive spacing")
    tolerance = ulp_multiplier * spacing
    disagreement = float(np.max(np.abs(analytic - official)))
    return CornerCompatibilityAudit(
        verified=disagreement <= tolerance,
        maximum_coordinate_disagreement_m=disagreement,
        tolerance_m=tolerance,
        global_coordinate_scale_m=scale,
    )
