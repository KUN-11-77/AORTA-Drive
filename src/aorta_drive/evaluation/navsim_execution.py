"""Bridge desired local poses through NAVSIM's official execution operator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from aorta_drive.certification.geometry import Pose2D
from aorta_drive.certification.transform import certify_global_to_ego


@dataclass(frozen=True)
class OfficialExecutionBatch:
    """Reference and realized state arrays under the frozen NAVSIM simulator."""

    reference_states: npt.NDArray[np.float64]
    realized_states: npt.NDArray[np.float64]
    reference_local_poses: npt.NDArray[np.float64]
    realized_local_poses: npt.NDArray[np.float64]
    deterministic_replay_bit_exact: bool
    maximum_transform_error_m: float
    proposal_interval_s: float


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _states_to_local_poses(
    states: npt.ArrayLike,
    *,
    current_pose: Pose2D,
    x_index: int,
    y_index: int,
    heading_index: int,
) -> tuple[npt.NDArray[np.float64], float]:
    array = np.asarray(states, dtype=np.float64)
    if array.ndim != 3 or not np.all(np.isfinite(array)):
        raise ValueError("state batch must be a finite [batch,time,state] array")
    if min(x_index, y_index, heading_index) < 0 or max(
        x_index, y_index, heading_index
    ) >= array.shape[2]:
        raise ValueError("state pose indices lie outside the state array")
    poses = np.empty((array.shape[0], array.shape[1], 3), dtype=np.float64)
    maximum_error = 0.0
    for batch_index in range(array.shape[0]):
        for time_index in range(array.shape[1]):
            transform = certify_global_to_ego(
                (
                    float(array[batch_index, time_index, x_index]),
                    float(array[batch_index, time_index, y_index]),
                ),
                current_pose,
            )
            poses[batch_index, time_index] = (
                transform.x,
                transform.y,
                _wrap_angle(
                    float(array[batch_index, time_index, heading_index])
                    - current_pose.yaw
                ),
            )
            maximum_error = max(maximum_error, transform.error_radius)
    return poses, maximum_error


def execute_official_navsim_batch(
    cache: Any,
    desired_local_poses: npt.ArrayLike,
    *,
    desired_interval_s: float = 0.5,
    proposal_interval_s: float = 0.1,
) -> OfficialExecutionBatch:
    """Execute desired challengers with the same operator used by ``pdm_score``.

    ``desired_local_poses`` has shape ``[challenger, 8, 3]`` and excludes the
    current origin.  Native is prepended internally, and all trajectories share
    one initial ego state and one official simulator call.
    """

    desired = np.asarray(desired_local_poses, dtype=np.float64)
    if desired.ndim != 3 or desired.shape[1:] != (8, 3):
        raise ValueError("desired challengers must have shape [batch,8,3]")
    if desired.shape[0] == 0 or not np.all(np.isfinite(desired)):
        raise ValueError("desired challenger batch must be non-empty and finite")
    if not math.isclose(desired_interval_s, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("official desired interface is frozen at 0.5 seconds")
    if not math.isclose(proposal_interval_s, 0.1, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("official proposal simulator is frozen at 0.1 seconds")

    from nuplan.planning.simulation.trajectory.trajectory_sampling import (  # type: ignore[import-untyped]
        TrajectorySampling,
    )

    from navsim.common.dataclasses import Trajectory  # type: ignore[import-not-found]
    from navsim.evaluate.pdm_score import (  # type: ignore[import-not-found]
        get_trajectory_as_array,
        transform_trajectory,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (  # type: ignore[import-not-found]
        PDMSimulator,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (  # type: ignore[import-not-found]
        StateIndex,
    )

    desired_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    start_time = cache.ego_state.time_point
    native_states = get_trajectory_as_array(cache.trajectory, proposal_sampling, start_time)
    challenger_states: list[npt.NDArray[np.float64]] = []
    for poses in desired:
        trajectory = Trajectory(poses=poses, trajectory_sampling=desired_sampling)
        global_trajectory = transform_trajectory(trajectory, cache.ego_state)
        challenger_states.append(
            get_trajectory_as_array(global_trajectory, proposal_sampling, start_time)
        )
    reference_states = np.stack([native_states, *challenger_states]).astype(
        np.float64, copy=False
    )
    first = PDMSimulator(proposal_sampling).simulate_proposals(
        reference_states, cache.ego_state
    )
    second = PDMSimulator(proposal_sampling).simulate_proposals(
        reference_states, cache.ego_state
    )
    if first.shape != reference_states.shape or not np.all(np.isfinite(first)):
        raise ValueError("official simulator returned invalid state shape or values")
    current = cache.ego_state.rear_axle
    current_pose = Pose2D(float(current.x), float(current.y), float(current.heading))
    reference_local, reference_error = _states_to_local_poses(
        reference_states,
        current_pose=current_pose,
        x_index=int(StateIndex.X),
        y_index=int(StateIndex.Y),
        heading_index=int(StateIndex.HEADING),
    )
    realized_local, realized_error = _states_to_local_poses(
        first,
        current_pose=current_pose,
        x_index=int(StateIndex.X),
        y_index=int(StateIndex.Y),
        heading_index=int(StateIndex.HEADING),
    )
    return OfficialExecutionBatch(
        reference_states=reference_states,
        realized_states=first,
        reference_local_poses=reference_local,
        realized_local_poses=realized_local,
        deterministic_replay_bit_exact=bool(np.array_equal(first, second)),
        maximum_transform_error_m=max(reference_error, realized_error),
        proposal_interval_s=proposal_interval_s,
    )
