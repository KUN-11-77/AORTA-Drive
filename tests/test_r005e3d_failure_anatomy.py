from __future__ import annotations

import numpy as np
import pytest

from analyze_r005e3d_failure import (
    _contextual_components,
    _first_divergence_time,
    _geometry_features,
    _selection,
    _threshold_frontier,
)


def test_selection_separates_benefit_and_harm() -> None:
    prediction = np.asarray([[0.0, 0.4], [0.0, 0.3]], dtype=np.float32)
    target = np.asarray([[0.0, 0.2], [0.0, -0.5]], dtype=np.float32)
    mask = np.ones_like(prediction, dtype=np.bool_)
    result = _selection(prediction, target, mask)
    assert result["selected"].tolist() == [1, 1]
    assert result["gain"].tolist() == pytest.approx([0.2, -0.5])


def test_threshold_frontier_is_explicitly_native_relative() -> None:
    prediction = np.asarray(
        [[0.0, 0.4], [0.0, 0.3], [0.0, -0.2]], dtype=np.float32
    )
    target = np.asarray(
        [[0.0, 0.2], [0.0, -0.5], [0.0, 0.7]], dtype=np.float32
    )
    mask = np.ones_like(prediction, dtype=np.bool_)
    rows = _threshold_frontier(prediction, target, mask)
    zero = next(row for row in rows if row["threshold"] == 0.0)
    assert zero["nonnative_choice_count"] == 2
    assert zero["mean_gain_over_native"] == pytest.approx(-0.1)
    assert rows[-1]["nonnative_choice_count"] == 0
    assert rows[-1]["mean_gain_over_native"] == 0.0


def test_first_divergence_time_preserves_never_crossed() -> None:
    distance = np.asarray([[0.0, 0.2, 0.6], [0.0, 0.1, 0.2]], dtype=np.float64)
    result = _first_divergence_time(distance, threshold_m=0.5, sample_interval_s=0.1)
    assert result[0] == pytest.approx(0.2)
    assert np.isnan(result[1])


def test_geometry_features_are_native_relative() -> None:
    poses = np.zeros((1, 2, 3, 3), dtype=np.float32)
    poses[0, 0, :, 0] = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    poses[0, 1, :, 0] = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)
    poses[0, 1, -1, 2] = np.float32(np.pi / 2.0)
    result = _geometry_features(poses)
    assert result["final_position_delta_m"][0, 0] == 0.0
    assert result["final_position_delta_m"][0, 1] == pytest.approx(1.0)
    assert result["path_length_delta_m"][0, 1] == pytest.approx(1.0)
    assert result["final_abs_yaw_delta_rad"][0, 1] == pytest.approx(np.pi / 2.0)


def test_contextual_components_preserve_registered_index() -> None:
    ledgers = [
        {
            "recording_group": "g",
            "anchor_index": 0,
            "candidates": [
                {
                    "registered_index": 0,
                    "filtered_one_frame_score": 0.5,
                    "two_frame_extended_comfort": 1.0,
                    "contextual_utility": 0.5625,
                    "hard_metrics": {"a": 1.0, "b": 1.0},
                },
                {
                    "registered_index": 2,
                    "filtered_one_frame_score": 0.25,
                    "two_frame_extended_comfort": 0.0,
                    "contextual_utility": 0.21875,
                    "hard_metrics": {"a": 1.0, "b": 1.0},
                },
            ],
        }
    ]
    metadata = {0: {"recording_group": "g", "anchor_index": 0}}
    result = _contextual_components(ledgers, metadata, candidate_count=3)
    assert result["filtered"][0, 2] == pytest.approx(0.25)
    assert result["contextual"][0, 2] == pytest.approx(0.21875)
    assert np.isnan(result["filtered"][0, 1])
