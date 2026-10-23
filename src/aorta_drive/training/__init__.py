"""Training-data and optimization contracts for registered AORTA probes."""

from aorta_drive.training.r004ec import (
    JoinedAnchor,
    LoadedAnchor,
    deterministic_interleaved_epoch,
    load_joined_anchors,
    load_training_anchor,
    sample_dense_features_torch,
)

__all__ = [
    "JoinedAnchor",
    "LoadedAnchor",
    "deterministic_interleaved_epoch",
    "load_joined_anchors",
    "load_training_anchor",
    "sample_dense_features_torch",
]
