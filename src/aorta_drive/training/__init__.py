"""Training-data and optimization contracts for registered AORTA probes."""

from aorta_drive.training.r004ec import (
    JoinedAnchor,
    LoadedAnchor,
    deterministic_interleaved_epoch,
    learning_rate_at_update,
    load_joined_anchors,
    load_training_anchor,
    qualifies_single_group_overfit,
    sample_dense_features_torch,
)

__all__ = [
    "JoinedAnchor",
    "LoadedAnchor",
    "deterministic_interleaved_epoch",
    "learning_rate_at_update",
    "load_joined_anchors",
    "load_training_anchor",
    "qualifies_single_group_overfit",
    "sample_dense_features_torch",
]
