"""Training-data and optimization contracts for registered AORTA probes."""

from aorta_drive.training.r004ec import (
    JoinedAnchor,
    LoadedAnchor,
    TeacherArrays,
    deterministic_interleaved_epoch,
    learning_rate_at_update,
    load_joined_anchors,
    load_teacher_arrays,
    load_training_anchor,
    qualifies_single_group_overfit,
    qualifies_visual_necessity,
    sample_dense_features_torch,
    summarize_group_maxima,
)

__all__ = [
    "JoinedAnchor",
    "LoadedAnchor",
    "TeacherArrays",
    "deterministic_interleaved_epoch",
    "learning_rate_at_update",
    "load_joined_anchors",
    "load_teacher_arrays",
    "load_training_anchor",
    "qualifies_single_group_overfit",
    "qualifies_visual_necessity",
    "sample_dense_features_torch",
    "summarize_group_maxima",
]
