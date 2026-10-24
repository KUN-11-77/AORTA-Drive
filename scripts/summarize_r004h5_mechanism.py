"""One-shot, read-only mechanism analysis for the frozen R004H5 result."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from aorta_drive.evaluation.appearance_mechanism import oriented_grid_edges
from aorta_drive.vision.appearance import APPEARANCE_CONDITIONS
from aorta_drive.vision.feature_cache import sha256_file


_CONTROL_KEYS = ("zero", "prior", "pooled", "uniform")


def _cluster_interval(
    values: npt.NDArray[np.float64], rng: np.random.Generator
) -> dict[str, float]:
    if values.shape != (42,) or not np.all(np.isfinite(values)):
        raise ValueError("cluster interval requires 42 finite group values")
    draws = rng.integers(0, len(values), size=(10_000, len(values)))
    means = np.mean(values[draws], axis=1)
    return {
        "mean_m": float(np.mean(values)),
        "ci95_low_m": float(np.quantile(means, 0.025)),
        "ci95_high_m": float(np.quantile(means, 0.975)),
    }


def _group_rows(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["recording_group"])].append(record)
    if len(grouped) != 42 or any(len(rows) != 16 for rows in grouped.values()):
        raise ValueError("analysis requires 42 groups with 16 paired anchors each")
    return dict(sorted(grouped.items()))


def _group_condition_row(
    rows: Sequence[dict[str, Any]], condition: str
) -> dict[str, float]:
    learned = float(
        np.mean([row["conditions"][condition]["task_mae_m"] for row in rows])
    )
    return {
        "learned_task_mae_m": learned,
        "appearance_mae_m": float(
            np.mean([row["conditions"][condition]["appearance_mae_m"] for row in rows])
        ),
        "teacher_cosine": float(
            np.mean([row["conditions"][condition]["teacher_cosine"] for row in rows])
        ),
        "zero_task_mae_m": float(np.mean([row["zero_task_mae_m"] for row in rows])),
        "prior_task_mae_m": float(
            np.mean(
                [row["leave_one_group_out_teacher_prior"]["task_mae_m"] for row in rows]
            )
        ),
        "pooled_task_mae_m": float(
            np.mean(
                [row["probes"][f"spatial_pool_{condition}"]["task_mae_m"] for row in rows]
            )
        ),
        "uniform_task_mae_m": float(
            np.mean([row["probes"]["uniform_fit_mean_DINO"]["task_mae_m"] for row in rows])
        ),
    }


def _paired_control_summary(
    grouped: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(20261024)
    conditions: dict[str, Any] = {}
    worst: dict[str, Any] = {}
    for condition in APPEARANCE_CONDITIONS:
        per_group = {
            group: _group_condition_row(rows, condition)
            for group, rows in grouped.items()
        }
        learned = np.asarray(
            [row["learned_task_mae_m"] for row in per_group.values()], dtype=np.float64
        )
        condition_summary: dict[str, Any] = {
            "learned_task_mae_m": float(np.mean(learned)),
            "appearance_mae_m": float(
                np.mean([row["appearance_mae_m"] for row in per_group.values()])
            ),
            "teacher_cosine": float(
                np.mean([row["teacher_cosine"] for row in per_group.values()])
            ),
            "controls": {},
        }
        for control in _CONTROL_KEYS:
            control_values = np.asarray(
                [row[f"{control}_task_mae_m"] for row in per_group.values()],
                dtype=np.float64,
            )
            margins = control_values - learned
            condition_summary["controls"][control] = {
                "task_mae_m": float(np.mean(control_values)),
                "control_minus_learned": _cluster_interval(margins, rng),
                "learned_better_group_count": int(np.count_nonzero(margins > 0.0)),
                "tied_group_count": int(np.count_nonzero(margins == 0.0)),
            }
        worst_task_group = max(
            per_group, key=lambda group: per_group[group]["learned_task_mae_m"]
        )
        worst_drift_group = max(
            per_group, key=lambda group: per_group[group]["appearance_mae_m"]
        )
        worst_cosine_group = min(
            per_group, key=lambda group: per_group[group]["teacher_cosine"]
        )
        worst[condition] = {
            "highest_task_mae": {
                "recording_group": worst_task_group,
                **per_group[worst_task_group],
            },
            "highest_appearance_drift": {
                "recording_group": worst_drift_group,
                **per_group[worst_drift_group],
            },
            "lowest_teacher_cosine": {
                "recording_group": worst_cosine_group,
                **per_group[worst_cosine_group],
            },
        }
        conditions[condition] = condition_summary
    return conditions, worst


def _slot_probe_summary(
    grouped: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, float]]:
    names = [f"drop_camera_{index}" for index in range(3)] + [
        f"drop_history_{index}" for index in range(4)
    ]
    output: dict[str, dict[str, float]] = {}
    for name in names:
        group_values = []
        for rows in grouped.values():
            group_values.append(
                np.mean(
                    [
                        (
                            row["probes"][name]["task_mae_m"],
                            row["probes"][name]["appearance_mae_m"],
                            row["probes"][name]["teacher_cosine"],
                        )
                        for row in rows
                    ],
                    axis=0,
                )
            )
        macro = np.mean(group_values, axis=0)
        output[name] = {
            "task_mae_m": float(macro[0]),
            "field_change_from_clean_mae_m": float(macro[1]),
            "teacher_cosine": float(macro[2]),
        }
    return output


def _safe_artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("field artifact escapes result root")
    return path


def _salient_edge_proxy(
    grouped: dict[str, list[dict[str, Any]]], result_root: Path
) -> dict[str, Any]:
    threshold = 0.5
    per_group: dict[
        str, dict[str, list[tuple[float, float, float, float, float]]]
    ] = {
        group: {condition: [] for condition in APPEARANCE_CONDITIONS}
        for group in grouped
    }
    total_edges = 0
    salient_edges = 0
    nonempty_anchors = 0
    for group, rows in grouped.items():
        for row in rows:
            path = _safe_artifact(result_root, str(row["field_artifact"]))
            if sha256_file(path) != row["field_artifact_sha256"]:
                raise ValueError("field artifact hash mismatch during salient-edge analysis")
            with np.load(path, allow_pickle=False) as archive:
                fields = {name: np.asarray(archive[name]) for name in archive.files}
            teacher = oriented_grid_edges(fields["teacher"])
            clean = oriented_grid_edges(fields["clean"])
            mask = np.abs(teacher) >= threshold
            total_edges += int(len(teacher))
            salient_edges += int(np.count_nonzero(mask))
            if not np.any(mask):
                continue
            nonempty_anchors += 1
            zero = float(np.mean(np.abs(teacher[mask])))
            clean_energy = float(np.mean(np.abs(clean[mask])))
            for condition in APPEARANCE_CONDITIONS:
                prediction = oriented_grid_edges(fields[condition])
                task = float(np.mean(np.abs(prediction[mask] - teacher[mask])))
                drift = float(np.mean(np.abs(prediction[mask] - clean[mask])))
                denominator = float(
                    np.linalg.norm(prediction[mask]) * np.linalg.norm(teacher[mask])
                )
                cosine = (
                    float(np.dot(prediction[mask], teacher[mask]) / denominator)
                    if denominator > 0.0
                    else 0.0
                )
                per_group[group][condition].append((task, zero, drift, cosine, clean_energy))
    conditions: dict[str, Any] = {}
    for condition in APPEARANCE_CONDITIONS:
        group_means = np.asarray(
            [
                np.mean(per_group[group][condition], axis=0)
                for group in per_group
                if per_group[group][condition]
            ],
            dtype=np.float64,
        )
        if group_means.shape != (42, 5):
            raise ValueError("salient-edge proxy does not cover all 42 groups")
        conditions[condition] = {
            "task_mae_m": float(np.mean(group_means[:, 0])),
            "zero_task_mae_m": float(np.mean(group_means[:, 1])),
            "appearance_mae_m": float(np.mean(group_means[:, 2])),
            "teacher_cosine": float(np.mean(group_means[:, 3])),
            "clean_edge_energy_mae_m": float(np.mean(group_means[:, 4])),
        }
    return {
        "definition": "absolute_teacher_edge_at_least_0p5m_posthoc_geometry_salience_proxy",
        "threshold_m": threshold,
        "total_edge_count": total_edges,
        "salient_edge_count": salient_edges,
        "salient_edge_fraction": salient_edges / total_edges,
        "anchors_with_at_least_one_salient_edge": nonempty_anchors,
        "conditions": conditions,
        "claim_limit": (
            "Teacher-gradient salience is not a trajectory footprint, causal decision region, "
            "or safety metric. This post-hoc diagnostic cannot establish driving benefit."
        ),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    raw = json.loads(args.result.read_text(encoding="utf-8"))
    if raw.get("status") != "COMPLETED" or raw.get("fit_anchor_count") != 672:
        raise ValueError("R004H5 raw result is incomplete")
    records = raw.get("anchor_ledgers")
    if not isinstance(records, list) or len(records) != 672:
        raise ValueError("R004H5 anchor ledger is incomplete")
    grouped = _group_rows(records)
    conditions, worst = _paired_control_summary(grouped)
    return {
        "experiment_id": "R004H5-frozen-mechanism-followup-v1",
        "status": "COMPLETED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "raw_result_sha256": sha256_file(args.result),
        "fit_group_count": 42,
        "fit_anchor_count": 672,
        "bootstrap_seed": 20261024,
        "bootstrap_replicates": 10_000,
        "condition_control_summary": conditions,
        "worst_group_diagnostics": worst,
        "slot_drop_probes": _slot_probe_summary(grouped),
        "teacher_salient_edge_proxy": _salient_edge_proxy(grouped, args.result_root),
        "unresolved_by_existing_artifacts": [
            "mismatched_image_or_scene_permutation_causal_test",
            "input_pixel_and_DINO_feature_perturbation_magnitude",
            "held_out_source_group_generalization",
            "real_weather_or_target_OOD_robustness",
            "trajectory_footprint_decision_relevance",
            "driving_or_safety_improvement",
        ],
        "claim_boundary": (
            "Frozen fit-only post-hoc mechanism analysis. Pooled, uniform and slot-drop "
            "results are off-manifold probes, not retrained fair baselines. Confidence "
            "intervals describe 42 observed fit groups and do not establish generalization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
