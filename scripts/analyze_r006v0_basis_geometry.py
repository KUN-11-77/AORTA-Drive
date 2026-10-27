#!/usr/bin/env python3
"""Post-hoc geometry diagnostic for the stopped R006V0 basis experiment.

This script cannot alter the preregistered R006V0 decision.  It only asks
whether the learned channel subspaces point in similar directions across
folds, which separates rank-threshold instability from wholesale directional
instability.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from aorta_drive.vision.r006_cache import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def analyze(result_root: Path) -> dict[str, Any]:
    ranks = (4, 8, 16, 32)
    layer = "trajectory_query_secondary"
    rank_results: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}

    for rank in ranks:
        bases: dict[int, np.ndarray] = {}
        for fold in range(6):
            fold_dir = result_root / f"fold_{fold}" / "primary"
            result_path = fold_dir / "fold_result.json"
            result = _load_json(result_path)
            row = result["candidates"][layer][str(rank)]
            basis_path = fold_dir / str(row["basis_file"])
            basis = np.load(basis_path, allow_pickle=False).astype(np.float64)
            if basis.shape != (256, rank) or not np.isfinite(basis).all():
                raise ValueError(f"invalid basis: {basis_path}")
            bases[fold] = basis
            input_hashes[str(basis_path.relative_to(result_root)).replace("\\", "/")] = (
                sha256_file(basis_path)
            )

        pairs: list[dict[str, Any]] = []
        for left, right in combinations(range(6), 2):
            singular_values = np.linalg.svd(
                bases[left].T @ bases[right], compute_uv=False
            )
            normalized_projector_overlap = float(
                np.sum(singular_values**2) / rank
            )
            largest_principal_angle_degrees = float(
                np.degrees(
                    np.arccos(np.clip(float(np.min(singular_values)), -1.0, 1.0))
                )
            )
            pairs.append(
                {
                    "left_fold": left,
                    "right_fold": right,
                    "normalized_projector_overlap": normalized_projector_overlap,
                    "smallest_principal_cosine": float(np.min(singular_values)),
                    "median_principal_cosine": float(np.median(singular_values)),
                    "largest_principal_angle_degrees": largest_principal_angle_degrees,
                }
            )

        overlaps = [row["normalized_projector_overlap"] for row in pairs]
        angles = [row["largest_principal_angle_degrees"] for row in pairs]
        rank_results[str(rank)] = {
            "pair_count": len(pairs),
            "normalized_projector_overlap_mean": float(np.mean(overlaps)),
            "normalized_projector_overlap_min": float(np.min(overlaps)),
            "normalized_projector_overlap_max": float(np.max(overlaps)),
            "largest_principal_angle_degrees_mean": float(np.mean(angles)),
            "largest_principal_angle_degrees_max": float(np.max(angles)),
            "pairs": pairs,
        }

    return {
        "artifact": "R006V0_POSTHOC_BASIS_GEOMETRY_DIAGNOSTIC_V1",
        "status": "DESCRIPTIVE_ONLY__PREREGISTERED_STOP_IMMUTABLE",
        "layer": layer,
        "fold_count": 6,
        "rank_results": rank_results,
        "input_basis_sha256": dict(sorted(input_hashes.items())),
        "interpretation_boundary": (
            "Projector overlap is a rotation-invariant descriptive diagnostic. "
            "It neither changes STOP_GLOBAL_SHARED_INTERFACE nor establishes "
            "mediation, DINO recoverability, CPTR efficacy, OOD robustness, or "
            "driving benefit. Folds have overlapping discovery data, so these "
            "pairwise values are not independent confirmation statistics."
        ),
    }


def main() -> int:
    args = parse_args()
    result = analyze(args.result_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact": result["artifact"],
                "status": result["status"],
                "rank32_overlap_mean": result["rank_results"]["32"][
                    "normalized_projector_overlap_mean"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
