from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from aorta_drive.vision.feature_cache import sha256_file
from audit_r004h7bs1_development_support import audit
from run_r004h7bs1_development_support import run


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_development_support_runner_and_auditor_reproduce_pass(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    groups = tuple(f"group-{index:02d}" for index in range(24))
    source_split = tmp_path / "split.csv"
    with source_split.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("recording_group", "split"))
        writer.writeheader()
        writer.writerows(
            {"recording_group": group, "split": "development"} for group in groups
        )
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "H7BS1": {
                    "minimum_valid_anchors_per_group": 12,
                    "candidate_panel": {
                        "node_grid": {
                            "x_min_m": -32.0,
                            "y_min_m": -32.0,
                            "columns": 64,
                            "rows": 64,
                            "resolution_m": 1.0,
                        },
                        "vehicle": {
                            "rear_overhang_m": 1.127,
                            "front_extent_m": 4.049,
                            "half_width_m": 1.1485,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    panel_root = tmp_path / "panels-root"
    (panel_root / "panels").mkdir(parents=True)
    records = []
    realized = np.zeros((7, 41, 3), dtype=np.float64)
    eligible = np.ones(6, dtype=np.bool_)
    for group in groups:
        for anchor in range(16):
            scene = f"{group}-{anchor:02d}"
            relative = f"panels/{scene}.npz"
            artifact = panel_root / relative
            np.savez_compressed(
                artifact,
                realized_local_poses=realized,
                challenger_eligible=eligible,
            )
            records.append(
                {
                    "recording_group": group,
                    "anchor_index": anchor,
                    "scene_token": scene,
                    "panel_artifact": relative,
                    "panel_artifact_sha256": sha256_file(artifact),
                }
            )
    panel_manifest = tmp_path / "panels.json"
    _json(
        panel_manifest,
        {
            "record_count": 384,
            "outcome_fields_included": False,
            "records": records,
        },
    )
    fit_result = tmp_path / "fit.json"
    _json(
        fit_result,
        {
            "decision": "FIT_PREFIX_SELECTED",
            "development_accessed": False,
            "summary": {"selected_prefix_index": 25},
        },
    )
    fit_audit = tmp_path / "fit-audit.json"
    _json(
        fit_audit,
        {
            "status": "PASS",
            "result_sha256": sha256_file(fit_result),
            "recomputed_selected_prefix_index": 25,
        },
    )
    lock = tmp_path / "lock.yaml"
    lock.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "synthetic-h7bs1-development",
                "selected_prefix_index": 25,
                "frozen_input_sha256": {
                    "fit_result": sha256_file(fit_result),
                    "fit_audit": sha256_file(fit_audit),
                    "development_panel_manifest": sha256_file(panel_manifest),
                    "source_split_manifest": sha256_file(source_split),
                    "base_config": sha256_file(base_config),
                },
                "implementation_sha256": {
                    "support_adequacy_source": sha256_file(
                        repository / "src/aorta_drive/evaluation/support_adequacy.py"
                    ),
                    "runner_source": sha256_file(
                        repository / "scripts/run_r004h7bs1_development_support.py"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    result = run(
        argparse.Namespace(
            lock=lock,
            base_config=base_config,
            fit_result=fit_result,
            fit_audit=fit_audit,
            panel_manifest=panel_manifest,
            panel_root=panel_root,
            source_split_manifest=source_split,
            output=result_path,
        )
    )
    assert result["decision"] == "DEVELOPMENT_SUPPORT_PASS"
    assert result["summary"]["by_prefix"]["25"]["valid_anchor_count"] == 384
    audit_path = tmp_path / "audit.json"
    audited = audit(
        argparse.Namespace(
            lock=lock,
            base_config=base_config,
            result=result_path,
            panel_root=panel_root,
            source_split_manifest=source_split,
            output=audit_path,
        )
    )
    assert audited["status"] == "PASS"
    assert audited["recomputed_decision"] == "DEVELOPMENT_SUPPORT_PASS"
