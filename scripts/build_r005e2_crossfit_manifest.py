"""Build the outcome-free R005E2 recording-group cross-fit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from aorta_drive.protocol.visual_recoverability import (
    audit_nested_group_manifest,
    build_nested_group_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r005e1b-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.r005e1b_result.read_text(encoding="utf-8"))
    if result.get("development_calibration_test_navdream_target_accessed") is not False:
        raise ValueError("R005E1B sealed-role flag is not false")
    if result.get("full_official_pdms_claimed") is not False:
        raise ValueError("R005E1B unexpectedly claims full official PDMS")
    rows = list(result["anchor_ledgers"])
    group_counts = Counter(str(row["recording_group"]) for row in rows)
    if len(rows) != 672 or len(group_counts) != 42:
        raise ValueError("expected 42 groups and 672 anchor rows")
    if set(group_counts.values()) != {16}:
        raise ValueError("every R005E1B group must contain exactly 16 anchors")

    manifest: dict[str, Any] = build_nested_group_manifest(
        sorted(group_counts),
        outer_fold_count=6,
        inner_validation_count=7,
        outer_salt="aorta-r005e2-outer-v1",
        inner_salt="aorta-r005e2-inner-v1",
    )
    manifest["source_result"] = str(args.r005e1b_result.as_posix())
    manifest["source_result_sha256"] = _sha256(args.r005e1b_result)
    manifest["anchor_count"] = len(rows)
    manifest["anchors_per_group"] = 16
    manifest["sealed_roles_accessed"] = False
    manifest["scientific_metrics_emitted"] = False
    manifest["audit"] = audit_nested_group_manifest(
        manifest,
        expected_groups=sorted(group_counts),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
