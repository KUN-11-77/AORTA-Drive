#!/usr/bin/env python
"""Mechanical audit for an R006V0 source-fit activation cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aorta_drive.vision.r006_cache import sha256_file, validate_group_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()

    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def check(name: str, value: bool) -> None:
        checks.append({"check": name, "pass": bool(value)})

    check("manifest PASS", manifest.get("status") == "PASS")
    check("sealed roles untouched", manifest.get("development_calibration_test_navdream_target_accessed") is False)
    check("no scientific metrics", manifest.get("scientific_metrics_emitted") is False)
    expected_groups = int(manifest.get("groups", -1))
    expected_anchors = int(manifest.get("anchors", -1))
    if not args.allow_smoke:
        check("complete formal cache", manifest.get("complete_formal_cache") is True)
        check("42 groups", expected_groups == 42)
        check("672 anchors", expected_anchors == 672)
    else:
        check("smoke group count positive", 0 < expected_groups <= 42)
        check("smoke anchor count", expected_anchors == expected_groups * 16)

    seen_groups = set()
    seen_files = set()
    schema = None
    total_anchors = 0
    for record in manifest.get("shards", []):
        path = args.cache_dir / record["file"]
        check(f"{record['file']}: exists", path.is_file())
        if not path.is_file():
            continue
        check(f"{record['file']}: hash", sha256_file(path) == record["sha256"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        shard_schema = validate_group_shard(payload)
        check(f"{record['file']}: group binding", payload["recording_group"] == record["recording_group"])
        check(f"{record['file']}: unique group", payload["recording_group"] not in seen_groups)
        check(f"{record['file']}: unique file", record["file"] not in seen_files)
        seen_groups.add(payload["recording_group"])
        seen_files.add(record["file"])
        total_anchors += len(payload["anchors"])
        if schema is None:
            schema = shard_schema
        check(f"{record['file']}: schema", shard_schema == schema == manifest.get("schema"))

    check("shard count", len(seen_groups) == expected_groups)
    check("anchor total", total_anchors == expected_anchors)
    passed = all(bool(row["pass"]) for row in checks)
    report = {
        "artifact": "R006V0_ACTIVATION_CACHE_AUDIT_V1",
        "status": "PASS" if passed else "FAIL",
        "integrity_scope": "source-fit activation-cache mechanics only",
        "manifest_sha256": sha256_file(manifest_path),
        "check_count": len(checks),
        "failed_check_count": sum(not bool(row["pass"]) for row in checks),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(checks)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
