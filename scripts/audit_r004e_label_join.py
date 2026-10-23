"""Audit whether a node-label manifest exactly covers frozen R004E anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from aorta_drive.protocol.training_join import (
    audit_label_join,
    label_manifest_tokens,
    load_training_anchors,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    args = parser.parse_args()

    anchors = load_training_anchors(args.time_set)
    tokens = label_manifest_tokens(args.label_manifest)
    audit = audit_label_join(anchors, tokens)
    result = {
        "experiment_id": "R004E-existing-label-join-audit",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "status": "PASS" if audit.complete else "FAIL",
        "time_set_path": str(args.time_set.resolve()),
        "time_set_sha256": _sha256(args.time_set),
        "label_manifest_path": str(args.label_manifest.resolve()),
        "label_manifest_sha256": _sha256(args.label_manifest),
        "allowed_splits": ["train", "development"],
        "calibration_test_or_target_accessed": False,
        "join": {
            "anchor_count": audit.anchor_count,
            "recording_group_count": audit.recording_group_count,
            "label_token_count": audit.label_token_count,
            "matched_anchor_count": audit.matched_anchor_count,
            "missing_anchor_count": audit.missing_anchor_count,
            "unexpected_label_count": audit.unexpected_label_count,
            "duplicate_label_token_count": audit.duplicate_label_token_count,
        },
        "interpretation": (
            "label manifest exactly covers the frozen anchors"
            if audit.complete
            else "the supplied label manifest cannot supervise the frozen R004E anchors"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not audit.complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
