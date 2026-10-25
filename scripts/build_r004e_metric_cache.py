"""Build official NAVSIM metric caches for only the frozen R004E anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aorta_drive.protocol.cache_fingerprint import fingerprint_metric_cache
from aorta_drive.protocol.training_join import (
    TrainingAnchor,
    expected_metric_cache_path,
    load_training_anchors,
    select_preflight_anchors,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _balanced_log_chunks(
    anchors: tuple[TrainingAnchor, ...], worker_count: int
) -> list[list[tuple[str, list[str]]]]:
    by_log: dict[str, list[str]] = defaultdict(list)
    for anchor in anchors:
        by_log[anchor.log_name].append(anchor.scene_token)
    chunks: list[list[tuple[str, list[str]]]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for log_name, tokens in sorted(by_log.items(), key=lambda item: (-len(item[1]), item[0])):
        index = min(range(worker_count), key=lambda candidate: (loads[candidate], candidate))
        chunks[index].append((log_name, sorted(tokens)))
        loads[index] += len(tokens)
    return [chunk for chunk in chunks if chunk]


def _cache_chunk(
    chunk: list[tuple[str, list[str]]],
    config_payload: dict[str, Any],
) -> tuple[int, int]:
    from navsim.planning.metric_caching.caching import cache_scenarios
    from omegaconf import OmegaConf

    config = OmegaConf.create(config_payload)
    arguments = [
        {"cfg": config, "log_file": log_name, "tokens": tokens}
        for log_name, tokens in chunk
    ]
    results = cache_scenarios(arguments)
    return (
        sum(int(result.successes) for result in results),
        sum(int(result.failures) for result in results),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-set", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--navsim-root", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--max-anchors", type=int)
    parser.add_argument("--max-recording-groups", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for name, path in (
        ("time_set", args.time_set),
        ("log_root", args.log_root),
        ("map_root", args.map_root),
        ("navsim_root", args.navsim_root),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.num_history_frames <= 0:
        raise ValueError("num_history_frames must be positive")

    output = args.output_directory.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("output directory is nonempty; use a new path or --resume")
    output.mkdir(parents=True, exist_ok=True)
    metric_cache_root = output / "metric_cache"
    metric_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["NUPLAN_MAPS_ROOT"] = str(args.map_root.resolve())

    anchors = load_training_anchors(args.time_set)
    selected = select_preflight_anchors(
        anchors,
        maximum_anchors=args.max_anchors,
        maximum_recording_groups=args.max_recording_groups,
    )
    worker_count = min(args.workers, len({anchor.log_name for anchor in selected}))
    config_payload: dict[str, Any] = {
        "train_test_split": {
            "scene_filter": {
                "_target_": "navsim.common.dataclasses.SceneFilter",
                "_convert_": "all",
                "num_history_frames": args.num_history_frames,
                "num_future_frames": 10,
                "frame_interval": 1,
                "has_route": True,
                "max_scenes": None,
                "log_names": None,
                "tokens": None,
            }
        },
        "navsim_log_path": str(args.log_root.resolve()),
        "synthetic_scenes_path": str((output / "unused_synthetic_scenes").resolve()),
        "metric_cache_path": str(metric_cache_root),
        "force_feature_computation": not args.resume,
        "proposal_sampling": {
            "_target_": (
                "nuplan.planning.simulation.trajectory.trajectory_sampling."
                "TrajectorySampling"
            ),
            "_convert_": "all",
            "num_poses": 40,
            "interval_length": 0.1,
        },
    }
    chunks = _balanced_log_chunks(selected, worker_count)
    successes = 0
    failures = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_cache_chunk, chunk, config_payload) for chunk in chunks]
        for future in as_completed(futures):
            chunk_successes, chunk_failures = future.result()
            successes += chunk_successes
            failures += chunk_failures
            print(
                f"completed {successes + failures}/{len(selected)} "
                f"success={successes} failure={failures}",
                flush=True,
            )

    records: list[dict[str, object]] = []
    missing: list[str] = []
    for anchor in selected:
        path = expected_metric_cache_path(metric_cache_root, anchor)
        if not path.is_file():
            missing.append(str(path))
            continue
        records.append(
            {
                "recording_group": anchor.recording_group,
                "anchor_index": anchor.anchor_index,
                "split": anchor.split,
                "log_name": anchor.log_name,
                "scene_token": anchor.scene_token,
                "relative_cache_path": path.relative_to(output).as_posix(),
                "cache_bytes": path.stat().st_size,
                "cache_sha256": _sha256(path),
                "teacher_input_sha256": fingerprint_metric_cache(path),
            }
        )
    if failures or missing or successes != len(selected) or len(records) != len(selected):
        raise RuntimeError(
            f"incomplete metric cache: successes={successes}, failures={failures}, "
            f"missing={len(missing)}, expected={len(selected)}"
        )

    source_files = {
        "navsim_caching": args.navsim_root / "planning/metric_caching/caching.py",
        "metric_cache_processor": (
            args.navsim_root / "planning/metric_caching/metric_cache_processor.py"
        ),
        "metric_cache": args.navsim_root / "planning/metric_caching/metric_cache.py",
    }
    logical_payload = [
        {
            "recording_group": record["recording_group"],
            "anchor_index": record["anchor_index"],
            "split": record["split"],
            "log_name": record["log_name"],
            "scene_token": record["scene_token"],
            "teacher_input_sha256": record["teacher_input_sha256"],
        }
        for record in records
    ]
    logical_hash = hashlib.sha256(
        json.dumps(logical_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "experiment_id": "R004E-frozen-anchor-metric-cache",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": args.implementation_revision,
        "time_set_sha256": _sha256(args.time_set),
        "allowed_splits": sorted({anchor.split for anchor in selected}),
        "calibration_test_or_target_accessed": False,
        "anchor_count": len(selected),
        "recording_group_count": len({anchor.recording_group for anchor in selected}),
        "log_count": len({anchor.log_name for anchor in selected}),
        "worker_count": worker_count,
        "num_history_frames": args.num_history_frames,
        "resume": args.resume,
        "logical_manifest_sha256": logical_hash,
        "navsim_source_sha256": {
            name: _sha256(path) for name, path in source_files.items()
        },
        "records": records,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.move(temporary, output / "manifest.json")
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
