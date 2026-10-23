"""Read-only R002F audit of proof-carrying NAVSIM coordinate transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import pathlib
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aorta_drive.certification.geometry import Pose2D
from aorta_drive.certification.transform import certify_ego_to_global

PROBE_AXIS_M = (-50.0, 0.0, 50.0)


@dataclass(frozen=True)
class CacheTransformAudit:
    cache_id: str
    sha256: str
    heading_rad: float
    max_epsilon_transform_m: float
    max_libm_difference_m: float
    max_libm_to_diagnostic_budget_ratio: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pickle(path: Path) -> Any:
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    return pickle.loads(lzma.decompress(path.read_bytes()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cache-count", type=int, default=164)
    arguments = parser.parse_args()

    cache_root = Path(os.environ["AORTA_NAVSIM_METRIC_CACHE_ROOT"]).resolve()
    navsim_root = Path(os.environ["AORTA_NAVSIM_SOURCE_ROOT"]).resolve()
    sys.path.insert(0, str(navsim_root.parent))
    repository_root = Path(__file__).resolve().parents[1]
    transform_source = repository_root / "src/aorta_drive/certification/transform.py"

    cache_paths = sorted(cache_root.rglob("metric_cache.pkl"))
    failures: list[str] = []
    if len(cache_paths) != arguments.expected_cache_count:
        failures.append(
            f"cache count {len(cache_paths)} != expected {arguments.expected_cache_count}"
        )

    records: list[CacheTransformAudit] = []
    max_abs_heading = 0.0
    max_epsilon = 0.0
    max_libm_difference = 0.0
    max_libm_ratio = 0.0
    for path in cache_paths:
        cache_id = f"{path.parents[2].name}/{path.parent.name}"
        try:
            cache = _load_pickle(path)
            state = cache.ego_state.rear_axle
            pose = Pose2D(float(state.x), float(state.y), float(state.heading))
            cache_max_epsilon = 0.0
            cache_max_libm_difference = 0.0
            cache_max_ratio = 0.0
            coordinate_scale = max(1.0, abs(pose.x), abs(pose.y))
            for qx in PROBE_AXIS_M:
                for qy in PROBE_AXIS_M:
                    certificate = certify_ego_to_global((qx, qy), pose)
                    cosine = math.cos(pose.yaw)
                    sine = math.sin(pose.yaw)
                    observed_x = pose.x + cosine * qx - sine * qy
                    observed_y = pose.y + sine * qx + cosine * qy
                    observed_difference = math.hypot(
                        observed_x - certificate.x,
                        observed_y - certificate.y,
                    )
                    diagnostic_budget = certificate.error_radius + 32.0 * math.ulp(coordinate_scale)
                    ratio = observed_difference / diagnostic_budget
                    if ratio > 1.0:
                        raise ValueError(
                            "libm observation exceeds non-certificate compatibility budget"
                        )
                    cache_max_epsilon = max(cache_max_epsilon, certificate.error_radius)
                    cache_max_libm_difference = max(cache_max_libm_difference, observed_difference)
                    cache_max_ratio = max(cache_max_ratio, ratio)
            max_abs_heading = max(max_abs_heading, abs(pose.yaw))
            max_epsilon = max(max_epsilon, cache_max_epsilon)
            max_libm_difference = max(max_libm_difference, cache_max_libm_difference)
            max_libm_ratio = max(max_libm_ratio, cache_max_ratio)
            records.append(
                CacheTransformAudit(
                    cache_id=cache_id,
                    sha256=_sha256(path),
                    heading_rad=pose.yaw,
                    max_epsilon_transform_m=cache_max_epsilon,
                    max_libm_difference_m=cache_max_libm_difference,
                    max_libm_to_diagnostic_budget_ratio=cache_max_ratio,
                )
            )
        except Exception as error:
            failures.append(f"{cache_id}: {type(error).__name__}: {error}")

    cache_manifest_hash = hashlib.sha256(
        json.dumps(
            [(record.cache_id, record.sha256) for record in records],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    protocol_payload = {
        "transform_source_sha256": _sha256(transform_source),
        "probe_axis_m": PROBE_AXIS_M,
        "heading_domain_rad": (-4.0, 4.0),
        "taylor_degree": 40,
        "libm_compatibility_ulp": 32,
        "libm_used_by_certificate": False,
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "run_id": "R002F",
        "status": "completed" if not failures else "failed",
        "scope": "coordinate-transform arithmetic qualification; no performance evidence",
        "cache_count": len(cache_paths),
        "cache_pass_count": len(records),
        "cache_failure_count": len(cache_paths) - len(records),
        "probe_count_per_cache": len(PROBE_AXIS_M) ** 2,
        "protocol_sha256": protocol_hash,
        "cache_manifest_sha256": cache_manifest_hash,
        "transform_source_sha256": protocol_payload["transform_source_sha256"],
        "max_abs_heading_rad": max_abs_heading,
        "max_epsilon_transform_m": max_epsilon,
        "max_libm_difference_m": max_libm_difference,
        "max_libm_to_diagnostic_budget_ratio": max_libm_ratio,
        "libm_used_by_certificate": False,
        "failures": failures,
        "records": [asdict(record) for record in records],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "records"},
            indent=2,
        )
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
