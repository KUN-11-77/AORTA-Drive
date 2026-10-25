"""Build one exact, source-fit-only visual cache for the frozen R005E2 roster."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from aorta_drive.protocol.r005e2_visual_cache import anchors_from_panel_roster
from aorta_drive.protocol.visual_recoverability import canonical_sha256
from aorta_drive.vision.feature_cache import (
    dino_python_tree_sha256,
    sha256_file,
    write_npy_exact,
)
from scripts.build_r004eb_visual_cache import (
    CHECKPOINT_SHA256,
    DINO_SOURCE_FILE_COUNT,
    DINO_SOURCE_SHA256,
    EXPECTED_DIMENSION,
    EXPECTED_PARAMETERS,
    PATCH_SHAPE,
    _plan_artifacts,
    _preprocess_image,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    if args.batch_size != 8:
        raise ValueError("R005E2 freezes DINO feature generation batch size to 8")
    if sha256_file(args.panel_identity_roster) != args.expected_roster_sha256:
        raise ValueError("frozen R005E2 panel identity roster hash mismatch")
    roster = cast(
        dict[str, Any],
        json.loads(args.panel_identity_roster.read_text(encoding="utf-8")),
    )
    anchors, roster_audit = anchors_from_panel_roster(roster)
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("frozen DINO checkpoint SHA-256 mismatch")
    source_sha256, source_file_count = dino_python_tree_sha256(args.dino_repo)
    if source_sha256 != DINO_SOURCE_SHA256 or source_file_count != DINO_SOURCE_FILE_COUNT:
        raise RuntimeError("frozen DINO source-tree contract mismatch")

    args.output_directory.mkdir(parents=True)
    started = time.perf_counter()
    records, images = _plan_artifacts(
        anchors,
        log_root=args.log_root,
        sensor_root=args.sensor_root,
        output_root=args.output_directory,
    )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise RuntimeError("R005E2 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = torch.hub.load(  # type: ignore[no-untyped-call]
        str(args.dino_repo),
        "dinov3_vith16plus",
        source="local",
        weights=str(args.checkpoint),
    )
    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError("unexpected frozen DINO parameter count")
    if int(model.embed_dim) != EXPECTED_DIMENSION or int(model.patch_size) != 16:
        raise RuntimeError("unexpected frozen DINO embedding contract")

    image_items = sorted(images.items())
    feature_hashes: dict[str, str] = {}
    batch_cardinalities: list[int] = []
    torch.cuda.reset_peak_memory_stats(device)
    for batch_start in range(0, len(image_items), args.batch_size):
        batch_items = image_items[batch_start : batch_start + args.batch_size]
        batch_cardinalities.append(len(batch_items))
        tensors = [
            _preprocess_image(
                args.sensor_root / str(record["image_relative_path"]), torch, functional
            )
            for _image_sha, record in batch_items
        ]
        batch = torch.cat(tensors, dim=0).to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            layers = model.get_intermediate_layers(
                batch,
                n=4,
                reshape=True,
                return_class_token=False,
                norm=True,
            )
            features = torch.stack(tuple(layers), dim=0).mean(dim=0)
        expected_shape = (len(batch_items), EXPECTED_DIMENSION, *PATCH_SHAPE)
        if features.dtype != torch.bfloat16 or tuple(features.shape) != expected_shape:
            raise RuntimeError("unexpected dense DINO feature shape or dtype")
        if not bool(torch.isfinite(features).all().item()):
            raise RuntimeError("DINO produced a nonfinite dense feature")
        bits = features.contiguous().view(torch.uint16).cpu().numpy()
        for index, (image_sha256, record) in enumerate(batch_items):
            feature_path = args.output_directory / str(record["feature_relative_path"])
            feature_hashes[image_sha256] = write_npy_exact(feature_path, bits[index])
        completed = min(batch_start + len(batch_items), len(image_items))
        print(f"extracted {completed}/{len(image_items)} unique image features", flush=True)

    for record in records:
        for slot in cast(list[dict[str, object]], record["slots"]):
            slot["feature_sha256"] = feature_hashes[str(slot["image_sha256"])]
    stable_payload = {
        "panel_identity_roster_sha256": args.expected_roster_sha256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dino_source_sha256": DINO_SOURCE_SHA256,
        "batch_cardinalities": batch_cardinalities,
        "records": records,
    }
    artifact_bytes = sum(path.stat().st_size for path in args.output_directory.rglob("*.npy"))
    result: dict[str, Any] = {
        "schema": "aorta.r005e2.frozen_visual_cache.v1",
        "experiment_id": "R005E2A-frozen-source-fit-visual-cache",
        "implementation_revision": args.implementation_revision,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "allowed_splits": ["train"],
        "development_calibration_test_navdream_target_accessed": False,
        "utility_candidate_scores_or_labels_accessed": False,
        "recording_group_count": roster_audit.recording_group_count,
        "anchor_count": roster_audit.anchor_count,
        "slot_count": 12,
        "image_reference_count": len(records) * 12,
        "unique_image_count": len(image_items),
        "artifact_bytes": artifact_bytes,
        "duration_seconds": time.perf_counter() - started,
        "worker_contract": {
            "device": str(device),
            "batch_size": args.batch_size,
            "batch_cardinalities": batch_cardinalities,
            "image_order": "ascending_raw_image_sha256",
            "seed": args.seed,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "cuda_device_name": torch.cuda.get_device_name(device),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "feature_contract": {
            "model": "dinov3_vith16plus",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "dino_python_source_tree_sha256": source_sha256,
            "dino_python_source_file_count": source_file_count,
            "parameter_count": parameter_count,
            "preprocessing": (
                "RGB_288x512_bicubic_align_corners_false_antialias_"
                "imagenet_normalization"
            ),
            "feature": "normalized_last_4_layer_mean",
            "storage_dtype": "uint16_bfloat16_bits",
            "shape": [EXPECTED_DIMENSION, *PATCH_SHAPE],
        },
        "projection_contract": {
            "coordinate_dtype": "float64",
            "coordinate_shape": [4096, 12, 2],
            "mask_dtype": "bool",
            "mask_shape": [4096, 12],
            "invalid_coordinate_storage": "exact_zero",
            "slot_order": "history_oldest_to_current_then_left_front_right",
        },
        "input_hashes": {
            "panel_identity_roster_sha256": args.expected_roster_sha256,
            "legacy_frozen_builder_sha256": sha256_file(
                Path(__file__).with_name("build_r004eb_visual_cache.py")
            ),
            "builder_script_sha256": sha256_file(Path(__file__)),
        },
        "logical_manifest_sha256": canonical_sha256(stable_payload),
        "records": records,
    }
    _write_json(args.output_directory / "manifest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-identity-roster", type=Path, required=True)
    parser.add_argument("--expected-roster-sha256", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261025)
    return parser.parse_args()


if __name__ == "__main__":
    output = build(parse_args())
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))
