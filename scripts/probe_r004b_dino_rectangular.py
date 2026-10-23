"""Qualify the frozen DINOv3-H+ rectangular dense-token contract for R004B."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

EXPECTED_CHECKPOINT_SHA256 = "7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488"
EXPECTED_DIMENSION = 1280
EXPECTED_PARAMETERS = 840_633_600


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preprocess(path: Path, height: int, width: int) -> tuple[torch.Tensor, list[int]]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    raw_shape = [int(rgb.shape[0]), int(rgb.shape[1])]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = functional.interpolate(
        tensor,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (tensor - mean) / std, raw_shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--last-n", type=int, default=4)
    args = parser.parse_args()

    if args.height % 16 or args.width % 16:
        raise ValueError("input dimensions must be divisible by the 16-pixel patch size")
    checkpoint_hash = _sha256(args.checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"unexpected DINOv3-H+ checkpoint SHA-256: {checkpoint_hash}")

    device = torch.device(args.device)
    tensor, raw_shape = _preprocess(args.image, args.height, args.width)
    model = torch.hub.load(
        str(args.dino_repo),
        "dinov3_vith16plus",
        source="local",
        weights=str(args.checkpoint),
    )
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"unexpected parameter count: {parameter_count}")
    if int(model.embed_dim) != EXPECTED_DIMENSION or int(model.patch_size) != 16:
        raise RuntimeError(
            f"unexpected model contract: dim={model.embed_dim}, patch={model.patch_size}"
        )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        layers = model.get_intermediate_layers(
            tensor.to(device=device, dtype=torch.bfloat16),
            n=args.last_n,
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        feature = torch.stack(tuple(layers), dim=0).mean(dim=0)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    expected_shape = (1, EXPECTED_DIMENSION, args.height // 16, args.width // 16)
    layer_shapes = [list(layer.shape) for layer in layers]
    finite = bool(torch.isfinite(feature).all().item())
    if tuple(feature.shape) != expected_shape or not finite:
        raise RuntimeError(
            f"invalid dense feature: shape={tuple(feature.shape)}, finite={finite}"
        )
    result = {
        "experiment_id": "R004B-dino-rectangular-contract",
        "implementation_revision": args.implementation_revision,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_script_sha256": _sha256(Path(__file__)),
        "image": str(args.image),
        "image_sha256": _sha256(args.image),
        "raw_shape": raw_shape,
        "preprocessing": {
            "color": "RGB",
            "resize_shape": [args.height, args.width],
            "resize_mode": "torch_bicubic_align_corners_false_antialias_true",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "crop": False,
        },
        "model": "dinov3_vith16plus",
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "checkpoint_sha256": checkpoint_hash,
        "parameter_count": parameter_count,
        "embed_dimension": int(model.embed_dim),
        "patch_size": int(model.patch_size),
        "last_n_normalized_layer_mean": args.last_n,
        "layer_shapes": layer_shapes,
        "feature_shape": list(feature.shape),
        "finite": finite,
        "inference_dtype": str(feature.dtype),
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "training_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
