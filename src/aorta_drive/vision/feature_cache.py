"""Deterministic storage and sampling primitives for frozen dense visual features."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import numpy.typing as npt


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dino_python_tree_sha256(root: Path) -> tuple[str, int]:
    """Hash the local DINO Python source tree using a path-sensitive manifest."""

    candidates = [root / "hubconf.py", *(root / "dinov3").rglob("*.py")]
    paths = sorted(path for path in candidates if path.is_file())
    if not paths or root / "hubconf.py" not in paths:
        raise ValueError("DINO source tree must contain hubconf.py and dinov3 Python sources")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{sha256_file(path)}  {relative}\n".encode())
    return digest.hexdigest(), len(paths)


def bfloat16_bits_to_float32(
    bits: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Decode raw bfloat16 bit patterns into exactly represented float32 values."""

    array = np.asarray(bits)
    if array.dtype != np.uint16:
        raise TypeError("bfloat16 storage must use uint16 bit patterns")
    words = array.astype(np.uint32) << np.uint32(16)
    return words.view(np.float32)


def write_npy_exact(path: Path, array: npt.NDArray[np.generic]) -> str:
    """Write one non-object array in deterministic NPY form and return its digest."""

    if array.dtype.hasobject:
        raise TypeError("object arrays are prohibited")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.lib.format.write_array(  # type: ignore[no-untyped-call]
            stream, np.ascontiguousarray(array), allow_pickle=False
        )
    return sha256_file(path)


def load_dense_bfloat16_bits(
    path: Path,
    *,
    expected_shape: tuple[int, int, int] = (1280, 18, 32),
) -> npt.NDArray[np.uint16]:
    """Load and validate a dense frozen DINO feature artifact."""

    with path.open("rb") as stream:
        array = np.lib.format.read_array(  # type: ignore[no-untyped-call]
            stream, allow_pickle=False
        )
    if array.dtype != np.uint16 or array.shape != expected_shape:
        raise ValueError(
            f"dense feature must have uint16 shape {expected_shape}, got "
            f"{array.dtype} {array.shape}"
        )
    decoded = bfloat16_bits_to_float32(array)
    if not np.all(np.isfinite(decoded)):
        raise ValueError("dense feature contains nonfinite bfloat16 values")
    return np.asarray(array, dtype=np.uint16)


def validate_projection_arrays(
    patch_xy: npt.ArrayLike,
    mask: npt.ArrayLike,
    *,
    node_count: int = 4096,
    slot_count: int = 12,
    patch_shape: tuple[int, int] = (18, 32),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Validate the persisted patch coordinates and explicit support mask."""

    coordinates = np.asarray(patch_xy)
    support = np.asarray(mask)
    if coordinates.dtype != np.float64 or coordinates.shape != (node_count, slot_count, 2):
        raise ValueError(
            "patch_xy must have float64 shape "
            f"({node_count}, {slot_count}, 2)"
        )
    if support.dtype != np.bool_ or support.shape != (node_count, slot_count):
        raise ValueError(f"mask must have bool shape ({node_count}, {slot_count})")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("all persisted projection coordinates must be finite")
    if np.any(coordinates[~support] != 0.0):
        raise ValueError("invalid projection coordinates must use exact zero storage")
    height, width = patch_shape
    valid = coordinates[support]
    if valid.size and (
        np.any(valid[:, 0] < 0.0)
        or np.any(valid[:, 0] > width - 1.0)
        or np.any(valid[:, 1] < 0.0)
        or np.any(valid[:, 1] > height - 1.0)
    ):
        raise ValueError("valid projection coordinates must lie in the patch-center hull")
    return np.ascontiguousarray(coordinates), np.ascontiguousarray(support)


def bilinear_sample_dense_numpy(
    dense_features: npt.ArrayLike,
    patch_xy: npt.ArrayLike,
    mask: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Materialize masked node-slot features from dense maps using bilinear sampling.

    This CPU implementation is the executable oracle for the later data-loader path.
    Dense features must have shape ``[slots, channels, height, width]``. Invalid slots
    receive exact zeros after sampling and therefore cannot leak sentinel values.
    """

    dense = np.asarray(dense_features, dtype=np.float32)
    if dense.ndim != 4:
        raise ValueError("dense_features must have shape [slots, channels, height, width]")
    slot_count, channels, height, width = dense.shape
    coordinates = np.asarray(patch_xy, dtype=np.float32)
    support = np.asarray(mask)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (slot_count, 2):
        raise ValueError("patch_xy must have shape [nodes, slots, 2]")
    if support.dtype != np.bool_ or support.shape != coordinates.shape[:2]:
        raise ValueError("mask must be boolean and match [nodes, slots]")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("patch coordinates must be finite")
    output = np.zeros((coordinates.shape[0], slot_count, channels), dtype=np.float32)
    for slot in range(slot_count):
        indices = np.flatnonzero(support[:, slot])
        if not len(indices):
            continue
        xy = coordinates[indices, slot]
        if (
            np.any(xy[:, 0] < 0.0)
            or np.any(xy[:, 0] > width - 1.0)
            or np.any(xy[:, 1] < 0.0)
            or np.any(xy[:, 1] > height - 1.0)
        ):
            raise ValueError("supported coordinates lie outside the patch-center hull")
        x0 = np.floor(xy[:, 0]).astype(np.int64)
        y0 = np.floor(xy[:, 1]).astype(np.int64)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        wx = (xy[:, 0] - x0).astype(np.float32)
        wy = (xy[:, 1] - y0).astype(np.float32)
        f00 = dense[slot, :, y0, x0]
        f10 = dense[slot, :, y0, x1]
        f01 = dense[slot, :, y1, x0]
        f11 = dense[slot, :, y1, x1]
        top = f00 + wx[:, None] * (f10 - f00)
        bottom = f01 + wx[:, None] * (f11 - f01)
        output[indices, slot] = top + wy[:, None] * (bottom - top)
    if not np.all(np.isfinite(output)):
        raise ValueError("sampled node-slot features must be finite")
    return output
