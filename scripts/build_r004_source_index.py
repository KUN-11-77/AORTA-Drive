"""Build a deterministic token-to-recording index from NAVSIM source logs.

This script intentionally reproduces only the frozen ``SceneFilter`` semantics
needed by R004.  It does not load sensors, labels, maps, model predictions, or
scores.  Run it where the official NAVSIM log pickles are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

_LOG_PATTERN = re.compile(r"^(?P<group>.*_veh-\d+)_(?P<start>\d+)_(?P<end>\d+)$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_scalar(lines: list[str], key: str) -> str:
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise ValueError(f"missing scene-filter key: {key}")


def _parse_quoted_list(lines: list[str], key: str) -> list[str]:
    header = f"{key}:"
    try:
        start = next(index for index, line in enumerate(lines) if line == header) + 1
    except StopIteration as error:
        raise ValueError(f"missing scene-filter list: {key}") from error
    values: list[str] = []
    for line in lines[start:]:
        match = re.fullmatch(r"\s+-\s+'([^']+)'\s*", line)
        if match is None:
            if line.strip():
                break
            continue
        values.append(match.group(1))
    if not values:
        raise ValueError(f"empty scene-filter list: {key}")
    return values


def parse_scene_filter(path: Path) -> dict[str, Any]:
    """Parse the frozen subset of a NAVSIM SceneFilter YAML file."""

    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        "num_history_frames": int(_parse_scalar(lines, "num_history_frames")),
        "num_future_frames": int(_parse_scalar(lines, "num_future_frames")),
        "frame_interval": int(_parse_scalar(lines, "frame_interval")),
        "has_route": _parse_scalar(lines, "has_route").lower() == "true",
        "log_names": _parse_quoted_list(lines, "log_names"),
        "tokens": _parse_quoted_list(lines, "tokens"),
    }


def _recording_group(log_name: str) -> tuple[str, int, int]:
    match = _LOG_PATTERN.fullmatch(log_name)
    if match is None:
        raise ValueError(f"invalid NAVSIM log name: {log_name}")
    return match.group("group"), int(match.group("start")), int(match.group("end"))


def _eligible_tokens(
    frames: list[dict[str, Any]],
    *,
    history: int,
    future: int,
    interval: int,
    has_route: bool,
    allowed_tokens: set[str],
) -> list[str]:
    num_frames = history + future
    result: list[str] = []
    for start in range(0, len(frames), interval):
        window = frames[start : start + num_frames]
        if len(window) < num_frames:
            continue
        current = window[history - 1]
        if has_route and not current["roadblock_ids"]:
            continue
        token = str(current["token"])
        if token in allowed_tokens:
            result.append(token)
    return result


def build_index(config_path: Path, log_root: Path) -> dict[str, Any]:
    """Return a result-independent source token index and its provenance."""

    config = parse_scene_filter(config_path)
    official_logs = config.pop("log_names")
    official_tokens = config.pop("tokens")
    if len(official_logs) != len(set(official_logs)):
        raise ValueError("scene filter contains duplicate log names")
    if len(official_tokens) != len(set(official_tokens)):
        raise ValueError("scene filter contains duplicate tokens")
    token_set = set(official_tokens)

    records: list[dict[str, Any]] = []
    observed_tokens: list[str] = []
    for log_name in sorted(official_logs):
        path = log_root / f"{log_name}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            frames = pickle.load(stream)  # noqa: S301 - official frozen NAVSIM input
        if not isinstance(frames, list):
            raise TypeError(f"expected frame list in {path}")
        tokens = _eligible_tokens(
            frames,
            history=config["num_history_frames"],
            future=config["num_future_frames"],
            interval=config["frame_interval"],
            has_route=config["has_route"],
            allowed_tokens=token_set,
        )
        group, fragment_start, fragment_end = _recording_group(log_name)
        records.append(
            {
                "log_name": log_name,
                "recording_group": group,
                "fragment_start": fragment_start,
                "fragment_end": fragment_end,
                "scene_count": len(tokens),
                "token_sha256": _canonical_sha256(tokens),
                "tokens": tokens,
            }
        )
        observed_tokens.extend(tokens)

    if len(observed_tokens) != len(set(observed_tokens)):
        raise ValueError("eligible token occurs in more than one source log")
    if set(observed_tokens) != token_set:
        missing = sorted(token_set - set(observed_tokens))[:10]
        extra = sorted(set(observed_tokens) - token_set)[:10]
        raise ValueError(f"source token mismatch: missing={missing}, extra={extra}")

    logical_rows = [
        {
            "log_name": record["log_name"],
            "recording_group": record["recording_group"],
            "scene_count": record["scene_count"],
            "token_sha256": record["token_sha256"],
        }
        for record in records
    ]
    return {
        "schema_version": 1,
        "protocol": "official navtrain SceneFilter semantics; metadata only",
        "source_config_sha256": _sha256_file(config_path),
        "scene_filter": config,
        "official_log_count": len(official_logs),
        "official_token_count": len(official_tokens),
        "logical_sha256": _canonical_sha256(logical_rows),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-filter", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_index(args.scene_filter, args.log_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}))


if __name__ == "__main__":
    main()
