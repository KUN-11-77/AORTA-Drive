"""Execute the pre-registered Gate A CPU qualification runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "experiments" / "results"
RUNS = {
    "R001": ROOT / "tests" / "gate_a" / "test_r001_graph_oracle.py",
    "R002": ROOT / "tests" / "gate_a" / "test_r002_continuous_lifting.py",
    "R002B": ROOT / "tests" / "gate_a" / "test_r002b_legacy_geometry_counterexamples.py",
    "R002C": ROOT / "tests" / "gate_a" / "test_r002c_swept_outer_cover.py",
    "R003": ROOT / "tests" / "gate_a" / "test_r003_calibration_and_risk.py",
}


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _implementation_tree_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    relevant = [
        line
        for line in result.stdout.splitlines()
        if "experiments/results/" not in line.replace("\\", "/")
    ]
    return bool(relevant)


def _run(run_id: str, test_path: Path) -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", "-q", str(test_path.relative_to(ROOT))]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    duration = time.perf_counter() - started
    return {
        "run_id": run_id,
        "status": "completed" if result.returncode == 0 else "failed",
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_revision": _git_revision(),
        "implementation_tree_dirty": _implementation_tree_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "command": command,
        "duration_seconds": round(duration, 6),
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _write_result(record: dict[str, object]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = str(record["run_id"])
    json_path = RESULT_ROOT / f"{run_id}.json"
    markdown_path = RESULT_ROOT / f"{run_id}.md"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    status = record["status"]
    timestamp = record["timestamp_utc"]
    revision = record["git_revision"]
    dirty = record["implementation_tree_dirty"]
    duration = record["duration_seconds"]
    return_code = record["return_code"]
    output = str(record["stdout"]).strip()
    markdown = f"""# {run_id} Gate A Result

- **Status:** {status}
- **Timestamp (UTC):** {timestamp}
- **Git revision:** `{revision}`
- **Dirty implementation tree:** `{dirty}`
- **Duration:** {duration} seconds
- **Return code:** {return_code}

## Scope

This is a synthetic CPU qualification run. It is not NAVSIM performance evidence,
an OOD guarantee, or a paper-level method result.

## Test output

```text
{output}
```
"""
    markdown_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", choices=sorted(RUNS))
    arguments = parser.parse_args()
    failed = False
    selected_runs = arguments.runs or sorted(RUNS)
    for run_id in selected_runs:
        record = _run(run_id, RUNS[run_id])
        _write_result(record)
        print(f"{run_id}: {record['status']} ({record['duration_seconds']}s)")
        failed |= record["status"] != "completed"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
