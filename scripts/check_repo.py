"""Fail fast when required research-repository invariants are missing."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "docs/RESEARCH_BRIEF.md",
    "docs/METHOD_SPEC.md",
    "docs/CLAIM_EVIDENCE_MATRIX.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "experiments/registry.csv",
    "src/aorta_drive/__init__.py",
    "tests/test_affine_factorization.py",
)
FORBIDDEN_TRACKED_PARTS = {"data", "datasets", "outputs", "checkpoints", "wandb"}
MAX_SOURCE_BYTES = 5 * 1024 * 1024


def main() -> int:
    errors: list[str] = []
    for relative_path in REQUIRED:
        if not (ROOT / relative_path).is_file():
            errors.append(f"missing required file: {relative_path}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in FORBIDDEN_TRACKED_PARTS:
            errors.append(f"artifact directory contains a file: {relative}")
        if path.stat().st_size > MAX_SOURCE_BYTES:
            errors.append(f"oversized repository file (>5 MiB): {relative}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Repository structure check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
