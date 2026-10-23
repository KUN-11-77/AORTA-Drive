# Contributing

## Before changing code

1. Read `docs/RESEARCH_BRIEF.md`, `docs/METHOD_SPEC.md`, and `AGENTS.md`.
2. Link the change to a registered hypothesis, bug, or experiment.
3. For a meaningful experiment, reserve a run ID in `experiments/registry.csv` before execution.

## Development workflow

```powershell
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest
python scripts/check_repo.py
```

Pull requests must state:

- the scientific or engineering question;
- the exact change and non-goals;
- tests and validation performed;
- claim/evidence impact;
- reproducibility and data-provenance impact.

Do not commit generated datasets, checkpoints, caches, target-domain labels, or credentials.

