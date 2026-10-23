# AORTA-Drive

[![CI](https://github.com/KUN-11-77/AORTA-Drive/actions/workflows/ci.yml/badge.svg)](https://github.com/KUN-11-77/AORTA-Drive/actions/workflows/ci.yml)

**AORTA — Affine-Operator Robust Trajectory Arbitration**

Research code for learning task-sufficient, set-valued feasibility operators for appearance-robust end-to-end driving.

> **Research status:** theory and infrastructure stage. No experimental performance claim has been established yet.

## Core idea

For a candidate trajectory panel with true feasibility margins

\[
\mathbf m = (m_1,\ldots,m_K)^\top,
\]

AORTA factorizes the task operator as

\[
\mathbf m = a\boldsymbol\rho + b\mathbf 1,
\qquad a>0,
\]

where \(\boldsymbol\rho\) describes relative feasibility structure, while \((a,b)\) determine physical scale and the absolute safety anchor. The deployed system predicts a joint operator set and changes a frozen planner only when every compatible operator agrees that the native trajectory is infeasible and an alternative is feasible.

The intended paper framing is:

> **Learning What Is Safer Before Knowing What Is Safe: Affine Task Operators for Appearance-Robust Driving**

## Repository map

```text
AORTA-Drive/
├── configs/                  # Versioned experiment and model contracts
├── docs/                     # Theory, claims, protocols, roadmap, decisions
├── experiments/              # Lightweight registry and human-readable summaries
├── paper/                    # Manuscript and figure specifications
├── references/               # Literature index; no redistributed copyrighted PDFs
├── scripts/                  # Reproducibility and repository checks
├── src/aorta_drive/          # Installable research package
├── tests/                    # Unit and contract tests
└── .github/                  # CI, issue forms, and pull-request template
```

Large datasets, checkpoints, caches, and run artifacts do not belong in Git. They must live under `NAVSIM_EXP_ROOT` or another explicitly configured artifact root.

## Quick start

Create a dedicated environment; do not modify the shared `dinov3` environment in place.

```powershell
conda env create -f environment.yml
conda activate aorta-drive
python -m pip install -e ".[dev]"
python -m pytest
python scripts/check_repo.py
```

The package currently contains only reference implementations of the mathematical factorization and native-preserving arbitration. Training and evaluation entry points will be added only after their protocols are frozen.

## Research discipline

- All claims start as hypotheses in [`docs/CLAIM_EVIDENCE_MATRIX.md`](docs/CLAIM_EVIDENCE_MATRIX.md).
- Every meaningful run receives a registered ID in [`experiments/registry.csv`](experiments/registry.csv).
- Target-domain data cannot select models, thresholds, augmentations, or checkpoints.
- Results must be traceable to a config, code revision, data manifest, seed, and artifact path.
- Failed experiments are retained and interpreted; they are not silently overwritten.
- Source-domain statistical guarantees are never presented as arbitrary-OOD guarantees.

## Current documents

- [Research brief](docs/RESEARCH_BRIEF.md)
- [Method specification](docs/METHOD_SPEC.md)
- [Claim–evidence matrix](docs/CLAIM_EVIDENCE_MATRIX.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Reproducibility contract](docs/REPRODUCIBILITY.md)
- [Roadmap](docs/ROADMAP.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)

## Citation and license

Citation metadata and an open-source license will be added after the author list and release policy are explicitly approved. Until then, this repository should be treated as pre-release research material; absence of a license does not grant reuse rights.

