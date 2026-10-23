# Experiment Registry

`registry.csv` is the append-only human-readable index. Register a run before using research-scale compute. Store large outputs under `NAVSIM_EXP_ROOT`, and place a concise result card under `experiments/results/<experiment_id>.md` when complete.

Allowed statuses: `planned`, `running`, `completed`, `failed`, `stopped`, `invalid`.

The initial `T0-operator-qualification` entry describes repository unit qualification only; it is not a paper experiment and supports no empirical claim.
