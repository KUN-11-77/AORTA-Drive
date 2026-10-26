# R005E3A compact result bundle

Git-safe compact copy of `/data/khsong/aorta_runs/r005e3a_77f37e4` plus the separately registered read-only
collapse diagnosis.

Included:

- formal `audit.json`, logs, progress/run/GPU contracts, exit code and full-run SHA-256 manifest;
- all 18+18 primary/replay `result.json`, `predictions.npz` and job logs;
- compact-bundle SHA-256 manifest;
- post-hoc outer-train diagnosis JSON, log and exact diagnostic source used remotely.

Excluded:

- 36 checkpoints (about 656 MiB total), retained in the immutable remote run. Their primary/replay equality is
  supported by the frozen auditor and per-job hashes.

All 119 retained files match `SHA256SUMS_COMPACT.txt`. Interpretation is in
`experiments/R005/R005E3A_RESULTS.md`; integrity limits are in
`experiments/R005/R005E3A_EXPERIMENT_AUDIT.md`.
