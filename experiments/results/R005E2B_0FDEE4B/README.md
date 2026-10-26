# R005E2B compact result bundle

This directory is the Git-safe compact copy of the completed formal run
`/data/khsong/aorta_runs/r005e2b_science_0fdee4b_multislot_v2/`.

Included:

- frozen `audit.json`, `progress.json`, run/GPU contracts and full-run `SHA256SUMS.txt`;
- all 54 primary and 54 replay `result.json` files;
- all OOF `predictions.npz`, including the preregistered V2 input interventions;
- the registered post-hoc `failure_anatomy.json`.

Excluded from Git:

- 108 model checkpoints (about 2.0 GiB total);
- runtime logs.

The complete remote directory had 329 non-log files and passed `sha256sum -c`. The compact copy retains
221 of those hashed files; every retained file was checked against the full manifest, and all primary/replay
prediction arrays were independently compared bit-for-bit. Checkpoint state equality remains supported by the
frozen remote auditor and the per-job state hashes, not by a second local checkpoint load. The external target
bundle used for local diagnostics is stored outside Git at
`D:/AORTA-Drive-artifacts/R005E2B_0FDEE4B/bundle_primary/`.

Scientific interpretation is in `experiments/R005/R005E2B_RESULTS.md`; integrity limitations are in
`experiments/R005/R005E2B_EXPERIMENT_AUDIT.md`.
