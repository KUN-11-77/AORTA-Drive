# R005E1B frozen result bundle

Run directory: `/data/khsong/aorta_runs/r005e1b_1e5362e`

Frozen source commit: `1e5362ef6d65dd7e568466317ed347aef312fa75`

## Contents

- `primary/result.json`: complete 672-anchor primary materialization;
- `replay/result.json`: independent complete replay;
- `audit.json`: mechanical replay/recomputation verdict;
- `SHA256SUMS.txt`: hashes generated on the server before transfer;
- `inputs/`: frozen E1A qualification, current/previous cache manifests, repaired context roster and repaired B4R1 result;
- `official_source_snapshot.tar.gz` and `official_source/`: frozen NAVSIM evaluator-source snapshot used for post-run source-alignment review;
- `*.log`, `progress.json`, `run.pid`, `exit_code.txt`, `CODE_COMMIT`, `SOURCE_ROOT`: execution provenance.

The copied `progress.json` files are the final append-only progress snapshots and retain
`IN_PROGRESS` even after reaching `672/672`. Terminal state is determined by
`result.json.status=COMPLETED`, `audit.json.status=PASS`, and `exit_code.txt=0`.

## Verified hashes

The local copies of all five frozen input artifacts match the SHA-256 values embedded in
`configs/experiment/r005e1b_contextual_oracle.yaml`. The server-side result hashes in
`SHA256SUMS.txt` were checked successfully before transfer.

Scientific interpretation and claim limits are recorded in
`experiments/R005/R005E1B_RESULTS.md`.

The independent post-run integrity review is intentionally `WARN`, not unconditional
`PASS`: EC was not recomputed from raw state arrays, runtime import provenance is not
fully reconstructable from this bundle, and the reviewer backend was not cross-family.
