# Experiment Protocol

## 1. Gates

Experiments progress in order. A later gate cannot rescue a failed earlier claim.

1. **T0 — Algebra and implementation:** factorization reconstruction, degeneracy, equivariance, interval composition.
2. **T1 — Synthetic identifiability:** construct indistinguishable observations with different anchors; verify point-estimator failure and set behavior.
3. **T2 — GT-geometry panel study:** test whether relative shape is more stable under controlled margin perturbations.
4. **T3 — Source-session calibration:** calibrate on source sessions and test joint coverage on disjoint source sessions.
5. **T4 — Appearance OOD operator test:** freeze all choices; evaluate relative error, coverage, and sharpness.
6. **T5 — Closed-loop decision test:** enable native-preserving arbitration and measure intervention utility.
7. **T6 — Full benchmark comparison:** only after C1–C4 survive.

## 2. Data partition policy

- Split by scene/session before frame sampling.
- Keep calibration and evaluation sessions disjoint.
- Freeze target-domain data until all source-domain choices are committed.
- Record dataset version, scene IDs, filters, and code commit in every run.

## 3. Primary metrics

- relative-shape weighted error and rank agreement;
- joint coverage and set sharpness for \((\log a,b)\);
- simultaneous margin-panel coverage;
- native false intervention rate;
- certified-native-failure recall;
- replacement success conditional on intervention;
- downstream NAVSIM metrics with uncertainty intervals.

## 4. Statistics

- The unit of resampling is the independent scene/session, not the frame.
- Report effect sizes and scene-clustered intervals, not only point estimates.
- Aggregate across predeclared seeds; retain every seed.
- A calibration result is incomplete without sharpness.
- Multiple primary comparisons require a predeclared correction or hierarchical testing order.

## 5. Run registration

Before execution, add one row to `experiments/registry.csv` and commit:

- immutable experiment ID;
- exact claim ID;
- hypothesis and primary endpoint;
- data split and intervention;
- config path, seed set, compute budget;
- success and stop criteria.

After execution, append status and artifact location without rewriting the original hypothesis.

## 6. Integrity rules

- Never report a dry run as an experiment.
- Never infer unavailable ground truth from the model being evaluated.
- Never normalize scores with target test statistics.
- Failed and interrupted runs remain in the registry.
- Figures and tables must be reproducible from stored machine-readable results.
