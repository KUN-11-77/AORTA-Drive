# R004H1 Experiment Integrity Audit

**Status:** `REVIEW_UNAVAILABLE`

The required independent cross-model reviewer backend was not exposed in this session.
No executor self-review was substituted for an external verdict.

Available evidence:

- immutable raw result and checkpoint hashes exist in the external evidence manifest;
- a separate mechanical audit replayed the complete selector, zero-field gate, checkpoint
  hashes, strict state loading and resource caps;
- ground truth is the frozen dataset-derived R004E-A road-distance label, not a prediction;
- no metric is normalized by the model's own output statistics;
- the scope is one seed, 42 fit groups and 8 historically exposed selection groups.

Claim impact: only the preregistered selection-side central-signal gate may be reported as
passed. Generalization, tail control, learned uncertainty, coverage, task utility, Gate B,
OOD robustness and novelty remain unsupported.
