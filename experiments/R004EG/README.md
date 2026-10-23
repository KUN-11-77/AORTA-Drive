# R004E-G — Heterogeneous edge-envelope oracle

- **Question:** Is the revised observation-and-edge scale interface mathematically and
  numerically safe enough to justify registering a later real training probe?
- **Scope:** CPU-only synthetic arithmetic, exact graph oracle, and metadata-only role
  assignment. No feature, label, development, official calibration, test, or target-domain
  payload is read.
- **Config:** `configs/experiment/r004eg_heterogeneous_envelope.yaml`.
- **Independent review:** R004E-G was declared READY at v8.4 review Round 3; the paper
  method remained REVISE.
- **Must pass:** deterministic disjoint 42/8/16 roles; four-term residual enclosure;
  outward division/multiplication; invalid-group fail-closed semantics; joint-rank oracle;
  tail-loss counterexample; interior scale gauge; heterogeneous graph readout.
- **Interpretation:** passing qualifies primitives and the protocol only. It does not prove
  learned uncertainty, historical proxy coverage, Gate B, real certificate utility, OOD
  robustness, or algorithmic novelty.
- **Status:** REGISTERED, not yet executed.
