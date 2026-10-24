# R004H3 — Historically exposed proxy-calibration rehearsal

- **Question:** can the fully frozen H2R1 model and four registered controls execute the
  heterogeneous session-level calibration pipeline without invalid groups, optimistic
  arithmetic, role leakage or nondeterministic thresholds?
- **Data:** exactly 16 historically exposed proxy groups × 16 anchors, read once. These are
  not fresh calibration sessions and cannot support coverage claims.
- **Controls:** learned, per-observation pooled, uniform 1 m and evidence-only. Every control
  receives its own threshold; sharing the learned threshold is prohibited.
- **Score:** outward-safe four-node edge residual divided outward by scale, then the maximum
  over all 16 times and 8064 edges in a recording group.
- **Finite-sample rule:** `alpha=0.10`, `n=16`, rank `ceil(17×0.9)=16`; each threshold is
  therefore that control's maximum complete-session score.
- **Invalid rule:** one invalid group makes that entire control variant infinite; no group
  may be filtered. H4 registration requires all four variants to remain finite.
- **Prohibited:** fitting, parameter updates, checkpoint writes, shared thresholds,
  development/official-calibration/test/target reads, and any coverage interpretation.
- **Status:** REGISTERED; not yet executed.
