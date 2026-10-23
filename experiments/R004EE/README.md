# R004E-E — Excess-residual localization

- **Question:** Why does the frozen epoch-1 visual field have larger development
  group-tail residual than the zero-difference field?
- **Config:** `configs/experiment/r004ee_excess_residual.yaml`.
- **Data:** the same frozen 24 development recording groups; calibration, test and
  target remain prohibited.
- **Experimental unit:** one recording group. Its 16 anchors and 8064 edges per
  anchor are nested diagnostic measurements, not independent replicates.
- **Frozen explanations:** predicted-gradient overshoot; group-max edge incident to
  an all-missing node; or a fully observed non-overshooting group-max edge.
- **Support rule:** an explanation is supported only when its group-level event
  occurs in at least 20 of 24 groups. Overlap between overshoot and missing evidence
  is retained. If none reaches 20, the verdict is mixed mechanism.
- **Teacher integrity check:** every teacher edge must satisfy
  `|D label| <= 1 m + endpoint error ledgers + 1e-5 m`.
- **Status:** REGISTERED. No training, parameter fitting, threshold search or
  calibration access is permitted.
- **Claim boundary:** this diagnostic selects the next falsification control; it
  cannot establish causality, architecture superiority, coverage, utility or Gate B.
