# R004H2R1 — RMS-gauge-repaired scale-only fit

- **Question:** does removing only the diagnosed positive activation-magnitude gauge make
  the bounded 194-parameter scale head fit a better proper score than uniform 1 m?
- **Only change from R004H2:** the detached scale-branch input is divided by its uncentered
  per-observation global RMS. The potential path and all learned parameter counts are
  unchanged.
- **Frozen unchanged:** H1 checkpoint, 42 fit/8 selection roles, seed, schedule, AdamW
  `3e-4`, ten epochs, bounded-Laplace objective, selector, uniform baseline and cap gate.
- **Go/no-go:** the selected macro group-mean proper loss must be strictly below the same
  uniform 1 m primary and neither cap-neighborhood fraction may reach 0.95.
- **Prohibited:** centering, trainable normalization, learning-rate tuning, new capacity,
  proxy/development/calibration/test/target payloads and any H2R2 retry.
- **Interpretation:** PASS would establish only that the targeted interface repair rescues
  selection-side scale fitting and would authorize H3 registration. It would not establish
  calibration, coverage, task utility, Gate B, OOD robustness or novelty.
- **Status:** REGISTERED; not yet executed.
