# R004H1 — Fresh fit-only Stage P potential

- **Question:** does a freshly initialized potential/trunk retain transferable central
  visual signal after the historical 66 train groups are separated into fit, selection and
  proxy roles?
- **Optimization:** exactly 42 fit groups × 16 anchors, once per epoch for 20 epochs, using
  the unchanged Stage-P loss and optimizer schedule.
- **Selection:** exactly 8 selection groups. The primary key is macro group-mean edge MAE;
  the secondary key is the mean of the two largest exact group maxima; final tie-break is
  earliest epoch.
- **Go/no-go:** H2 may be registered only if the selected primary is strictly below the
  zero-field primary computed on the same selection labels.
- **Prohibited:** loading the old epoch-1 checkpoint; proxy-cal/development/official
  calibration/test/target payloads; candidate, planner, certificate or scale metrics;
  hyperparameter search and metric-based early stopping.
- **Interpretation:** a PASS supports only the existence of selection-side central visual
  signal and authorizes R004H2 registration. It is not uncertainty, coverage, utility,
  Gate B or novelty evidence.
- **Status:** REGISTERED at implementation revision
  `634f990c7d8cf9598054237a70119c7b0c0773b4`; not yet executed.
