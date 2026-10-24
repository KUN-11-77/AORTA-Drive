# R004H2D — Read-only scale-interface conditioning diagnosis

- **Question:** is the first H2 failure primarily caused by the arbitrary magnitude of the
  frozen Stage-P spatial representation, the bounded scale parameterization, or the
  registered optimization step?
- **Sample:** the four lowest-index anchors from the first frozen fit group and the same
  from the first frozen selection group; eight anchors total. No proxy or later role opens.
- **Measurements:** raw feature statistics; exact initial bias/weight gradients; the same
  quantities after uncentered per-observation RMS normalization; and an analytic,
  non-mutating replay of AdamW's first bias-corrected step.
- **Mathematical control:** at the common initial scale of 1 m, compare the bounded-sigmoid
  derivative with a log-scale derivative. A derivative ratio below 10 cannot support
  parameterization-driven initial gradient amplification.
- **Decision:** at least 6/8 anchors must meet a predeclared cause-specific vote. Otherwise
  the diagnosis is `UNRESOLVED` and no H2 revision is authorized.
- **Prohibited:** optimizer construction, parameter mutation, checkpoint output, payload
  roles beyond fit/selection, hyperparameter search, scale fitting or metric selection.
- **Interpretation:** a resolved diagnosis may authorize registration of exactly one H2
  revision targeted at that cause. It does not establish scale quality or downstream claims.
- **Status:** COMPLETED — `TRUNK_ACTIVATION_SCALE_DOMINANT` with 8/8 votes. See
  `experiments/results/R004H2D.md`.
