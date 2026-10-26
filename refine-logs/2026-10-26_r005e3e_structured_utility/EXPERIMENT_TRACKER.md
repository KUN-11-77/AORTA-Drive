# R005E3E Experiment Tracker

| Run ID | Purpose | Split | Priority | Status | Unlock |
|---|---|---|---|---|---|
| R005E3E0 | causal exact temporal interface and latency | 42 source-fit groups / 672 anchors | DEFERRED | HOLD_VISUAL_FIRST | retained decision-line backup; R006 visual qualification has priority |
| R005E3E1 | monolithic vs factorized training qualification | fold-0 outer-train / one seed | MUST | HOLD_E3E0 | B2 only |
| R005E3E2 | group-heldout factorized visual confirmation | 6-fold × 3-seed | MUST | HOLD_E3E1 | return to architecture question |

Forbidden shortcuts: candidate-index blacklist, same-sample threshold tuning, label-fed comfort at inference, extra network
modules before the preceding gate, development/calibration/test/NavDream/target access.
