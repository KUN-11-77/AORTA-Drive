# R005E3E Experiment Plan

> 2026-10-26 status：`HOLD_VISUAL_FIRST__NOT_EXECUTED`。本计划保留为决策线备选；当前不与 R006 视觉机制资格并行执行。

**Problem:** monolithic proper-mean scorer learns average relation but rare binary temporal-gate errors dominate decision gain.

**Method thesis:** compute causally known candidate-history dynamics explicitly and use proper component heads only for unknown scene outcomes before native-relative selection.

## Claim Map

| Claim | Minimum convincing evidence | Block |
|---|---|---|
| Supporting: temporal compatibility is a legitimate inference input | outcome-blind exact replay, causal provenance, deterministic execution, bounded latency | B0 |
| Primary G1 qualification: structured utility learning fixes the observed tail without relying on larger capacity | matched training-domain positive raw gain and capture; then group-heldout confirmation | B1, B2 |
| Anti-claim: gains are not label leakage, extra parameters or all-abstain behavior | two-stage audit, parameter/FLOP match, intervention counts and harm ledger | B0-B2 |

## Paper Storyline

- Main paper may use this only as the principled base scorer beneath the later visual contribution.
- Appendix should contain exact target algebra, interface audit, full loss and latency scaling.
- Cut: loss zoo, candidate-index blacklist, post-hoc threshold search, extra uncertainty modules before OOF gain.

## Experiment Blocks

### B0 — Causal Temporal Interface

- Dataset/split: 42 source-fit groups, all 672 anchors; zero training.
- Systems: outcome-blind causal operator; independent E1B ledger audit.
- Metrics: candidate-level exactness, determinism, nonfinite count, P50/P95/P99/max latency, K-scaling.
- Success: all exact and operator P95 ≤8.88ms.
- Failure: leakage/unavailable state/exactness => STOP; latency-only => HOLD for implementation optimization.
- Priority: MUST-RUN.

### B1 — Matched Training-Domain Factorization Qualification

- Dataset/split: fold-0 outer-train, seed 20261025 only.
- Systems: frozen E3D monolithic mean vs two-head `F/H + analytic C`; matched backbone, init, order, steps,
  trainable parameters and FLOPs.
- Metrics: raw gain, absolute capture, beneficial/harmful/neutral choices, relation MSE; component MSE/BCE only diagnostic.
- Success: positive raw gain, capture ≥10%, nonzero beneficial intervention, exact replay; no criterion may be replaced by
  relative improvement over a harmful parent.
- Failure: no positive gain => factorization unsupported; no architecture expansion.
- Priority: MUST-RUN only after B0 GO.

### B2 — Group-Heldout Visual Confirmation

- Dataset/split: frozen 6-fold×3-seed source groups.
- Systems: monolithic vs factorized; each with correct/mismatch/removed input interventions.
- Metrics: group-equal gain/regret, component calibration, scene-binding intervention, paired bootstrap CI.
- Success: factorized-correct has positive gain and reliably beats its controls and monolithic scorer.
- Failure: training-only gain => overfit; correct≈controls => visual mechanism unsupported.
- Priority: MUST-RUN only after B1 PASS.

## Run Order

| Milestone | Goal | Decision gate | Cost |
|---|---|---|---|
| M0 | B0 exactness/latency | GO/HOLD/STOP | CPU, expected <1h plus replay |
| M1 | B1 single fold/seed | PASS/FAIL | about one E3D primary+replay |
| M2 | B2 6-fold×3-seed | GO/HOLD/STOP | only estimated after B1 profiler |

## Risks and Mitigations

- Evaluator leakage: materializer has no target argument; label comparison lives in a separate process.
- Benchmark overfitting: expose a general `known dynamics + unknown scene outcome` interface and later test actual-history.
- Added capacity: match trainable count/FLOPs or add frozen/dummy control head.
- Mean gain hides tails: always report beneficial/harmful counts, gain sums, worst groups and maximum harm.
- Latency: batch candidates through one simulator call; cache previous realized state once per control cycle.

## Final Checklist

- [x] Main bottleneck and anti-claim are explicit.
- [x] Zero-training legitimacy gate precedes learning.
- [x] Simplicity is defended with a matched monolithic control.
- [x] OOF confirmation is separated from training qualification.
- [x] DQ/NavDream remain locked.
