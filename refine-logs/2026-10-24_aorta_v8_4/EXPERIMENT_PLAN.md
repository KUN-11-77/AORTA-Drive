# AORTA-v8.4 Claim-Driven Experiment Plan

## Current gate position

R004E-G and R004H0 passed. R004H1 then completed and passed its selection-side zero-field
central-signal gate. The next executable block is R004H2, but **no scale optimization is
authorized until its exact proper-loss configuration, selected H1 checkpoint, cap-collapse
rule, controls and compute budget are committed and pinned**. Proxy-cal, development,
official calibration, test and target remain sealed.

## Claim map

| ID | Claim candidate | Minimum evidence | Falsifier |
|---|---|---|---|
| H1 | A fresh fit-only potential retains transferable central visual signal | selection and one-read development group-mean edge MAE beats zero field | fresh potential does not beat zero, or legality/replay fails |
| H2 | observation-dependent scale allocation adds task value on the same potential | learned scale strictly beats uniform on common-denominator corrected certificate recall | learned <= uniform or gain relies on invalid filtering |
| H3 | edge-wise localization, not only observation-wide amplitude, causes the gain | learned strictly beats independently recalibrated pooled scale | learned <= pooled |
| H4 | the interface is practically nontrivial | opportunity denominator nonzero and recall exceeds preregistered minimum | no opportunities or near-total abstention |

No R004H outcome can establish official coverage or novelty because proxy-cal and development
are historically exposed.

## Frozen architecture candidate

- Stage P: exact `QuotientFieldPredictor`, initialized from registered seed, trained only on
  42 fit groups.
- Refactor requirement: expose the final 96-channel decoded map without changing the
  potential output. A bitwise or registered-tolerance replay against the pre-refactor model
  is a MUST-RUN preflight.
- Stage S: one `Conv2d(96, 2, kernel_size=1)` scale head, zero weights and a frozen bias that
  represents 1 m at initialization; 194 trainable parameters.
- Edge order: all rightward entries followed by all downward entries, exactly matching
  `oriented_grid_gradients`.
- Scale bounds: `[2^-10, 2^4] m` with bounded sigmoid.
- Stage S freezes every Stage P parameter.

## Sequential blocks

### H0 — protocol and implementation preflight

- **Data:** IDs and one existing fit anchor only; no proxy/development/calibration payload.
- **Checks:** persisted 42/8/16 role manifest and hashes; no overlaps; historical exposure;
  trunk-refactor potential replay; 8064 scale order; exact 1 m initialization; candidate
  independence; finite forward/backward; only scale-head parameters receive gradients.
- **Pass:** every structural check passes and no prohibited split opens.
- **Fail:** repair implementation and register a new revision; do not train.
- **Cost:** <30 min, one RTX 3090.

### H1 — fresh Stage P potential replay

- **Fit data:** 42 groups x 16 anchors.
- **Selection data:** 8 groups x 16 anchors after every epoch.
- **Objective:** unchanged v8.3 potential Huber plus its Stage-P smooth-tail term.
- **Schedule:** one registered seed, at most 20 epochs, same AdamW/lr/warmup/cosine/clip as
  R004E-C2, with anchor schedule regenerated only from fit group IDs.
- **Checkpoint key:** macro mean of per-group mean edge MAE; secondary mean of the two worst
  exact group maxima; tertiary earliest epoch. This key must be frozen before execution.
- **Completion:** all epochs/inputs legal and finite and exactly one checkpoint is selected.
- **Pass to H2:** the selected checkpoint's macro mean of eight selection-group mean edge
  MAEs is strictly below the zero-field value on the same selection labels. Otherwise H1
  records `STOP` and no scale fitting is permitted. H1 never inspects proxy or development.
- **Cost estimate:** 42/66 of the previous C2 run, measured wall-time used to update the
  remaining budget after epoch 1.

### H2 — scale-head fit and matched controls

- **Fit/selection:** same roles as H1; Stage P frozen.
- **Loss:** bounded Laplace proper loss only.
- **Proposed schedule to freeze:** one seed, 10 epochs, AdamW, batch 1, fit anchor once per
  epoch, base lr `3e-4`, betas `(0.9,0.95)`, weight decay `1e-4`, clip `1.0`, one-epoch
  warmup and cosine decay; no augmentation or search.
- **Checkpoint key:** selection macro mean proper loss, then earliest epoch.
- **Controls generated from the final frozen Stage P/S pair:** learned, geometric pooled,
  uniform 1 m, deterministic evidence-only, zero-field reference.
- **Pass:** one legal frozen scale checkpoint and all controls replay deterministically.
- **Fail:** nonfinite/cap collapse/order mismatch/gradient leakage stops before proxy read.

### H3 — proxy-cal pipeline rehearsal

- **Data:** exactly the 16 registered proxy groups, read once after H1/H2 freeze.
- **Score:** complete session maximum of the R004E-G outward-safe score.
- **Threshold:** `alpha=0.1`, rank 16 maximum; one threshold per control.
- **Invalid:** any invalid group fails that entire variant and records infinite threshold.
- **Interpretation:** protocol rehearsal only, no empirical coverage claim.
- **Fail:** no development read for the failed variant.

### H4 — one-read development qualification

- **Data:** all 24 groups x 16 frozen times, exactly once after every choice above is frozen.
- **Primary endpoint:** common-denominator continuous-corrected
  `R_cert|better` on method-independent teacher opportunities.
- **Controls:** learned vs pooled vs uniform on identical potential; evidence-only and zero
  as secondary references.
- **Independent unit:** recording group. Report group numerators/denominators and paired
  cluster bootstrap; anchors are nested.
- **Invalid/fallback:** remains in the fixed denominator and counts not certified.
- **Decision hierarchy:** legality/soundness > existence of opportunities > learned vs
  uniform > learned vs pooled > practical magnitude.

## Items that must be frozen before H0 registration

1. Exact role CSV and its logical SHA-256.
2. Refactored trunk API and scale-head source-tree hash.
3. Stage-P checkpoint key and Stage-S schedule.
4. Deterministic evidence-only scale formula.
5. Teacher continuous-margin computation and its numerical ledger.
6. Meaningful non-inclusion definition and threshold.
7. `delta_M`, `gamma_task` (currently `0 m` in R004A), practical recall difference and
   bootstrap seed/tie rule.
8. Zero-opportunity rule: qualification fails as “no candidate opportunity”; recall is not
   reported as zero or one.
9. Exact artifact paths, resume policy, wall-time/disk/GPU-memory limits.

## Ablation discipline

MUST-RUN only:

- learned vs pooled vs uniform scale on same potential；
- evidence-only scale；
- zero field historical reference；
- cap saturation and invalid-reason diagnostics.

Do not add joint training, tail loss, ensembles, MC dropout, alternative backbones, VLM/RL,
target adaptation or learned candidates to rescue a failed R004H.

## Compute budget and stopping

- H0: <0.5 GPU-hour.
- H1: upper bound 20 epochs over 672 anchors/epoch; stop only for registered illegality or
  resource failure, not poor interim metrics.
- H2: upper bound 10 epochs over 672 anchors/epoch.
- H3/H4: inference-only; exact wall time measured in preflight before the one-read H4 run.
- Only one registered seed until H4 qualifies the mechanism. Additional seeds are forbidden
  as a rescue strategy and considered only after a positive predeclared result.

## Theory update rules

- H0 arithmetic/ordering failure: implementation repair; theory claim remains untested.
- H1 central signal failure: reject the fresh potential route; do not train scale.
- H2 fit failure: reject the minimal head or optimization implementation; no proxy read.
- H3 infinite/invalid threshold: reject the affected variant; do not filter groups.
- H4 learned <= uniform: reject scale-adaptation claim.
- H4 learned > uniform but <= pooled: retain only observation-level scale adaptation;
  reject spatial-localization claim.
- H4 learned > pooled but recall practically trivial: theory may be valid but system lacks
  decision value; Gate B remains failed.
- Positive development qualification: freeze the final model and only then consider the
  official calibration protocol; it still does not authorize novelty.
