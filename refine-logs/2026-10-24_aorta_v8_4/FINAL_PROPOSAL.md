# AORTA-v8.4 Final Proposal — Calibrated Heterogeneous Edge Envelopes

## 1. Scientific problem

For the same physical road scene and driving intent, appearance changes can degrade a
frozen planner. AORTA does not try to reconstruct an absolute safe map or declare the
native trajectory safe. It asks whether candidate-independent visual road evidence can
certify that a challenger does not reduce continuous nominal-road margin relative to the
native trajectory, and intervene only when that relative certificate is nonnegative.

The decisive observation is R004E-F: the frozen visual potential improves average edge
MAE over a zero field in all 24 development groups, yet loses on the exact group maximum
in all 24. The remaining problem is therefore not global sign inversion or complete lack
of visual signal. It is whether sparse residual tail can be localized without destroying
the simultaneous field event required by adaptive trajectory selection.

## 2. Method thesis and claim boundary

A frozen observation model outputs:

1. a candidate-independent node potential \(u(X)\)；
2. a candidate-independent positive edge scale shape \(s_e(X)\).

The session score is the maximum outward-safe normalized residual over every frozen time
and graph edge. A legal official calibration quantile turns \(s\) into heterogeneous metre
budgets \(w_e=q_\alpha s_e\). The existing exact min-plus readout then converts the single
simultaneous field event into a robust relative set-min bound for all adaptive swept-set
pairs, followed by the existing continuous geometry/map ledger.

This is a **minimal candidate mechanism**, not a proven necessary or sufficient mechanism.
Normalized conformal, Laplace NLL, difference constraints and shortest paths are standard
tools. Algorithmic novelty remains unapproved until real matched controls and an independent
novelty search are complete.

## 3. Architecture

```text
frozen DINOv3-H+ evidence and masks
                │
                ▼
      fit-only spatial trunk H(X)
          ┌─────┴─────┐
          ▼           ▼
  node potential u   right/down scale logits a
          │           │
          │           ▼
          │   bounded positive s_e(X)
          └─────┬─────┘
                ▼
 official calibration of max outward residual / scale
                ▼
 heterogeneous budgets q_alpha s_e
                ▼
 exact native-seeded min-plus transform
                ▼
 continuous geometry/map deductions
                ▼
 native-preserving relative-margin arbitration
```

- Grid: frozen 64x64 nodes and 8064 canonical right/down edges.
- Scale head: one minimal 1x1 right/down head on the frozen final trunk map; no second
  backbone, candidate encoder, ensemble or learned solver.
- Scale:

\[
s_e(X)=2^{-10}\mathrm m+(2^4-2^{-10})\mathrm m\,\sigma(a_e(X)).
\]

- Inputs: observation features and evidence masks only. Candidate, planner, certificate,
  calibration label and target-domain inputs are structurally prohibited.

## 4. Legal training

The 66 historical train groups are deterministically assigned 42 fit, 8 selection and 16
proxy-cal groups. All are explicitly historically exposed; proxy-cal is a workflow rehearsal
and cannot support formal coverage.

### Stage P

Train a fresh v8.3 potential/trunk only on the 42 fit groups. Select one checkpoint using
only the 8 selection groups, then freeze DINO, projection, trunk and potential. The old
66-group checkpoint is diagnostic only.

### Stage S

Train only the scale head on fit groups with

\[
\mathcal L_{scale}=\mathbb E\left[
\frac{|(Du-D\widetilde d)_e|}{s_e(X)}+
\log\frac{s_e(X)}{1\mathrm m}
\right].
\]

Select by the same proper loss on selection groups. No group-tail term is allowed: for a
single residual \(a\), adding \(\lambda a/s\) moves the stationary scale from \(a\) to
\((1+\lambda)a\), invalidating the earlier anti-inflation claim.

## 5. Statistical theorem

For a predictor and upper-score map frozen independently of official calibration, assume
\(Z_1,\ldots,Z_n,Z_{new}\) are jointly exchangeable conditional on all external training
and selection information. With

\[
E_i^+=\max_{t,e}z_{it,e}^+,
\quad k=\lceil(n+1)(1-\alpha)\rceil,
\quad q_\alpha=E^+_{(k)},
\]

standard ranks give

\[
P(E_{new}^+\le q_\alpha)\ge k/(n+1)\ge1-\alpha.
\]

The score upper-bounds the exact represented-value residual, so the event implies

\[
\forall t,e:\quad |(Du-D\widetilde d)_{t,e}|
\le \operatorname{up}(q_\alpha s_{t,e}).
\]

The existing arbitrary-nonnegative-weight graph theorem and continuous lifting then imply
the registered relative-margin certificate. This is a marginal source-session guarantee,
not conditional, target-OOD, collision or complete driving safety.

Any invalid official calibration group fails the entire variant; it cannot be removed or
hidden by an order statistic.

## 6. Numerical theorem interface

For edge \(i\to j\), certification never uses ordinary pre-rounded gradients. It encloses

\[
g_u^-\le u_j-u_i\le g_u^+,
\qquad
g_d^-\le\widetilde d_j-\widetilde d_i\le g_d^+,
\]

then computes

\[
a_e^+=\max\{\operatorname{up}(g_u^+-g_d^-),
\operatorname{up}(g_d^+-g_u^-),0\}.
\]

Division by scale and threshold-times-scale multiplication round upward; graph path sums
round upward and final subtraction downward. Overflow/nonfinite/out-of-range values reject.
Map and geometry errors remain separate deductions.

R004E-G qualified this interface with 19 focused tests, 131 full tests, 2000 randomized
exact arithmetic cases, a cancellation counterexample, scale-gauge case and graph oracle.

## 7. Matched mechanism controls

Every scale control shares the identical frozen potential and receives its own threshold:

1. learned observation-and-edge scale；
2. learned scale geometrically pooled within each observation and broadcast over edges；
3. uniform 1 m scale；
4. deterministic evidence-only scale；
5. zero-field historical reference, separated from same-potential attribution.

Learned over uniform supports general scale adaptation. Learned over both uniform and pooled
is required to support edge-wise spatial allocation. Neither result alone proves novelty.

## 8. Decision-space endpoint

For frozen unit \(U=(Z,t)\), define a method-independent opportunity set \(Q_B(U)\) from the
same candidate provider, sound geometry, meaningful continuous non-inclusion and teacher
improvement \(\ge\delta_M\). Let \(B(U)=1\) when this set is nonempty and

\[
C_m(U)=1\iff\exists j\in Q_B(U):L^{cert,m}_{j0}\ge\gamma_{task}.
\]

The primary endpoint is the common-denominator

\[
R^m_{cert|better}=\frac{\sum_U C_m(U)}{\sum_U B(U)}.
\]

Invalid/fallback units remain in the denominator when \(B=1\). Report group-clustered
uncertainty, candidate supply, unconditional opportunity, final selection and paired bound
sharpness separately. Development is a method-formation panel, not a new confirmation set.

## 9. Current status and stop rule

- Theory/numerical primitive gate R004E-G: PASS.
- Real learned scale: untested.
- Historical proxy coverage: not authorized.
- Official calibration: sealed.
- Gate B: not passed.
- Novelty/top-venue method claim: not authorized.

R004H must be separately registered before GPU work. The line stops if legality or soundness
fails, learned does not beat uniform on the same potential, learned does not beat pooled for
the spatial claim, the common opportunity denominator is empty, or corrected-certificate
recall remains trivial.
