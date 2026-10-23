# AORTA-Drive Method Specification v0.1

## 1. Objects and task quantity

Let (S) denote the physical scene, (X\sim p(X\mid S,A)) an image under appearance (A), and \(\tau\) a candidate trajectory. Its swept footprint is \(\mathcal B(\tau)\). Given a signed scene-distance field \(d_S\), define the task margin

\[
M_S(\tau)=\inf_{x\in\mathcal B(\tau)} d_S(x).
\]

Positive margin means the swept trajectory remains within the modeled feasible region; negative margin indicates violation. This definition is only as complete as the modeled constraint field and must not be described as universal safety.

For a probe distribution \(\nu\) over trajectories, two appearances of the same scene are task-equivalent when they induce the same margin operator almost everywhere under \(\nu\):

\[
X_1\sim_S X_2 \iff M_{S,X_1}(\tau)=M_{S,X_2}(\tau)\quad \nu\text{-a.e.}
\]

The learned object is therefore an operator over candidate decisions, not a globally invariant feature vector.

## 2. Canonical candidate panel

For scene intent \(c\), construct a deterministic ordered panel

\[
\Pi(c)=\{\tau_1,\ldots,\tau_K\}
\]

with fixed semantics and source-defined weights \(w_i>0\), \(\sum_iw_i=1\). The corresponding margin panel is

\[
\mathbf m=[M_S(\tau_1),\ldots,M_S(\tau_K)]^\top.
\]

Panel construction is part of the method specification. Changing it after observing target results changes the estimand and is prohibited without a new experiment registration.

## 3. Affine task-operator factorization

Define

\[
b=\sum_iw_im_i,\qquad
a=\sqrt{\sum_iw_i(m_i-b)^2},\qquad
\rho_i=\frac{m_i-b}{a},
\]

when \(a>\epsilon\). Then

\[
\mathbf m=a\boldsymbol\rho+b\mathbf 1,
\quad a>0,
\quad \sum_iw_i\rho_i=0,
\quad \sum_iw_i\rho_i^2=1.
\]

The decomposition is unique under these constraints. Degenerate panels with \(a\le\epsilon\) are flagged rather than assigned arbitrary relative geometry.

Interpretation:

- \(\boldsymbol\rho\): relative candidate safety shape;
- \(a\): separation or sensitivity scale;
- \(b\): absolute safety anchor relative to the zero boundary.

The primary hypothesis is not that this identity is novel—it is algebraic—but that assigning different statistical roles and prediction mechanisms to these terms yields a better robust decision interface.

## 4. Identifiability and set-valued prediction

Under appearance-only observation, distinct physical scenes may be observationally indistinguishable while having different absolute margins. Consequently, a universally correct point estimator of \(b\) is impossible without additional assumptions. AORTA-Drive therefore predicts a calibrated set

\[
(a,b)\in\mathcal C_\alpha(X,\Pi)
\]

and induces a joint margin set

\[
\mathcal M_\alpha=
\left\{a\hat{\boldsymbol\rho}+b\mathbf 1:(a,b)\in\mathcal C_\alpha\right\}.
\]

Joint coverage is essential: independently calibrated per-candidate intervals do not automatically provide simultaneous panel coverage.

## 5. Architecture

### 5.1 Frozen visual encoder

A frozen DINOv3-H+ encoder produces multi-scale image tokens. Freezing isolates the proposed operator-learning contribution from backbone adaptation.

### 5.2 Execution-tube tokenizer

Each candidate trajectory is converted into a swept-footprint tube with position, heading, time, footprint extent, and intent features. Tokens are sampled along the tube rather than only at centerline waypoints.

### 5.3 Geometry-aligned cross-attention

Candidate tokens query visual tokens at projected spatial support. The output is a candidate-conditioned evidence vector \(h_i\). This keeps the learned interface tied to where the candidate would execute.

### 5.4 Permutation-equivariant panel reasoner

A panel transformer processes \(\{h_i\}_{i=1}^K\) with shared parameters and candidate-semantic embeddings. Reordering the panel and its metadata must reorder candidate outputs consistently.

### 5.5 Two-role prediction heads

- **Relative-shape head:** predicts raw scores, then applies weighted centering and normalization to produce \(\hat{\boldsymbol\rho}\).
- **Anchor-set head:** predicts a joint conditional distribution or nested regions for \((\log a,b)\).

Composition is analytic: \(\hat{\mathbf m}=\hat a\hat{\boldsymbol\rho}+\hat b\mathbf1\). This prevents a hidden direct-margin shortcut in the primary model.

## 6. Learning objective

The relative term uses a weighted shape loss plus optional pairwise ranking:

\[
\mathcal L_{rel}=\sum_iw_i(\hat\rho_i-\rho_i)^2
+\lambda_{rank}\mathcal L_{rank}.
\]

The anchor term uses a proper multivariate scoring rule, initially the energy score, on \(z=(\log a,b)\):

\[
\mathcal L_{ES}=
\mathbb E\|\hat z-z\|-
\frac12\mathbb E\|\hat z-\hat z'\|.
\]

The total objective is

\[
\mathcal L=\mathcal L_{rel}+\lambda_{anchor}\mathcal L_{ES}.
\]

Calibration is a separate source-only procedure and is never optimized on target labels.

## 7. Native-preserving arbitration

Let candidate \(n\) be the native planner output and let \([L_i,U_i]\) be simultaneous bounds induced by \(\mathcal M_\alpha\). The arbiter:

1. keeps the native trajectory unless \(U_n<0\);
2. if native violation is certified, selects the highest-priority alternative with \(L_i\ge\delta\);
3. otherwise keeps the native output and raises an abstention/fallback signal.

This asymmetric rule encodes a conservative intervention policy. It does not claim that a candidate with positive modeled margin is safe under unmodeled constraints.

## 8. Theory obligations

The paper must prove or state with exact assumptions:

1. uniqueness of the weighted affine decomposition for non-degenerate panels;
2. an observational non-identifiability result for absolute margin point prediction;
3. correctness of joint-set composition when \((a,b)\) and relative-shape error are covered;
4. monotonic widening of induced margin bounds as component uncertainty grows;
5. a finite-panel approximation bound under an explicit Lipschitz and coverage assumption.

Only Item 1 is currently implemented and unit-tested. Items 2–5 are proof obligations, not established results.

## 9. Minimal algorithmic baselines

- direct per-candidate margin regression;
- direct margin regression with appearance-consistency loss;
- factorized point prediction without sets;
- factorized model with marginal rather than joint calibration;
- full AORTA-Drive without arbitration;
- full AORTA-Drive with native-preserving arbitration.

## 10. Stop conditions

Stop or revise the main idea if controlled experiments show any of the following:

- relative-shape prediction is not more stable than direct margins;
- joint calibration fails on held-out source sessions before OOD evaluation;
- arbitration gains vanish under matched candidate coverage;
- improvement comes from panel expansion alone rather than the operator design;
- target-domain tuning is required for the reported result.
