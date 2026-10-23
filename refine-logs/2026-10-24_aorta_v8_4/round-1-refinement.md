# AORTA-v8.4 Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** 对同一物理道路场，仅外观变化就可能使冻结端到端 planner
  产生更差的 nominal-road margin；系统必须只在 candidate-independent 视觉证据能够
  认证 challenger 相对 native 不退化时干预。
- **Must-solve bottleneck:** R004E-F 证明当前视觉势在 24/24 development groups 上
  降低平均 edge MAE，却在 24/24 groups 上输掉 exact group maximum。统一全图尺度把
  少量高误差边传播成所有路径的共同惩罚，抹去其余区域的有效视觉信号。
- **Non-goals:** 不恢复绝对安全零点；不修改冻结 planner；不声称 collision、规则、
  舒适性、闭环安全或任意 appearance-OOD coverage；不把 conformal、heteroscedastic
  regression、difference constraints 或 Dijkstra 单独包装成创新。
- **Constraints:** NAVSIM v1/v2 source-only；冻结 DINOv3-H+、图、teacher、geometry、
  candidate provider 与官方 calibration/test/target splits；以 recording group 为独立
  单位；新训练前先通过定理和 synthetic falsification gates。
- **Success condition:** 在未读取官方 calibration 的 development qualification 中，
  candidate-independent 异质 edge envelope 必须保持 group-level tail 可靠性，同时在
  米制 graph bounds 上严格优于 zero-field 与 uniform-scale controls；否则停止该路线。

## Anchor Check

- **保留内容：** 仍然估计 candidate-independent 道路差分场；仍然只认证 challenger
  相对 native 的 nominal-road margin；planner、候选生成器、官方 calibration/test/target
  与连续几何合同均不改变。
- **本轮收窄：** “heterogeneous scale 是最小充分机制”降为“由 R004E-F 支持、但尚待
  证伪的最小候选机制”。R004E-F 没有证明 tail 可预测，也没有证明异质尺度必要。
- **没有问题漂移：** 新增尺度只改变同一 edge residual uncertainty set 的形状，不把
  任务改成完整地图恢复、候选评分、规划器训练或 OOD 安全保证。

## Round 1 Review Findings and Repairs

| 审查问题 | 修订 |
|---|---|
| proper loss 与 tail term 联合后不再 proper，且会主动放大尺度 | 删除所有进入 scale head 的 group-tail loss；尺度只由 bounded Laplace proper loss 拟合 |
| fit、checkpoint selection 与 internal calibration 混用 | 冻结 42/8/16 recording-group 三路拆分；internal-cal 不参与任何拟合、初始化统计、超参或选择 |
| 旧模型已看过全部 66 组，不能用于有限样本内部 coverage | 后续真实 probe 必须从 fit-only 重新训练势场；旧 checkpoint 只作诊断，不作 coverage predictor |
| 新增除法、分位数与乘法缺少保守数值合同 | 明确每一步向外舍入、拒绝规则和 exact-rational oracle；map/geometry 误差继续单独扣除 |
| 同一 potential 下的尺度归因不清 | 所有 learned/uniform/evidence-only scale 对照共享同一冻结势场，并分别合法重新校准 |
| “零错误授权”混淆 deterministic soundness 与 conformal miscoverage | 只在 field event 及其他合同成立的 oracle cases 要求零 false acceptance；coverage 单独按 rank 检验 |
| 方法创新性被过早授权 | 只保留“待查新的贡献候选”；若公平对照仅证明标准组合有效，则降级为严谨系统贡献 |

## Simplicity Check

### 保留的最小机制

1. 一个重新合法训练并冻结的 node-potential predictor；
2. 在其冻结视觉 trunk 上增加一个 right/down edge-scale head；
3. 一个 session-level normalized split-conformal score；
4. 已有 arbitrary nonnegative edge weights 的 exact min-plus readout；
5. 已有 continuous geometry 与 map-error ledger。

### 本轮明确删除

- 删除 group-tail surrogate 及其 \(\lambda,\tau\)、全组梯度和 micro-batch 复杂性；
- 删除“anti-inflation 是算法贡献”的说法；
- 不增加第二 backbone、ensemble、MC dropout、VLM、RL、learned graph solver；
- 不为 v8.4 重新发明 shortest-path theorem；
- 不把 development set 当作 coverage set。

这使首个真实 probe 只回答一个问题：**部署时可见的冻结视觉证据，能否预测 edge
residual 的相对空间形状，从而在保持同一 session coverage 合同的同时提高最终连续几何
修正后的证书锐度？**

## Revised Method Thesis

给定由 source-only fit/selection groups 合法训练并冻结的 candidate-independent 势场
\(u(X)\)，学习一个同样 candidate-independent 的正 edge scale shape \(s(X)\)。在与所有
模型选择隔离的 recording-group internal-cal 集上，校准

\[
R(Z)=\max_{t\in T(Z),e\in E}
\frac{|(Du(X_t)-D\widetilde d_t)_e|}{s_e(X_t)}.
\]

若 learned shape 能定位稀疏视觉 tail，则 \(q_\alpha s_e(X_t)\) 会把代价集中在不可靠边，
而不是像 uniform budget 一样惩罚所有路径。该机制不改变 coverage 的统计单位，也不改变
任意非负异质 edge weight 下已有 robust set-min theorem。

这是一项待验证的机制假设，不是已经成立的必要性、充分性或创新性结论。

## Architecture Contract

```text
frozen DINOv3-H+ evidence + frozen projection/masks
                         │
                         ▼
           fit-only trained spatial trunk H(X)
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       frozen potential u      trainable scale logits a
                                    │
                                    ▼
                   bounded positive edge scale s(X)
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
 internal-cal session scores                     inference observation
              │                                           │
              ▼                                           ▼
      frozen q_alpha                              w_e = up(q_alpha s_e)
              └─────────────────────┬─────────────────────┘
                                    ▼
                   existing exact min-plus readout
                                    ▼
                 existing continuous geometry correction
                                    ▼
                     native-preserving arbitration
```

### Tensor and dependency contract

- Grid and directed storage remain the frozen 64x64 node grid and 8064 undirected
  right/down edges in the existing canonical order.
- The scale head emits two channels on the 64x64 spatial feature map. Invalid right/down
  border entries are structurally removed; they are not learned dummy edges.
- The scale head may read only frozen source observation features and frozen evidence
  masks. It cannot read native/challenger trajectories, planner scores, candidate IDs,
  calibration labels or target-domain statistics.
- For the first real probe, DINO, projection, spatial trunk and potential head are frozen
  before scale fitting. Only the scale head is updated.
- A candidate-dependent uncertainty model may still admit another conformal construction,
  but it violates AORTA's single candidate-independent field-event reuse contract and is
  therefore out of scope.

### Positive bounded parameterization

Let the head output \(a_e(X)\in\mathbb R\). The represented scale is

\[
s_e(X)=s_{min}+(s_{max}-s_{min})\,\sigma(a_e(X)).
\]

The certification-boundary value is binary64 and has metre units. R004E-G will freeze and
test \(s_{min}=2^{-10}\,\mathrm m\) and \(s_{max}=2^4\,\mathrm m\). These broad powers-of-two
bounds are numerical validity limits, not fitted error quantiles and not claimed physical
uncertainty bounds. The head is initialized so that represented \(s=1\,\mathrm m\). Any
nonfinite logit/scale, wrong shape/order or value outside the closed registered range makes
the session ineligible and retains native.

## Legal Data and Training Contract

### Frozen three-way split of the former 66 training groups

Before any new fitting, unique recording-group IDs are ordered by SHA-256 of a registered
namespace, seed and exact group ID. The first 42 are **fit**, next 8 are **selection**, and
last 16 are **internal-cal**. The mapping is persisted and hashed before training.

- **fit (42):** gradient updates and fit-only optimizer state；
- **selection (8):** checkpoint selection and only preregistered early stopping；
- **internal-cal (16):** read exactly once after potential, trunk, scale model, caps,
  score and checkpoint are frozen；
- **development (24):** one final mechanism qualification read; it is not a coverage set；
- **official calibration/test/target:** remain sealed.

An old checkpoint that was trained on all 66 groups is ineligible for the internal
coverage claim, even if only its head is later modified.

### Stage P — legal potential predictor

A fresh potential/trunk instance is trained only on fit groups using the frozen v8.3
potential objective and architecture. No new objective or capacity is introduced in this
stage. A single checkpoint is selected on the 8 selection groups using a preregistered
potential metric. It is then frozen permanently for every scale control.

This retraining is required solely to restore data-role independence; it is not a claimed
v8.4 contribution.

### Stage S — scale-shape predictor

With \(u\) and trunk frozen, define \(a_{t,e}=|(Du-D\widetilde d)_{t,e}|\). The scale head
is fitted on fit groups only by the dimensionless bounded Laplace negative log likelihood

\[
\mathcal L_{scale}
=\frac1N\sum_{t,e}
\left[
\frac{a_{t,e}}{s_{t,e}}+\log\frac{s_{t,e}}{1\,\mathrm m}
\right].
\]

Within unconstrained interior points the population pointwise optimum is the conditional
mean absolute residual. At active bounds it is the clipped constrained optimum. This loss
prevents unbounded scale from improving the objective, but it does **not** prove calibrated
budget improvement, tail predictability or epistemic uncertainty.

No standardized maximum, group-tail surrogate, certificate outcome or development result
backpropagates into the scale head. The checkpoint is selected only on the selection groups
by their same proper loss, with the epoch rule frozen before fitting.

## Statistical Contract

For frozen measurable \((u,s)\), each internal-cal group yields exactly one upper-safe
score

\[
R_i^+=\max_{t\in T(Z_i),e\in E}
\operatorname{up}\!\left(
\frac{\operatorname{up}(|(Du-D\widetilde d)_{it,e}|)}
{s_{it,e}}
\right).
\]

For \(n=16\) and the sole internal qualification level \(\alpha=0.1\), let

\[
k=\left\lceil(n+1)(1-\alpha)\right\rceil=16,
\qquad q_\alpha=R^+_{(k)}.
\]

Under exchangeability of the 16 internal-cal source sessions and a future source session,
and assuming no ties are handled anti-conservatively, the usual rank argument gives

\[
\Pr\{R_{new}\le q_\alpha\}\ge \frac{k}{n+1}=\frac{16}{17}>0.9.
\]

The guarantee is marginal over the registered source-session distribution. It is not a
conditional per-weather guarantee, a statement about the repeatedly used 24 development
groups, or an appearance-OOD guarantee. \(n=16\) cannot support the registered finite
\(\alpha=0.05\) threshold; that level remains reserved for the sealed official calibration
set after all development gates pass.

### Simultaneous field event and adaptive reuse

On \(R_{new}\le q_\alpha\), every registered time and edge satisfies

\[
|(Du-D\widetilde d)_{t,e}|\le w_{t,e},
\qquad w_{t,e}=\operatorname{up}(q_\alpha s_{t,e}).
\]

The existing graph theorem already accepts arbitrary nonnegative \(w\), so the same event
implies valid discrete robust bounds for all nonempty swept-set pairs queried at that time.
For a fixed native set, one exact native-seeded transform serves every challenger set;
changing the native generally requires a new transform. Candidate adaptation after seeing
\((u,s)\) introduces no additional multiple-comparison term because all pairs are covered
by the one field event.

Map-to-continuous teacher error, challenger outer-cover error, native interior conditions
and final geometry/numerical deductions remain the separate existing lifting ledger. They
are not added again to \(w\).

## Numerical Soundness Contract

The certificate is sound relative to the represented binary64 predictor outputs only if
all new arithmetic is outward-safe.

1. **Residual upper bound:** compute both directed subtractions with rounding toward
   \(+\infty\), then take their maximum and zero; do not compute ordinary subtraction
   followed by `abs` and assume it is an upper bound.
2. **Standardization:** require finite \(s\ge s_{min}>0\); divide the residual upper bound
   by represented \(s\) and round toward \(+\infty\).
3. **Group maximum:** exact comparison of represented upper scores; maximum adds no
   arithmetic optimism.
4. **Quantile:** stable ascending order statistic at registered integer rank \(k\); no
   interpolation and no averaging.
5. **Edge budget:** multiply represented \(q_\alpha\) and \(s\), round toward
   \(+\infty\). Exact zero remains zero. Positive underflow is rounded to the smallest
   positive representable value; overflow or nonfinite output rejects the session.
6. **Optional edge-only correction:** if a future represented-input correction is added,
   it must be nonnegative and added toward \(+\infty\). No such correction is active in
   R004E-G.
7. **Graph readout:** reuse the existing outward-safe path additions and downward final
   subtraction. Invalid/disconnected/nonfinite cases retain native.

R004E-G must compare every new primitive against exact `Fraction` arithmetic over
registered boundary and randomized cases. Passing graph arithmetic with already-rounded
weights is insufficient unless score and weight construction also pass.

## Identifiability and Fair Controls

For any positive global constant \(c\), away from active caps/floors and ignoring finite
rounding,

\[
s'_e=cs_e,\quad R'_i=R_i/c,\quad q'_\alpha=q_\alpha/c,
\quad q'_\alpha s'_e=q_\alpha s_e.
\]

Thus global amplitude is a calibration gauge; only relative variation across observations
and edges can improve budgets. Exact invariance is not claimed when the transform hits
bounds or when finite outward rounding differs. R004E-G checks a power-of-two interior
rescaling where exact binary representation is available.

The real probe uses one frozen \(u\) for all controls, and every control receives its own
legal internal-cal threshold:

1. **learned heterogeneous scale** \(s_e(X)\)；
2. **uniform scale** \(s_e\equiv1\,\mathrm m\)；
3. **evidence-only scale**, a preregistered deterministic function of evidence masks with
   any fit parameters learned on fit only；
4. **zero-field control**, evaluated separately to retain the R004E-F reference.

The primary comparison is learned versus uniform on the same potential. Zero-field is not
allowed to confound potential quality with scale quality.

## Decision-Relevant Endpoints

Edge-level standardized maxima are validity diagnostics, not the final success endpoint.
After each control is independently calibrated, the same frozen candidate pairs pass
through:

\[
\text{heterogeneous edge weights}
\to \text{exact robust set-min}
\to \text{map/continuous geometry deductions}
\to \text{metre-valued relative-margin lower bound}.
\]

The first real probe pre-registers two co-primary development endpoints:

1. continuous-geometry-corrected certificate sharpness on the frozen eligible pair set；
2. certified recall among frozen certifiable opportunities at \(\gamma_{task}\), with
   candidate supply reported separately.

It must also report mean/worst-5/all-24 budget summaries, but those cannot alone qualify
the mechanism. A method that improves edge summaries yet not the corrected task certificate
fails the v8.4 mechanism claim.

## Falsification Plan

### R004E-G — theorem and synthetic oracle, before any retraining

R004E-G is CPU-only and must be registered before execution. It tests:

- deterministic, order-invariant and disjoint 42/8/16 group assignment；
- finite-sample rank behavior at \(n=16,\alpha=0.1\), plus exhaustive toy rank cases；
- outward residual, division and budget multiplication against exact rationals；
- nonpositive, out-of-range, underflow, overflow and nonfinite rejection；
- interior power-of-two global scale gauge invariance within registered numerical bound；
- the explicit counterexample showing that adding \(\lambda a/s\) moves the scale optimum
  from \(a\) to \((1+\lambda)a\), and therefore is absent from the revised method；
- heterogeneous weights through the existing solver against exact small-graph oracle；
- deterministic zero false authorization only conditional on the field event and all
  geometry contracts.

Any optimistic arithmetic counterexample, split leakage or theorem/oracle mismatch blocks
real training. Ordinary conformal miscoverage at the allowed rank is not mislabeled as a
deterministic soundness failure.

### R004H — later real mechanism probe

Only after R004E-G passes, freeze a complete R004H protocol containing Stage P, Stage S,
all controls, exact checkpoints, development endpoints and stop rules. Official
calibration/test/target remain unread.

The learned-scale route stops if any of the following occurs:

- legality: internal-cal influenced fitting/selection or any control is not independently
  recalibrated；
- soundness: a numerical/theorem/geometry gate fails；
- attribution: learned scale does not beat uniform scale on the same \(u\)；
- task value: no strict improvement on both registered corrected-certificate endpoints；
- collapse: improvement is caused only by invalid cases, cap saturation or a changed
  candidate set；
- novelty: later literature review/fair direct uncertainty-set baselines show only a
  routine standard-tool integration.

## Appearance Diagnostic Boundary

For input-dependent scale, the earlier fixed-scale diagnostic must be replaced by

\[
R(X')\le \kappa(X,X')R(X)
+\max_e\frac{|D[u(X')-u(X)]_e|}{s_e(X')},
\qquad
\kappa(X,X')=\max_e\frac{s_e(X)}{s_e(X')}.
\]

It may diagnose whether appearance changes alter standardized residuals through potential
or scale. It does not convert source split-conformal coverage into OOD coverage and is not
a training objective in v8.4.

## Contribution Boundary and Decision Rule

### Currently authorized

- a mathematically coherent candidate mechanism for localizing sparse visual residual
  tail while preserving the existing candidate-independent field event；
- a clean statistical/data contract and a decision-relevant falsification plan；
- reuse of an exact robust graph readout and continuous lifting already established in
  AORTA-v8.3.2.

### Not authorized

- “first”, “novel conformal method”, “new shortest-path algorithm” or “minimal sufficient”；
- arbitrary OOD, collision or complete driving-safety guarantees；
- top-venue method originality before dedicated novelty search and strongest direct
  uncertainty-set baselines；
- any claim that heteroscedastic NLL alone solves simultaneous tail risk.

### Paper-level survival rule

The method claim survives only if, under the same predictor/data/statistical contract,
heterogeneous local budgets deliver a strict corrected-certificate advantage over uniform
scale and the strongest global-gauge/joint-field uncertainty interfaces, and the advantage
cannot be reproduced by merely changing the loss of the uniform predictor. If not, v8.4 is
reported as a negative or incremental systems result rather than inflated into a new
algorithmic principle.

## Revised Experiment Handoff

- **First executable step:** R004E-G synthetic theorem/numerics/split audit.
- **First training step:** not yet authorized; requires R004E-G pass and a separately
  committed R004H registration.
- **Compute:** R004E-G CPU only; R004H estimate deferred until the legal Stage P replay and
  scale-head batch footprint are measured without reading development outcomes.
- **Sealed assets:** official calibration, test and target-domain data remain unopened.

