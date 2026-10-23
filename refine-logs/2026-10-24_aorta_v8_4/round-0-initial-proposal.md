# Research Proposal: AORTA-v8.4 Heterogeneous Edge Envelopes

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

## Technical Gap

V8.3 使用固定 (s_e=1) 并校准 session score

\[
R(Z)=\max_{t,e}|(Du-D\widetilde d)_e|.
\]

这在统计上正确，但一个坏边决定全图统一预算。R004E-F 表明这不是“没有视觉信号”：
视觉场平均 MAE 相对零场降低 10.79%，所有组的 field cosine 为正；失败来自少量
wrong-direction high-gradient edges。扩大 decoder、输出 Lipschitz projection 或删除
all-missing nodes都不针对这个瓶颈。最小充分修改是让视觉系统同时给出每条边的正
uncertainty shape，并对 standardized session maximum 做 calibration，使高不确定性只在
穿越对应边的 graph paths 上付费。

## Method Thesis

- **One-sentence thesis:** 由观测预测 candidate-independent node potential 与正 edge
  uncertainty shape，并以 recording-group maximum standardized residual 校准，可将稀疏
  视觉 tail 局部化为异质 graph metric，同时保留所有自适应候选共享的 simultaneous
  relative-margin certificate。
- **Smallest adequate intervention:** 保留冻结 backbone、共享视觉 trunk、节点势、
  exact min-plus readout 和 geometry theorem，只新增一个两方向 edge-scale head，并把
  loss 从 anchor-uniform residual 改为 proper scale loss 加 group-tail surrogate。
- **Frontier fit:** DINOv3-H+ 已提供强冻结视觉表征；LLM、VLM、diffusion、RL 或 learned
  planner 不解决 sparse edge-tail localization，加入它们会造成问题漂移。

## Contribution Focus

- **Dominant contribution:** observation-conditioned heterogeneous edge envelope：把
  candidate-free 视觉不确定性变成 exact graph metric，并给出 session-conformal 到任意
  adaptive swept-set pair 的完整条件链。
- **Supporting contribution:** 与独立 recording-group maximum 对齐的 scale-learning
  objective 和防尺度膨胀合同。
- **Explicit non-contributions:** DINO、Laplace NLL、normalized conformal scores、
  shortest paths 和 min-plus transform 本身；原创性必须经独立查新与公平 baseline 后
  才能授权。

## Proposed Method

### Complexity Budget

- **Frozen/reused:** DINOv3-H+ feature cache、12-slot projection、masked fusion、spatial
  trunk、node-potential head、teacher、graph、candidate provider、geometry checker、
  exact outward-safe solver。
- **New trainable component:** 一个共享 trunk 上的 horizontal/vertical edge log-scale
  head；无第二 backbone、candidate encoder 或 learned graph solver。
- **Intentionally excluded:** hard confidence mask、per-edge clipping、ensemble、MC dropout、
  VLM teacher、RL arbitration、learned candidates、target adaptation。

### System Overview

```text
frozen DINO evidence + masks
            │
            ▼
     shared spatial trunk H(X)
       ┌────┴────────┐
       ▼             ▼
node potential u   edge raw scale r
       │             │
       │       s = s_min + softplus(r), capped finite
       └────┬────────┘
            ▼
session calibration: max |Du-Dd| / s  -> q_alpha
            ▼
heterogeneous edge budget w_e = q_alpha s_e
            ▼
exact multi-source min-plus + continuous corrections
            ▼
native-preserving certified arbitration
```

### Core Mechanism

For observation (X_t), the network outputs zero-gauge potential (u_t\in\mathbb R^V)
and finite positive scale (s_t\in[s_{min},s_{max}]^E). The scale head emits one
right-edge and one down-edge channel from the shared 64×64 feature map; invalid border
entries are structurally removed, producing the exact frozen 8064-edge order.

Define residual

\[
r_{t,e}=(Du_t-D\widetilde d_t)_e,
\qquad
z_{t,e}=|r_{t,e}|/s_{t,e}.
\]

After all model choices are frozen, independent calibration sessions use

\[
R_i=\max_{t\in T(Z_i),e\in E}z_{it,e},
\quad
q_\alpha=R_{(\lceil(n+1)(1-\alpha)\rceil)}.
\]

Exchangeability gives the simultaneous event

\[
\forall t,e:\ |r_{t,e}|\le q_\alpha s_{t,e}
\]

with probability at least (1-\alpha). This remains valid for learned (s(X)) because
the complete predictor is trained and frozen independently of calibration, and the
score is one deterministic scalar per session. Set

\[
w_{t,e}=q_\alpha s_{t,e}+\delta_{t,e},
\]

where nonnegative (delta) contains registered numerical/teacher corrections. The
existing robust set-min theorem holds for arbitrary nonnegative edge weights, so one
exact min-plus transform certifies every candidate pair at that time. Scale and field
remain candidate-independent; candidate adaptation therefore does not invalidate the
single field event.

### Anti-inflation and Identifiability

The scale head is trained with a bounded Laplace proper loss

\[
\mathcal L_{proper}
=\frac1N\sum_{t,e}
\left(\frac{|r_{t,e}|}{s_{t,e}}+\log s_{t,e}\right).
\]

Uniform scale inflation cannot minimize this loss because the log term diverges. A
global rescaling (s\mapsto cs) is also neutralized at calibration time by
(q_\alpha\mapsto q_\alpha/c), so only spatial allocation of uncertainty can improve
meter-valued graph budgets. Frozen finite (s_{min},s_{max}) prevent zero division and
nonfinite graph weights; their values are fixed from train-only residual units, not
development/calibration performance.

### Group-Aligned Training Objective

Each optimization unit is one complete train recording group with 16 anchors. Define

\[
\widetilde R_g
=\tau\log\left[
\frac1{16|E|}\sum_{t,e}\exp(z_{t,e}/\tau)
\right].
\]

The minimal objective is

\[
\mathcal L_g=\mathcal L_{proper,g}+\lambda\widetilde R_g.
\]

It directly targets the same nesting level as calibration without pretending anchors
or edges are independent. Training uses group micro-batches with exact accumulated
sums/log-sum-exp; checkpoint selection is performed only after a frozen internal split
of the 66 former train groups. The 24 development groups remain a method-development
qualification set; official calibration stays unopened.

### Inference Path

One shared forward pass produces (u_t,s_t). The calibrated (q_\alpha) turns scale
into meter-valued edge weights. Existing outward-safe Dijkstra and asymmetric geometry
lifting are unchanged. Large uncertainty does not authorize a pair; it lengthens paths
and makes the lower bound more conservative. If any scale is nonfinite/out of range,
the system retains native.

### Failure Modes and Diagnostics

- **Scale ranks residuals poorly:** heterogeneous budgets do not beat uniform-scale
  sharpness; stop the learned-scale route.
- **Scale inflation/collapse:** proper-loss and bound tests fail; reject before real
  training rather than retune caps on development.
- **Good mean but unchanged group tail:** group surrogate is not aligned enough;
  report failure rather than add heads.
- **Candidate leakage:** any candidate/planner input to (s) invalidates the shared
  adaptive-pair theorem; structural test must reject it.
- **Global maximum still dominates (q):** report nontriviality failure; do not claim
  that heterogeneity automatically removes simultaneous-event conservatism.

### Novelty and Elegance Argument

The paper-level hypothesis is not “use heteroscedastic regression.” It is that a
candidate-free, observation-conditioned edge uncertainty shape is the minimal interface
that converts sparse visual failures into a reusable path metric, so a single
session-level event can serve arbitrarily many adaptive trajectory comparisons by an
exact robust set-min operator. This is a focused coupling of representation, statistical
unit and decision geometry. Its novelty remains provisional until a dedicated search
against normalized conformal field prediction, uncertainty-weighted neural shortest
paths and safety-filter/arbitration work.

## Claim-Driven Validation Sketch

### Claim 1: The heterogeneous envelope is mathematically sound and non-cheatable

- **Minimal experiment:** synthetic exchangeable sessions plus exact LP/min-plus oracle;
  scale-rescaling invariance, anti-inflation, edge-order and nonfinite rejection tests.
- **Baselines:** uniform scale; globally rescaled identical scale; adversarial inflated
  scale; exact LP.
- **Metric:** coverage-event mismatch, robust-bound error, meter-budget invariance,
  false acceptance count.
- **Required evidence:** zero false acceptance and equality within frozen numerical
  tolerance; global rescaling cannot improve meter-valued budgets.

### Claim 2: Learned spatial uncertainty recovers useful visual signal without hiding tail

- **Minimal experiment:** one registered source-only probe after R004E-G, using a frozen
  fit/internal-cal split of the 66 train groups and one final 24-group development read.
- **Baselines:** zero field; old visual uniform scale; evidence-count-only scale; scale
  head with group-tail term deleted.
- **Metric:** group-level simultaneous standardized residual, meter-valued worst-5/all-24
  edge budget and downstream exact-bound sharpness.
- **Required evidence:** strict improvement over zero and uniform-scale controls on both
  registered development endpoints, with no coverage or candidate-leakage violation.

## Experiment Handoff Inputs

- **Must prove:** learned observation-dependent scale preserves split-conformal coverage;
  heterogeneous weights preserve exact robust set-min readout; training cannot win by
  global scale inflation.
- **Must-run ablations:** uniform scale, evidence-count scale, no group-tail term.
- **Highest-risk assumption:** sparse high residuals are predictable from deployment
  evidence strongly enough to localize budget without making all useful paths expensive.

## Compute & Timeline Estimate

- R004E-G theorem/oracle: CPU, less than one day.
- Real C0/C1 contract: one RTX 3090, less than two GPU-hours.
- One full registered probe plus controls: approximately the prior 20-epoch cost per
  trainable variant; exact budget must be measured after group micro-batching.
- No official calibration, test, target-domain data or full NAVSIM evaluation until the
  new development qualification passes.
