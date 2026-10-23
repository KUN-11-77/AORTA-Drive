# AORTA-v8.4 Round 2 Refinement

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

问题锚点未改变。模型仍然输出与候选无关的道路差分证据，planner 与 candidate provider
不训练，干预仍由 challenger 相对 native 的 continuous nominal-road margin 下界授权。
本轮只修复数据资格、浮点区间和机制归因，不扩大安全语义。

## Round 2 Repairs

1. 42/8/16 中最后 16 个历史 train groups 改名为 **proxy-cal**。它们严格不参与 v8.4
   新训练与选择，但因历史上已参与旧模型训练/方法探索，不承载正式 split-conformal
   coverage；只用于在不读取官方 calibration 时排练完整流程。
2. 正式有限样本保证只由最终模型和全部协议冻结后首次读取的官方 calibration groups
   产生。
3. certification residual 从四个 represented node values 直接构造区间，不先普通计算
   两个 edge gradients。
4. proxy/official calibration group 不允许静默删除；任一非法 group 使该 variant 的阈值
   为 \(+\infty\) 并拒绝部署。
5. 增加无新参数的 per-observation pooled-scale control，区分整体 observation scale
   adaptation 与真正 edge-wise spatial allocation。
6. 将真实 probe 的单一 primary endpoint 冻结为共同 oracle-opportunity 分母上的
   \(R_{cert|better}\)；certificate sharpness 降为配对诊断，避免循环定义的双主终点。

## Simplicity Check

保留一个 node-potential predictor、一个 right/down scale head、一个 scalar session score、
已有 exact min-plus solver 和已有 continuous geometry lifting。Scale head 仍只用 bounded
Laplace proper loss；不恢复 group-tail loss、联合训练、第二 backbone、ensemble、VLM、RL
或 learned graph solver。新增 pooled control 只是对已学 \(s\) 做无参数池化，不增加模型。

## Method and Architecture

```text
frozen DINOv3-H+ + frozen projection/evidence masks
                         │
                         ▼
          fit-only trained spatial trunk H(X)
               ┌─────────┴─────────┐
               ▼                   ▼
       frozen node potential u   edge logits a
                                   │
                                   ▼
                         bounded scale s(X)
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
       sealed calibration score                  inference weights
               │                                       │
               ▼                                       ▼
            q_alpha                              up(q_alpha s)
               └───────────────────┬───────────────────┘
                                   ▼
                  existing exact graph certificate
                                   ▼
                existing continuous correction ledger
                                   ▼
                    native-preserving arbitration
```

- Frozen grid: 4096 nodes and canonical 8064 undirected right/down edges.
- Scale head: two channels on the frozen 64x64 trunk feature map; invalid border entries
  are structurally removed.
- Inputs: observation features and evidence masks only. No native/challenger/planner score,
  candidate ID, calibration label or target statistic may enter \(s\).
- First probe: DINO, projection, trunk and potential are frozen before Stage S; only the
  scale head trains.
- Scale representation:

\[
s_e(X)=2^{-10}\mathrm m+(2^4-2^{-10})\mathrm m\,\sigma(a_e(X)).
\]

  Bounds are broad numerical validity limits fixed before outcomes, not error quantiles.
  Certification receives binary64 represented values. Nonfinite or out-of-contract output
  invalidates the complete recording group.

## Data and Training Roles

The former 66 train groups are deterministically SHA-256 ordered under a registered
namespace and divided 42/8/16:

- **fit 42:** all Stage P/Stage S gradient updates and optimizer state；
- **selection 8:** preregistered checkpoint selection only；
- **proxy-cal 16:** one post-freeze internal pipeline rehearsal；
- **development 24:** one final mechanism qualification read；
- **official calibration/test/target:** sealed until their existing gates permit access.

The persisted role ledger must disclose that all 66 groups were historically exposed to
the project. “No v8.4 reuse” is not rewritten as “never seen.”

### Stage P — legal fresh potential

A fresh potential/trunk is trained only on fit groups using the already frozen v8.3
potential architecture and objective, including any Stage-P tail term already present.
One checkpoint is selected on selection groups by a precommitted metric and then frozen.
This replay restores clean v8.4 parameter fitting roles but does not retroactively make
proxy-cal historically independent.

### Stage S — scale only

With \(u,H\) frozen, fit the scale head on fit groups by

\[
\mathcal L_{scale}=\frac1N\sum_{t,e}
\left[
\frac{|(Du-D\widetilde d)_{t,e}|}{s_{t,e}}
+\log\frac{s_{t,e}}{1\mathrm m}
\right].
\]

The selection metric is the same proper loss on selection groups. In an ideal unrestricted
function space the population optimum is the conditional mean absolute residual, clipped
by the allowed scale range; no claim is made that a finite head or optimizer reaches it.
The objective is not a tail quantile and gives no guarantee that the sparse tail is
predictable. No proxy-cal/development score, certificate outcome or max-tail term enters
training or selection.

## Conditional Statistical Theorem

For any predictor and score construction fully frozen independently of an exchangeable
calibration sample \(Z_1,\ldots,Z_n\), define one upper-safe score per session,

\[
R_i^+=\max_{t\in T(Z_i),e\in E}\frac{a_{it,e}^+}{s_{it,e}},
\]

with division outward-rounded upward. Let

\[
k=\lceil(n+1)(1-\alpha)\rceil,\qquad q_\alpha=R^+_{(k)}.
\]

If \(k\le n\), standard split-conformal ranking gives

\[
P(R_{new}^+\le q_\alpha)\ge k/(n+1)\ge1-\alpha.
\]

Since the exact represented residual is no larger than \(a^+\), this implies simultaneous
coverage of every registered time and edge. The existing arbitrary-nonnegative-weight graph
theorem then covers every nonempty adaptive swept-set pair on the same field event.

This theorem applies later to the unopened official calibration set if all dependencies are
frozen first. For the historical proxy-cal set, \(n=16,\alpha=.1,k=16\) is only a protocol
statistic; **no 16/17 coverage claim is made from its observed threshold**. It cannot support
finite \(\alpha=.05\).

## Complete Four-Term Residual Interval

For represented binary64 node values on oriented edge \(i\to j\), the exact real residual
of those represented values is

\[
r_e=(u_j-u_i)-(\widetilde d_j-\widetilde d_i).
\]

Production certification first encloses both gradients:

\[
g_u^- = \operatorname{down}(u_j-u_i),\quad
g_u^+ = \operatorname{up}(u_j-u_i),
\]

\[
g_d^- = \operatorname{down}(\widetilde d_j-\widetilde d_i),\quad
g_d^+ = \operatorname{up}(\widetilde d_j-\widetilde d_i).
\]

It then computes

\[
a_e^+=\max\left\{
\operatorname{up}(g_u^+-g_d^-),
\operatorname{up}(g_d^+-g_u^-),0
\right\}.
\]

This directly encloses \(|r_e|\). It is forbidden to first ordinary-round \(Du\) and
\(D\widetilde d\) and then compare them. R004E-G includes the cancellation counterexample

\[
u_j=1,\ u_i=-2^{-53},\ \widetilde d_j=1,\ \widetilde d_i=0,
\]

whose exact residual is \(2^{-53}\) although naive rounded gradients can coincide.

After \(a^+\):

1. require finite \(s\in[s_{min},s_{max}]\), divide upward；
2. take exact represented maximum within group；
3. choose an uninterpolated integer order statistic；
4. compute \(w_e=\operatorname{up}(q_\alpha s_e)\)；
5. preserve exact zero; round positive underflow outward; reject overflow/nonfinite；
6. reuse outward path sums and downward final graph subtraction.

Map error and continuous geometry terms remain separate final deductions and are never
silently folded into \(w\).

## Invalid-Group Contract

- Fit/selection: a nonfinite loss or malformed tensor fails the run/checkpoint; no sample
  is dropped.
- Proxy/official calibration: all preregistered groups remain. Any group without a valid
  complete score receives \(+\infty\); hence a maximum-rank threshold becomes infinite and
  the variant cannot deploy.
- Development/test: every frozen unit remains in the denominator. Invalid prediction,
  geometry ineligibility or arithmetic failure retains native and counts as not certified;
  it is separately reported by reason.
- No post-hoc replacement, resampling or denominator shrinking is permitted.

## Scale Identifiability and Controls

A constant global rescaling is a calibration gauge only in the interior and exact-real
limit:

\[
s'=cs,\quad q'=q/c,\quad q's'=qs.
\]

Caps/floors and outward rounding can break exact equality. R004E-G therefore tests only
registered power-of-two interior cases against exact rational arithmetic.

Every control uses the same frozen \(u\) and gets its own proxy/official threshold:

1. learned observation-and-edge scale \(s_{t,e}\)；
2. **pooled observation scale**
   \(s^{pool}_{t,e}=\exp(|E|^{-1}\sum_e\log s_{t,e})\), broadcast to all edges；
3. uniform \(1\mathrm m\) scale；
4. preregistered evidence-mask-only scale；
5. zero-field reference, reported separately from the same-\(u\) attribution.

Learned beating uniform but not pooled supports only observation-level scale adaptation.
Learned beating both supports an edge-allocation effect, not yet a causal explanation of
which visual evidence predicts tail errors.

## Exact Evaluation Endpoint

Use the existing evaluation unit \(U=(Z,t)\): uniformly sampled session, then a frozen
time within session. For each unit, define the method-independent opportunity set

\[
Q_B(U)=\{j:\ j\text{ is a frozen candidate, geometry-eligible, meaningful continuous
non-inclusion, and }\Delta^{teacher}_{j0}\ge\delta_M\}.
\]

Thus \(B(U)=1\iff Q_B(U)\ne\varnothing\). For method \(m\),

\[
C_m(U)=1\iff \exists j\in Q_B(U):L^{cert,m}_{j0}\ge\gamma_{task}.
\]

The single primary mechanism endpoint is

\[
R^m_{cert|better}=\frac{\sum_U C_m(U)}{\sum_U B(U)},
\]

on the identical frozen denominator. Invalid/fallback units contribute zero to the
numerator and remain in the opportunity denominator when \(B=1\). Report per-group
numerators/denominators, macro session summaries and a paired group bootstrap; anchors are
not treated as independent replicates.

The paired diagnostic sharpness for each opportunity unit is

\[
S_m(U)=\max_{j\in Q_B(U)} L^{cert,m}_{j0}.
\]

Its full distribution and learned-minus-control paired differences are reported, with
invalid values recorded as \(-\infty\), not silently filtered. It is not a co-primary
success gate. Candidate supply \(P(B)\), unconditional \(P(C_m)\), selected useful rate and
invalid counts are reported separately.

R004H must freeze \(\delta_M,\gamma_{task}\), continuous non-inclusion thresholds,
bootstrap seed, tie rule and a minimum practically meaningful recall difference before
any new development read.

## R004E-G Falsification Gate

Before any retraining, a registered CPU audit must verify:

- deterministic/order-invariant/disjoint 42/8/16 assignment plus historical exposure
  labels；
- generic conformal rank enumeration and correct proxy wording；
- four-term interval arithmetic against `Fraction`, including cancellation；
- outward division and budget multiplication against exact rationals；
- invalid/overflow/underflow behavior and no silent group filtering；
- the previous tail-loss scale inflation counterexample；
- interior power-of-two gauge behavior；
- heterogeneous graph weights against the existing exact oracle；
- zero deterministic false authorization only when the field and geometry premises hold.

Any optimistic numerical case, leakage misstatement or graph mismatch blocks R004H. Passing
R004E-G qualifies the primitives and protocol only; it does not prove learned uncertainty,
coverage on historical groups, Gate B or innovation.

## R004H Real-Probe Decision Rule

R004H may be registered only after R004E-G passes. It must train the fresh Stage P model,
freeze it, train Stage S, compute all controls, read proxy-cal once, and then read the 24
development groups once under a complete precommitted endpoint contract.

- learned must strictly beat uniform on the primary recall endpoint to support scale
  adaptation；
- learned must strictly beat pooled scale to support edge-wise spatial allocation；
- no conclusion is allowed if legality/soundness fails or improvements depend on filtered
  invalid cases；
- official calibration remains sealed until architecture, checkpoints, scores and all
  decision rules are irrevocably frozen；
- novelty remains unauthorized until dedicated literature search and strongest
  global-gauge/joint-field baselines are complete.

## Appearance Diagnostic

For input-dependent scale, only the diagnostic inequality is retained:

\[
R(X')\le \kappa(X,X')R(X)+
\max_e\frac{|D[u(X')-u(X)]_e|}{s_e(X')},
\qquad
\kappa=\max_e\frac{s_e(X)}{s_e(X')}.
\]

It is not a training objective and never upgrades source coverage to OOD coverage.

## Contribution Boundary

The authorized object is now a falsifiable method candidate: observation-and-edge-dependent
scale allocation attached to a candidate-independent graph-difference certificate, with
legal calibration and exact decision-space evaluation. Normalized conformal, bounded
Laplace NLL, shortest paths, difference constraints and heteroscedastic regression are
explicitly standard components.

Top-venue algorithmic originality remains unproven. It survives only if edge-wise allocation
beats pooled/uniform and later beats the strongest direct global-gauge/joint-field interfaces
under the same predictor, calibration and continuous geometry contracts. Otherwise the
result is reported as incremental systems integration or a negative finding.

