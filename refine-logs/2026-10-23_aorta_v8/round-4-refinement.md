# Round 4 Refinement — V0 Freeze Candidate

本轮不增加新方法组件，只将 AORTA-v8.3 补齐为可执行前的冻结合同。

## Problem Anchor

- **问题:** appearance-only 变化可使冻结 planner 输出道路约束更差的轨迹；系统只在有证据证明 challenger 相对 native 的 nominal road margin 不下降时干预。
- **核心瓶颈:** 绝对道路零点可能不可辨识；必须在共享道路场的 additive quotient 上建立候选无关、选择后仍有效的相对证书。
- **Non-goals:** 不承诺任意 OOD、真实执行、碰撞、规则或完整闭环安全；不联合训练 planner；不把标准 conformal、difference constraints 或 min-plus solver 宣称为创新。
- **V0 成功标准:** 数学与几何 Gate A 通过，source Gate B 产生非平凡 meaningful non-inclusion 证书；在此之前没有性能或顶会原创主张。

## Final Core

冻结有限连通无向 4-neighbor 物理图 (G=(\Omega,E))。一个 candidate-free decoder 输出节点势 (u_\theta(X,q))。V0 使用固定物理 edge scale (s_e=\ell_e/\ell_{ref})，不实现 learned scale。

对 source calibration session：

\[
E_i=\max_{t\in T(Z_i)}\max_{e\in E}
\frac{|(Du_{it}-Dd_{it})_e|}{s_e},
\qquad
q_\alpha=E_{(\lceil(n+1)(1-\alpha)\rceil)}.
\]

若顺序统计量索引超过 (n)，系统不可部署。最终 edge budgets 为 (w_e=q_\alpha s_e)。

对 native swept set (B)，一次 multi-source min-plus transform：

\[
\psi_B(v)=\min_{b\in B}
[u_b+\operatorname{dist}_w(v,b)].
\]

任意 challenger swept set (A) 的 exact robust discrete contrast 为

\[
L_G(A,B)=\min_{a\in A}[u_a-\psi_B(a)]
=\inf_{d:\ |D(d-u)|\le w}
[\min_A d-\min_B d].
\]

每个 native 一次精确 Dijkstra，之后每个 challenger 只做集合 reduction。Dijkstra/min-plus 是标准实现工具，不是方法创新。

## Frozen V0 Runtime Contract

- 一个 native reference；
- 4-neighbor connected task graph；
- 图节点取 NAVSIM 数据接口中预注册的 ego-centric drivable raster crop，不在本理论阶段虚构尺寸；首个 experiment-plan 必须在任何运行前把 ROI、resolution、(|\Omega|)、(|E|) 写入不可变配置；
- fixed physical scale only；
- exact multi-source solver only；
- candidate-free field computed once per frame；
- 一条确定性 sound 的 continuous-geometry route；V0 默认 internal route，不能证明资格的 pair 拒绝；
- 冻结 DINOv3-H+ 与所有 planners；source-only train/development/calibration。

## Geometry Gate as a Proof-Carrying Interface

geometry checker 对 pair 返回

\[
\mathsf{GeomCert}_{j0}
=(\texttt{valid},h_j,\mathcal R,\texttt{protocol\_hash}).
\]

`valid=true` 只在确定性计算验证以下全部条件时产生：

1. (A_j\subseteq\mathcal B_j) 与 (A_0\subseteq\mathcal B_0)；
2. (A_j) 是 (mathcal B_j) 的内部 (h_j)-net，且 (h_j) 是解析/保守几何上界，不是有限测试点上的经验最大值；
3. 两个完整 footprints 均在认证 ROI (\mathcal R) 内且离散集合非空；
4. footprint、pose interpolation、车辆膨胀、grid 和数值容差与 `protocol_hash` 一致；
5. teacher distance 在该 ROI 的 1-Lipschitz 条件由 signed-distance 构造保证。

有效证书给出

\[
\Delta_{j0}^{cont}
\ge L_G(A_j,A_0)-h_j.
\]

checker 拒绝只使该 pair 不可授权，不消耗风险预算。如果 checker 未来只有 (1-\beta) 的 soundness，则必须把 field 与 geometry 事件联合记账：

\[
P(H=1)\le\min\{1,\alpha+\beta\}.
\]

V0 选择确定性 nominal-geometry soundness，即 (\beta=0)。这不覆盖 tracking/execution error。

## Unified Risk Quantifiers

令 (T(Z)) 为 calibration 前冻结的有限 session 时刻集。授权事件已经包含：候选动力学/意图资格、有效 `GeomCert` 和 (L_G-h_j\ge\gamma)。

\[
I_t(Z)=\mathbf1\{\text{time }t\text{ replaced native}\},
\qquad
I(Z)=\max_{t\in T(Z)}I_t(Z).
\]

唯一失败事件为 nominal continuous road failure：

\[
H(Z)=\mathbf1\left\{
\exists t\in T(Z):
I_t=1,
M_{0,t}^{cont}\ge0,
M_{sel,t}^{cont}<0
\right\}.
\]

设 (\mathcal F) 为所有 (t\in T(Z))、所有 (e\in E) 的 field coverage event，(\mathcal G) 为所有实际授权 pair 的 geometry bounds 有效。则

\[
H\subseteq\mathcal F^c\cup\mathcal G^c.
\]

V0 的确定性 checker 使 (P(\mathcal G^c)=0)，故 source-session 边际风险

\[
P_{D_{cal},Z}(H=1)\le\alpha.
\]

若未来 checker 为概率式，则替换为 (\alpha+\beta)，且干预条件风险最多为 (\min\{1,(\alpha+\beta)/P(I=1)\})。不得通过条件于 (\mathcal G) 重新解释 conformal coverage。

## Multi-source Numerical Contract

若实现为 super-source Dijkstra，使用

\[
c=\min_{b\in B}u_b,
\qquad
r_b=u_b-c\ge0.
\]

求解得到

\[
\widetilde\psi_B(v)=\min_b[r_b+\operatorname{dist}_w(v,b)]
=\psi_B(v)-c.
\]

最终必须恢复

\[
\psi_B(v)=\widetilde\psi_B(v)+c.
\]

Gate A 的 oracle cases 必须含：非零初始标签、共同 shift、(A=B)、严格包含、零权边、长链、环图和随机小图线性规划对照。浮点误差上界 (\varepsilon_{num}) 计入 (\gamma\)，不能靠事后 tolerance 放宽证书。

## Fair Baseline Contract

### Layer 1 — uncertainty geometry only

先训练一个合法 absolute SDF predictor (u_{abs})，它有实际 absolute anchor。冻结同一预测输出后分别构造：

1. absolute simultaneous field band + exact relative set-min；
2. global additive-gauge set + exact relative set-min；
3. local fixed-graph set + exact min-plus readout。

这一层只隔离 uncertainty geometry。absolute baseline 不继承一个任意 gauge 的 edge-only predictor，也不使用 test GT 对齐 offset。

### Layer 2 — legal end-to-end methods

在相同 frozen backbone、field architecture、source data、参数预算和选择规则下，各方法使用其合法训练目标：

- absolute SDF loss + absolute calibration；
- gauge/global relative loss + global uncertainty set；
- edge loss + local graph uncertainty set；
- 一个具体、预先冻结的 CDF-style low-rank joint residual construction。

所有方法对各自 uncertainty set 直接求保守的 relative set-min lower bound。对数值 infimum，primal feasible objective 通常是上界，不能冒充安全下界；必须使用解析解、dual certificate 或带方向正确的优化误差预算。

一般“所有 joint fields”可以包含本方法，不能作为可击败的有限 baseline 类。最近邻 CDF 的 rank、region、calibration score、solver 与误差预算必须在 experiment-plan 中具体冻结。

## Claim Levels

### Authorized now

- 已建立一条条件理论链：source-session edge event ⇒ all-pair discrete robust bound ⇒ sound continuous correction ⇒ adaptive nominal-road non-degradation。
- exact graph robust readout 的公式与标准 min-plus 实现明确。
- V0 具备进入 Gate A/B 的规范资格。

### Not authorized now

- 系统已经校准、实时或有效；
- 对 appearance OOD 有统计保证；
- 比 CDF、DriveReferee、scalar referee 更优；
- learned scale 必要；
- 顶会级原创性或“首次”。

## Innovation Decision Rule

只有下列四项同时成立，才能把论文定位为新方法而非严密应用：

1. 系统查新未发现等价的 candidate-free quotient-field + exact relative set-min selection 接口；
2. Layer 1 证明 local uncertainty geometry 在相同 predictor 与 coverage 下带来 meaningful non-inclusion 锐度；
3. Layer 2 证明合法训练后的完整方法优于具体 CDF/global/scalar 最近邻，而非利用 baseline handicap；
4. 一次 field evaluation 和一次 multi-source transform 的成本优于候选级重算，并在预注册运行配置下满足资源门。

若任一不成立，保留理论与工程价值，但降级为结构化 runtime-assurance 应用或停止论文主路线；不新增模块挽救标题。

## Handoff

下一步不是扩写网络，而是进入 experiment-plan，顺序严格为：

1. 冻结实际 NAVSIM ROI、resolution、graph size 与 geometry checker；
2. 实现 theorem oracle 和 deterministic geometry tests（Gate A）；
3. 用 source-only frozen feature cache 测最终米制 bound 的非平凡性（Gate B）；
4. Gate B 通过后才训练最小 field decoder；
5. 完成独立 novelty-check 后再决定论文贡献措辞。

本轮没有运行实验，也没有把设计审查当作系统验收。
