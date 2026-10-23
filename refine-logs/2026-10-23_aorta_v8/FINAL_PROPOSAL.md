# AORTA-v8.3 Final Proposal

## Calibrated Minitive Field Envelopes for Appearance-Robust Frozen-Planner Arbitration

> 状态：V0 条件理论规范已冻结；尚无实验性能、OOD coverage、实时性或顶会原创性结论。

## 1. 核心问题

同一物理场景仅改变外观时，冻结端到端 planner 可能产生道路约束更差的轨迹。AORTA 不要求视觉表征全局不变，也不尝试凭视觉恢复不可辨识的绝对安全零点。它只解决一个受限但明确的问题：

> 当 challenger 与 native 共享同一物理道路场时，能否从 source-only 视觉证据中认证 challenger 的 nominal road margin 不低于 native，并只在该证书成立时干预？

方法不覆盖碰撞、动态障碍、交通规则、真实 tracking error、完整闭环安全或任意 appearance OOD 的统计保证。

## 2. 冻结任务对象

令 (G=(\Omega,E)) 为有限、连通、无向的 4-neighbor ego-centric 任务图。GT signed-distance field 为 (d_S:\Omega\to\mathbb R)。冻结 tracker 将轨迹 (	au) 转为非空 swept-node set (A(\tau)subseteq\Omega)。

\[
M_S^\Omega(\tau)=\min_{q\in A(\tau)}d_S(q)-m_{buf}.
\]

网络学习 candidate-free 节点势 (u_\theta(X,q))，只要求恢复 (d_S) 的相对空间结构。共同常数属于 graph incidence operator 的零空间，且会在 trajectory contrast 中消去。

## 3. 最小架构

```text
camera history + calibrated camera/ego geometry
                         │
                         ▼
              frozen DINOv3-H+ tokens
                         │
        fixed physical query graph G=(Ω,E)
                         │
                         ▼
         one shared node-potential decoder uθ
                         │
       frozen tracker: trajectory → swept nodes
                         │
                         ▼
      one exact multi-source min-plus envelope
                         │
                         ▼
 sound geometry gate + native-preserving arbitration
```

V0 只有一个新增可训练组件：共享节点势 decoder。edge scale 固定为物理尺度

\[
s_e=\ell_e/\ell_{ref}
\]

（等距图可取 (s_e=1)）。learned scale、候选 head、panel transformer、soft-min、anchor particle、路径近似均不属于 V0。

## 4. 训练目标

令 (D) 为任意定向的 incidence matrix。V0 使用 source GT field 的 graph-gradient loss：

\[
\mathcal L_{edge}
=\frac1{|E|}\sum_{e\in E}
\operatorname{Huber}\left(
\frac{(Du_\theta-Dd_S)_e}{s_e}
\right).
\]

由于 (G) 连通，(Du) 将节点势确定到唯一 additive constant；可用 (\sum_q u_q=0) 作数值 gauge，但它不改变任何决策量。

训练、早停、图、ROI、尺度、候选资格和阈值全部 source-only 冻结。target 数据不得用于选择。

## 5. Source-session calibration

模型与协议冻结后，对独立 source calibration session (Z_i) 定义

\[
E_i=\max_{t\in T(Z_i)}\max_{e\in E}
\frac{|(Du_{it}-Dd_{it})_e|}{s_e}.
\]

取

\[
k=\lceil(n+1)(1-\alpha)\rceil,
\qquad q_\alpha=E_{(k)}.
\]

若 (k>n)，系统返回不可部署。交换性成立时，新 source session 满足

\[
P\left(
\forall t\in T(Z),e\in E:
|(Du-Dd)_e|\le q_\alpha s_e
\right)\ge1-\alpha.
\]

这是一项 source-session 边际保证，不是任意 OOD 保证。全局 maximum 仍可能被远处坏边主导，这是必须实证检查的保守性风险。

## 6. Graph-quotient uncertainty set

令 (w_e=q_\alpha s_e)，定义

\[
\mathcal U(u,w)=
\{d: |D(d-u)|_e\le w_e,\ \forall e\in E\}.
\]

该集合沿共同常数方向无界，但所有 set-min contrasts 有限。它是标准 graph-difference/Lipschitz polytope；此对象本身不作为原创主张。

## 7. Exact robust set-min theorem

对任意非空 challenger set (A) 与 native set (B)：

\[
L_G(A,B)
=\inf_{d\in\mathcal U(u,w)}
[\min_A d-\min_B d].
\]

在有限连通无向图和有限非负边权下，精确闭式为

\[
L_G(A,B)
=\min_{a\in A}\max_{b\in B}
[u_a-u_b-\operatorname{dist}_w(a,b)].
\]

定义标准 min-plus envelope

\[
\psi_B(v)=\min_{b\in B}
[u_b+\operatorname{dist}_w(v,b)],
\]

则

\[
\boxed{L_G(A,B)=\min_{a\in A}[u_a-\psi_B(a)]}.
\]

因此每个 native 只需一次 multi-source Dijkstra，复杂度为

\[
O((|\Omega|+|E|)\log|\Omega|)
+O(\sum_j|A_j|).
\]

若 super-source 初始标签先减去 (c=\min_Bu)，求得的是 (\widetilde\psi_B=\psi_B-c)，最终必须恢复 (+c)。

结构性质：(A\subseteq B\Rightarrow L_G(A,B)\ge0)，(A=B\Rightarrow L_G=0)，候选的加入/删除/重排不改变固定 pair 的 bound。

## 8. Proof-carrying geometry gate

V0 只授权具有确定性 nominal-geometry 证书的 pair。checker 必须证明：

1. native 与 challenger 的离散采样点都位于各自连续 footprint 内；
2. challenger samples 是连续 footprint 的内部 (h_j)-net，且 (h_j) 是保守解析上界；
3. 两个完整 footprints 位于认证 ROI，集合非空；
4. footprint、pose interpolation、车辆膨胀、grid 与数值协议 hash 一致；
5. teacher distance 在 ROI 上满足 1-Lipschitz。

由此

\[
\Delta_{j0}^{cont}
\ge L_G(A_j,A_0)-h_j.
\]

无法证明资格的 pair 直接拒绝，不消耗 field 风险预算。若未来 checker 只有概率 soundness，(\beta) 必须上界整个 session 中至少一次授权 pair 几何误接受事件，而非单次调用错误率。

## 9. Arbitration and risk

候选先通过冻结的动力学、意图、进度和偏离预算。只有在

\[
L_G(A_j,A_0)-h_j\ge\gamma_{task}+\varepsilon_{num}
\]

时才可授权，其中 (\varepsilon_{num}) 上界最终计算可能产生的乐观数值误差。若无候选合格则保留 native；否则选择 deviation cost 最小者。

在同一个冻结时刻集 (T(Z)) 上：

\[
I(Z)=\max_{t\in T(Z)}
\mathbf1\{\text{replace at }t\},
\]

\[
H(Z)=\mathbf1\{\exists t\in T(Z):
I_t=1,
M_{0,t}^{cont}\ge0,
M_{sel,t}^{cont}<0\}.
\]

若 geometry checker 对 nominal footprints 确定性 sound，则

\[
P_{D_{cal},Z}(H=1)\le\alpha.
\]

若 session-level geometry failure 至多为 (\beta)，则上界改为 (\min\{1,\alpha+\beta\})。这不是逐干预风险、固定 calibration realization 的 PAC 保证或真实车辆安全保证。

## 10. 公平基线

### Uncertainty-only layer

先训练合法 absolute SDF predictor，冻结同一输出后比较：absolute simultaneous band、global gauge set、local graph set；三者都使用其 uncertainty set 上的 exact relative set-min readout。

### End-to-end layer

在相同 backbone、field architecture、source data、参数预算和仲裁规则下，各方法使用合法训练目标：absolute SDF、global gauge、local edge，以及一个具体冻结的 CDF-style low-rank joint residual 方法。

任何数值 infimum 都必须提供解析下界、dual certificate 或方向正确的优化误差预算；primal feasible objective 不能直接当安全下界。

## 11. Qualification gates

- **Gate A — 数学/几何:** closed form 与线性规划一致；覆盖 shift、非零标签、相等/包含集合、零权边、长链、环图；geometry checker 的保守半径与数值误差上界成立。
- **Gate B — source 非平凡性:** 使用独立 held-out source 数据和合法预测器。允许在 Gate B 内训练一个小型 source-only probe；不得用 GT 自拟合或只有 feature cache 却无 (u) 的 oracle 结果替代视觉 residual。
- **Gate C — meaningful non-inclusion:** 连续 footprints 的差异超过网格容差，GT relative margin 严格改善达到预注册 (\delta_M)，并保持相同资格、进度和偏离预算。
- **Gate D — candidate/planner transfer:** 在同一 field event 下测试未见候选策略与未参与 calibration 的 frozen planner；不把它写成 target appearance coverage。

Gate A/B 未通过，不得进入完整 NAVSIM 长跑。

## 12. 创新边界

当前被授权的结论只有：

> 已形成 source-session edge event ⇒ all-pair discrete robust bounds ⇒ sound continuous correction ⇒ adaptive nominal-road non-degradation 的条件理论链。

当前未被授权：系统已校准/有效/实时、任意 OOD 保证、优于 CDF/DriveReferee/scalar referee、learned scale 必要、顶会原创或“首次”。

只有独立查新、两层公平对照、meaningful non-inclusion 非平凡收益和实际成本门全部通过后，才能把该系统从严密的结构化应用升级为方法论文主贡献。失败时执行降级或停止条件，不追加模块挽救标题。
