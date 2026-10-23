# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem:** 同一物理场景只发生外观变化时，完整冻结的端到端驾驶 planner 可能输出显著不同甚至违反道路约束的轨迹；我们要从视觉中学习一个可跨冻结 planner 使用的任务接口，使干预依据与物理道路约束一致，而不是依赖全局特征不变性。
- **Must-solve bottleneck:** 绝对道路裕度的安全零点在视觉退化下可能不可辨识，但候选之间的相对道路可行性仍可能保留；方法必须明确回答在不知道绝对安全锚点时，哪些替换仍能被合法授权，并对 shape/局部残差、选择后风险和几何离散误差负责。
- **Non-goals:** 不重建完整 HD map；不修改或联合训练原 planner；不宣称道路可行性等于碰撞、交通规则或闭环系统安全；不使用目标域做训练、阈值、早停或方法选择；不承诺任意 OOD 分布无关保证；不把标准化、conformal、DINO、图最短路或 SDF 单独宣称为创新。
- **Constraints:** NAVSIM v1/v2；冻结 DINOv3-H+ 与被审计 planner；source-only 学习和校准；候选生成域、动力学资格和意图偏离预算必须预先冻结；大规模实验前先通过结构资格；主要新增可训练组件不超过两个。
- **Success condition:** 在独立日志上证明：相对场结构比绝对 margin 更稳定且可校准；集合对专属下界在预先声明的 source-session 风险对象上有效；它能在非包含候选上授权绝对场界或全域 gauge 界无法授权的、相对 native 不新增道路失败的干预；优势不由候选覆盖、目标域选择、更强 backbone 或纯几何包含关系解释。

## Anchor Check

- **Original bottleneck:** 视觉退化使绝对安全零点不可靠，而同一道路场对 native 与 challenger 产生共享、空间相关的误差。
- **Why the revised method still addresses it:** 新方法只约束道路场的局部差分，绝对 offset 落在图 incidence operator 的零空间中；决策直接求“所有与证据一致的场里 challenger 相对 native 的最坏 margin 差”。
- **Reviewer suggestions rejected as drift:** 不引入动态障碍预测、交通规则推理、VLM、闭环控制器重训或完整 HD map 重建。

## Simplicity Check

- **Dominant contribution:** Graph-Quotient Conformal Field（GQCF）：对共享物理网格上场残差的图梯度作 session 校准，并把图差分不确定集解析转换为候选扫掠集合对的精确最坏相对 margin。
- **Components retained:** 一个共享节点势 decoder；同一 decoder 的正边尺度输出；无参数 hard-min 与最短路鲁棒读出；一个 session quantile。
- **Components removed:** 全域 gauge sup-norm 半径、候选 scalar head、soft-min、activity attention、anchor particle、panel transformer、低秩残差生成器。
- **Why this is the smallest adequate route:** 节点势保证全局 cycle consistency；边差分消除共同 offset；边尺度允许不确定性保持局部；最短路公式直接把局部证据变成集合级最坏界。删除其中任何一个对象都会分别失去 gauge 不变、空间局部性或集合对专属证书。

## Changes Made

### 1. 从全域 residual range 改为局部图差分不确定集

- **Reviewer said:** GCMF 的全域 sup-norm 可被一个与所有候选无关的不可见 cell 主导，导致正确但无用的全部 abstain；普通 absolute field band 也能覆盖任意候选查询。
- **Action:** 在固定连通任务图上校准 edge residual，而不是校准整个场的最大 node residual range。
- **Impact:** 不确定性通过候选相关的图路径传播；远处坏边只有在连接相关扫掠集合的最短路径必须经过它时才进入证书。

### 2. 将 inclusion structure 纳入决策而非只用于表征叙事

- **Reviewer said:** 若 challenger 的 swept set 包含于 native，minitivity 本身已给出不退化，无需视觉；旧下界仍统一减去 (2q)，没有利用这一结构。
- **Action:** 直接对图差分不确定集合求最坏 set-min contrast，并推导闭式最短路公式。
- **Impact:** 当 (A_j\subseteq A_0) 时，下界自动非负；实验必须单独报告非包含干预，不能把纯几何免费证书包装成视觉方法收益。

### 3. 建立最强同源消融链

- **Reviewer said:** 必须区分 absolute field、仅 residual-range 校准、gauge training 和完整方法。
- **Action:** 冻结同一 backbone、node potential、任务网格和 teacher，只改变训练/校准/鲁棒读出，形成 absolute sup-norm、global residual range、fixed-edge GQCF、learned-edge GQCF 的嵌套比较。
- **Impact:** 如果性能仅来自 dense field 或共同 shift 消除，完整方法不能获得算法创新主张。

### 4. 增加非平凡性与实时性合同

- **Reviewer said:** 候选策略无关的保证若导致零干预没有部署价值；图算法也必须有明确运行合同。
- **Action:** 大规模训练前先比较 source oracle headroom 与图路径惩罚；精确 Dijkstra 是参考实现，任何快速近似只能使用路径长度上界以保持保守性。
- **Impact:** 理论正确、证书锐度和运行代价成为三个独立通过条件。

---

# Research Proposal: AORTA-v8.2 — Graph-Quotient Conformal Fields for Candidate-Independent Road-Margin Arbitration

## 1. Research Thesis

### 1.1 One-sentence thesis

学习一个共享坐标的道路 clearance 节点势，只校准其局部图差分误差；随后在该图商空间不确定集中精确求解任意两条扫掠集合的最坏相对 margin，从而得到候选策略无关、集合结构感知的保守仲裁证书。

### 1.2 Precise scientific claim

本研究不声称视觉可以恢复绝对道路安全零点。主张是一个可证伪的条件命题：

> 若 source session 上学习到的局部 clearance 差分可被交换性校准，并且新 source session 落入该场级覆盖事件，则同一任务图内任何自适应产生的 challenger/native 扫掠集合对，都可以通过图最短路公式获得正确的相对道路 margin 下界。

外观 OOD 是否仍满足该事件是实证问题，不由 conformal 理论保证。

## 2. Technical Gap

已有路线分别解决了问题的一部分：

1. scalar relative scorer 可直接校准候选差值，但覆盖对象绑定候选生成过程；
2. dense map 或 conformal distance field 可产生共享场与绝对约束带，但常用全域绝对误差会被共同 offset 和无关坏区域主导；
3. global gauge residual range 消除了共同 offset，却仍给所有候选对扣除同一最坏半径；
4. minitive field 保证集合包含单调性，但若决策规则不利用它，理论结构不会转化为更紧证书。

缺少的对象不是又一个网络模块，而是：**一个定义在场商空间上的局部不确定集，以及从该集合到 set-min contrast 的候选对专属精确鲁棒变换。**

## 3. Mathematical Objects

### 3.1 Fixed task graph

令

\[
G=(\Omega,E),\qquad \Omega=\{q_1,\ldots,q_N\}
\]

为预先冻结的连通 ego-centric 道路任务图。节点是物理网格 cell，边仅连接预先声明的空间邻居；ROI、分辨率、连接方式和时间协议均不得由目标域结果选择。

GT drivable-area signed-distance field 为 (d_S:\Omega\to\mathbb R)，正值在道路可行侧。冻结 tracker 将任意合格轨迹变为非空 swept-cell set (A_\Omega(\tau)\)。离散道路 margin 为

\[
M_S^\Omega(\tau)=\min_{q\in A_\Omega(\tau)}d_S(q)-m_{\rm buf}.
\]

相对比较时固定 buffer 抵消。

### 3.2 Node potential and graph quotient

网络输出节点势

\[
u_\theta(X,q),\qquad q\in\Omega,
\]

以及正的 edge evidence scale

\[
s_\theta(X,e)\in[s_{\min},s_{\max}],\qquad e\in E.
\]

令 (D\in\mathbb R^{|E|\times|\Omega|}) 为任意定向的 incidence matrix。因为

\[
D(u+c\mathbf1)=Du,
\]

共同 offset 精确落入 (D) 的零空间。图连通时，

\[
\ker D=\operatorname{span}\{\mathbf1\},
\]

所以 edge differences 保留了除共同 offset 外的全部节点势信息。这是代数事实，不等于视觉上一定可辨识；遮挡与投影缺失仍可能使 edge residual 变大。

### 3.3 Graph-seminorm uncertainty set

给定非负半径 (q)，定义

\[
\mathcal U_q(u,s)
=\left\{
d:\ |D(d-u)|_e\le q s_e,\ \forall e\in E
\right\}.
\]

该集合沿 (mathbf1) 方向无界，但所有 set-min contrasts 有限且 gauge invariant。它不是逐 cell 独立 box；节点误差只能沿图路径累积，因此保留空间相关结构。

## 4. Exact Robust Set-Contrast Theorem

令 (A,B\subseteq\Omega) 为任意非空集合，定义真实相对 margin

\[
\Delta_d(A,B)=\min_{a\in A}d_a-\min_{b\in B}d_b.
\]

边权为

\[
w_e=q s_e,
\]

并令 (operatorname{dist}_w(a,b)) 为 (G) 上的加权最短路距离。

### Theorem 1 — Exact graph-quotient lower bound

\[
\boxed{
L_G(A,B;u,s,q)
=\min_{a\in A}\max_{b\in B}
\left[u_a-u_b-\operatorname{dist}_w(a,b)\right]
}
\]

精确等于

\[
\inf_{d\in\mathcal U_q(u,s)}\Delta_d(A,B).
\]

#### Proof sketch

写 (d=u+z)。首先

\[
\Delta_d(A,B)
=\min_{a\in A}\max_{b\in B}(d_a-d_b).
\]

由于 (A) 有限，外层对 (z) 与 (a) 的 infimum 可合并。固定 (a) 并令 (z_a=0)。图差分约束推出

\[
z_b\le\operatorname{dist}_w(a,b).
\]

另一方面，(z_v=\operatorname{dist}_w(a,v)) 满足每条边的 Lipschitz 约束，并同时达到所有这些上界。因此固定 (a) 的最优值为

\[
\max_{b\in B}
(u_a-u_b-\operatorname{dist}_w(a,b)),
\]

再对 (a\in A) 取 minimum 即得结论。

### Corollary 1 — Inclusion is recovered for free

若 (A\subseteq B)，则对每个 (a\in A)，内层 maximum 包含 (b=a) 项，其值为 0，故

\[
L_G(A,B)\ge0.
\]

这与任意场上 (min_A d\ge\min_B d) 完全一致。方法不得将这类干预的成功归因于视觉预测；它们属于 geometry-only baseline。

### Corollary 2 — Candidate insertion invariance

(u,s) 在候选生成前计算。加入、删除或重排其他候选不会改变任一固定集合对的 (L_G(A,B))。该性质是算法不变性；统计覆盖还需要下一节的 field event。

### Conservative fast approximation

若运行时使用任意可行路径长度 (widetilde d(a,b)\ge\operatorname{dist}_w(a,b))，则

\[
\widetilde L_G
=\min_a\max_b[u_a-u_b-\widetilde d(a,b)]
\le L_G.
\]

所以可用预定义路径库或 GPU wavefront 得到保守近似；不得使用未经证明的低估距离近似。精确 Dijkstra/shortest-path 版本是论文参考实现和数值审计基准。

## 5. Architecture

```text
camera history + intrinsics/extrinsics + ego poses
                         │
                         ▼
              frozen DINOv3-H+ tokens
                         │
          fixed physical query graph G=(Ω,E)
                         │ project/sample local evidence
                         ▼
       shared graph-field decoder (one component)
             ├── node potential uθ(X,q)
             └── positive edge scale sθ(X,e)
                         │
       frozen tracker: τ ↦ nonempty AΩ(τ)
                         │
                         ▼
 exact/conservative graph robust transform LG(Aj,A0)
                         │
      qualification + least-deviating certified choice
```

### 5.1 Shared field decoder

每个物理节点经可部署相机几何投影到多尺度 frozen tokens；采样特征、坐标、时间、visibility mask 和 source-only context 进入轻量共享 graph decoder。节点分支输出 (u_q)。边尺度由两端共享 embedding 和边方向/长度产生，不读取候选身份。

这是一个 decoder 的两个输出，不是两个独立大模块。节点势而非独立 edge prediction 保证 (Du) 自动 cycle-consistent；尺度只描述误差容许度，不改变势的一致性。

### 5.2 Evidence-state / task-state separation

- **Task state:** (u) 表示相对 clearance potential，供所有轨迹共享读取。
- **Evidence state:** (s) 表示边差分证据的不稳定程度，只影响 uncertainty set。

不得把 visibility、特征相似度或 scale 本身解释为道路状态。尺度退化只应使证书更保守。

## 6. Training Objective

对 source observation 和 GT field，定义 edge residual

\[
\rho_e=(D u_\theta-D d_S)_e.
\]

使用受限 Laplace score：

\[
\mathcal L_{\rm edge}
=\frac1{|E|}\sum_{e\in E}
\left(\frac{|\rho_e|}{s_e}+\log s_e\right).
\]

训练时固定 (s_{\min},s_{\max})，并对每帧的 (log s) 去除全局均值或施加 (rac1{|E|}\sum_e\log s_e=0) 约束，防止 (u,s,q) 的无意义整体缩放。该 NLL 与尺度头均不是创新；它们只是构造局部有效、可校准的 uncertainty geometry。

只使用 edge supervision 已对连通图上的节点场确定到一个 additive constant，正好匹配任务商空间。可选的数值 gauge 固定 (sum_q u_q=0) 不改变任何输出 contrast。

训练、模型选择和尺度裁剪边界只用 source train/development。目标域不参与阈值、早停或架构选择。

## 7. Source-Session Calibration

冻结模型后，对独立 source calibration session (Z_i) 定义

\[
E_i=max_{t\in T(Z_i)}\max_{e\in E}
\frac{|(D u_{it}-D d_{it})_e|}{s_{it,e}}.
\]

分数非负。取

\[
k=\lceil(n+1)(1-\alpha)\rceil,
\qquad q_\alpha=E_{(k)}.
\]

若 (k>n)，协议返回“样本不足、不可部署”。在 calibration sessions 与新 source session 交换时，split conformal 给出

\[
P\left(
\forall t\in T(Z_{n+1}),\forall e\in E:
|(D u-D d)_e|\le q_\alpha s_e
\right)\ge1-\alpha.
\]

在该单一场事件上，Theorem 1 同时覆盖该时刻所有非空 (A,B\subseteq\Omega)，包括看到 (u,s) 后由任意 frozen planner 自适应产生的候选。统计量词不包含候选生成策略，但仍包含 source observation/session 分布；它不自动覆盖目标域、闭环状态分布变化或新的传感器协议。

即使尺度头严重 misspecified，split conformal 的边际覆盖仍由校准分数恢复；misspecification 会损害的是锐度而非名义 source coverage。为防止“尺度 gaming”被效率指标掩盖，必须同时报告 scale clipping rate、分位数 (q_\alpha)、真实 edge coverage 与未标准化 edge residual。

## 8. Geometry and Decision Contract

### 8.1 Discrete certificate

对 challenger (j) 与 native (0)：

\[
L_{j0}^\Omega=L_G(A_\Omega(\tau_j),A_\Omega(\tau_0);u,s,q_\alpha).
\]

若 (L_{j0}^\Omega\ge0)，则在覆盖事件上

\[
M_S^\Omega(\tau_j)\ge M_S^\Omega(\tau_0).
\]

因此它认证的是 relative non-degradation，而非 challenger 绝对安全。只有当 native 的道路 margin 本来非负时，才可推出 challenger 不产生新的离路失败。

### 8.2 Continuous footprint ledger

| 层级 | 对象 | 覆盖状态 |
|---|---|---|
| GT graph field | (Dd_S) on fixed (E) | session calibration 直接覆盖 |
| Discrete swept set | (A_\Omega(\tau)) | Theorem 1 直接覆盖 |
| Continuous nominal footprint | (mathcal B(\tau)) | 仅在声明的 rasterization/Hausdorff 条件下 |
| Executed footprint | 含控制与动力学误差 | 不覆盖，除非另有 tracking bound |
| NAVSIM official DAC/PDMS | 官方 scorer | 独立 fidelity/downstream audit，不与证书混同 |

若 (A_\Omega(\tau_j)) 是 challenger 连续 footprint 的内部 (h_j)-net，且 (d_S) 为 1-Lipschitz，则保守使用

\[
L_{j0}^{cont}=L_{j0}^\Omega-h_j.
\]

若只有双边 Hausdorff 误差，减去 (eta_j+eta_0)。同一几何误差不能既写入 field target 又重复扣除。

### 8.3 Arbitration

候选先经过完全冻结的动力学、命令一致性、意图走廊和偏离预算资格层。定义

\[
\mathcal A_t=\{j:L_{j0,t}^{cont}\ge\gamma\},\qquad\gamma\ge0.
\]

集合为空则保留 native；否则选择 source-development 阶段冻结的 deviation cost 最小者。不得用目标域结果调 (gamma)。

## 9. Risk Contract

对随机 session (Z)：

- (I(Z)=1)：session 中至少干预一次；
- (H(Z)=1)：session 中至少一次出现 native 离散/约定连续道路 margin 非负，但替换轨迹对应 margin 为负。

在 field coverage event 上，所有 (L\ge0) 的干预都 relative non-degrading，因此 (H=0)。故

\[
P_{D_{cal},Z}(H=1)\le\alpha.
\]

条件风险只能由

\[
P(H=1\mid I=1)
\le\min\{1,\alpha/P(I=1)\}
\]

上界。该式不构成逐帧、逐次干预、固定 calibration realization 的 PAC 保证，也不构成碰撞或闭环安全保证。

## 10. Why the Graph Object Is Necessary

### 10.1 Versus scalar contrast calibration

scalar baseline 的 session score 对候选过程取 maximum；候选策略改变时随机对象改变。GQCF 校准 candidate-free edge event，随后同时覆盖图上所有 swept sets。

### 10.2 Versus absolute field sup-norm

absolute band 要为不可辨识共同 offset 付费，并把无关 cell 的最大误差传给所有候选。GQCF 不约束 offset，且只通过连接具体 (A,B) 的图路径累积局部误差。

### 10.3 Versus global gauge residual range

global gauge band 对所有候选统一扣除 residual range。GQCF 的惩罚为 pair-specific shortest-path metric；集合包含时自动得到零或正证书，局部不确定边可被其他低代价路径绕开。

### 10.4 Versus generic joint/low-rank CDF

generic joint field 可表达相关误差，但若最终仍用绝对 cellwise envelope 或全域范数，它不会自动给出上述 exact set-contrast transform。GQCF 的候选贡献是“graph quotient uncertainty set + exact robust min-difference operator”的组合，而非 dense field 或 conformal 本身。

这仍是待查新的候选创新；graph total variation、difference constraints、robust shortest paths、conformal fields 或 runtime assurance 文献若已给出等价对象，必须降级主张。

## 11. Structural Qualification Before Full Experiment

不得直接开始大规模 NAVSIM 训练。先用 source GT 与已有冻结特征完成三个资格检查：

1. **Identifiability:** edge residual 是否显著小于 absolute residual，且不是只靠全局 shift；遮挡边是否可由 (s_e) 合理隔离。
2. **Nontriviality:** 对 held-out source 候选，(u_a-u_b) 的 headroom 是否经 (q_\alpha s) 路径惩罚后仍产生非零证书。
3. **Beyond inclusion:** 至少有预先规定比例的 certified interventions 满足 (A_j\nsubseteq A_0)；否则收益只是 geometry-only minitivity。

任一失败即停止完整训练或缩小主张，不通过换目标域阈值挽救。

## 12. Claim-Driven Minimal Validation

### Claim 1 — GQCF 给出有效、候选策略无关且比全域界更紧的 contrast certificate

- **Setup:** 同一 frozen DINO、field decoder capacity、任务图、GT teacher 和 source split；测试原候选策略、未见候选扰动策略、未参与校准的 frozen planner。
- **Nested baselines:** absolute node field + sup-norm；同一 node field + optimal-shift residual range；gauge-trained node field + residual range；GQCF fixed (s=1)；GQCF learned (s)；scalar contrast + candidate-session calibration；可实现时加入 CDF-style joint field。
- **Metrics:** source-session edge-event coverage、所有查询 contrast coverage、bound sharpness、certified pair/intervention rate、跨候选策略 degradation、scale clipping 与运行时间。
- **Decisive evidence:** GQCF 保持预声明 coverage，在非包含候选上显著紧于 global gauge/absolute field，且候选策略变化不破坏 coverage；否则结构贡献失败。

### Claim 2 — 精确集合对证书支持低新增道路失败的有效仲裁

- **Setup:** 固定相同候选、资格层、deviation budget 和选择顺序。
- **Baselines:** keep-native；geometry-only inclusion；uncalibrated relative ranking；scalar calibrated；absolute box；global gauge GCMF；DriveReferee-style relative map rule；GQCF。
- **Metrics:** session (H)、intervention session rate、非包含干预率、真实 relative margin、oracle headroom、DAC/NC/EP/comfort/PDMS 分项。
- **Decisive evidence:** GQCF 在 source 风险对象上覆盖合格，提供非零非包含干预，并在 held-out candidate policy/planner 上比同容量 baselines 更稳定；若主要收益来自包含候选，视觉算法主张不成立。

## 13. Failure Modes and Stop Conditions

- **Local contrast 仍不可辨识:** 校准 (q_\alpha) 过大或关键边尺度普遍饱和，导致非包含证书为空。
- **Path accumulation 过保守:** 图距离随路径长度累积，远距离集合对全部 abstain；若 global gauge 更紧，停止图路线。
- **Graph misspecification:** 固定邻接无法表达道路拓扑或跨障碍“捷径”污染距离；图必须在 source-only 协议中修正，不能由 target 结果挑选。
- **Scale head 无效:** learned (s) 不优于 (s=1)，删除尺度头，保留 fixed-edge 版本。
- **Dense field 已足够:** absolute/gauge field baseline 在相同 coverage、transfer、复杂度上等价，撤回 GQCF 必要性。
- **等价先行工作:** 若现有 conformal structured prediction 或 graph-difference uncertainty 已给出同一 robust set-min 变换，改为增量应用贡献或停止。
- **下游指标退化:** 即使道路 margin 不退化，碰撞、规则、进度或舒适性显著下降，则系统无部署价值。

## 14. Novelty Boundary

不单独宣称以下内容为创新：冻结 DINO、SDF、edge NLL、incidence matrix、split conformal、shortest path、hard min、fallback。

唯一主贡献候选是：

> 将道路 clearance 估计表述为 graph additive-quotient field，使用 session-calibrated local difference polytope 表达视觉证据，并推导对任意 swept-set pair 精确、可保守近似的 robust min-margin transform，用于 frozen-planner-independent relative arbitration。

其价值必须同时通过四个门：数学精确、source coverage 正确、非包含证书非平凡、强同源 baseline 不可替代。任何一项未通过，都不得写成顶会级核心创新。

## 15. Compute and Handoff

- **理论/反例:** 先用枚举小图和线性规划逐例核对 Theorem 1、inclusion、近似保守方向；CPU，1–2 天。
- **结构资格:** 用 GT field 与 frozen feature cache 比较 absolute/range/edge residual；约 1–2 GPU-days 的特征提取加 CPU 图统计。
- **最小模型:** 一个共享 decoder，先 fixed scale，再决定是否训练 scale；3 seeds 预计 60–180 GPU-hours，实际预算由网格、相机、历史长度冻结后确定。
- **校准规模:** 至少 99 个真正独立 session 才能得到有限 99% split-conformal threshold；帧、corruption copy 和同 route 片段不得冒充独立 session。
- **实验交接输入:** 本文只定义假设、算法、定理、停止条件和两项最小验证；具体 run order、功效和资源排程交给 experiment-plan，不在 refine 阶段启动实验。
