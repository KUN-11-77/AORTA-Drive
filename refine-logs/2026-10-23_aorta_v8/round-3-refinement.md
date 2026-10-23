# Round 3 Refinement

## Problem Anchor

- **Bottom-line problem:** 同一物理场景只发生外观变化时，完整冻结的端到端驾驶 planner 可能输出显著不同甚至违反道路约束的轨迹；我们要从视觉中学习一个可跨冻结 planner 使用的任务接口，使干预依据与物理道路约束一致，而不是依赖全局特征不变性。
- **Must-solve bottleneck:** 绝对道路裕度的安全零点在视觉退化下可能不可辨识，但候选之间的相对道路可行性仍可能保留；方法必须明确回答在不知道绝对安全锚点时，哪些替换仍能被合法授权，并对局部残差、选择后风险、候选自适应性和几何离散误差负责。
- **Non-goals:** 不重建 HD map；不修改或联合训练原 planner；不宣称道路可行性等于碰撞、交通规则或闭环安全；不以 source conformal 冒充任意 OOD 保证；不把 DINO、SDF、split conformal、difference constraints、Dijkstra 或 min-plus transform 单独宣称为创新。
- **Constraints:** NAVSIM v1/v2；冻结 DINOv3-H+ 与 planner；source-only 学习、选择和校准；候选域与几何协议预先冻结；大规模实验前先过结构资格；主模型只有一个共享场 decoder，learned scale 只能作为可删除扩展。
- **Success condition:** 在严格同源对照下，局部差分场接口必须在预声明 source-session 风险对象上有效，并在具有连续几何意义的非包含候选上，比 scalar、absolute field、global-gauge exact readout 和最强联合场基线产生更紧且有用的相对 margin 证书；否则降级为结构化应用或停止。

## Anchor Check

- **Why this remains on problem:** 学习、校准和仲裁对象始终是同一道路 clearance 约束及 challenger-vs-native 非退化，不加入新的驾驶任务。
- **What is narrowed:** 不再把“新图算法”作为目标；标准 min-plus 求解只服务于原始视觉鲁棒仲裁问题。
- **No drift additions:** 不引入 VLM、动态障碍、交通规则、控制器训练或目标域适配。

## Simplicity Check

- **Primary model:** 一个 candidate-free 节点势 decoder；固定物理 edge scale 为主版本。
- **Primary calibration:** 一个 source-session 最大标准化 edge residual quantile。
- **Primary readout:** 每个 native 一次标准多源距离变换，所有 challenger 只做集合 reduction。
- **Optional extension:** learned positive edge scale；只有在 fixed metric 版本通过后且显著改善米制锐度时保留。
- **Removed:** 逐帧 geometric-mean normalization、逐候选最短路、默认路径近似、独立 uncertainty 网络、候选 head 与全域统一 (2q) 扣项。

## Reviewer Findings and Repairs

### 1. 标准归约不是创新

- **Finding:** 精确式可写成一次多源 min-plus 距离变换，是 difference-constraint Lipschitz envelope 的标准读出。
- **Repair:** 明确列入 non-contribution；主贡献候选改为针对视觉道路仲裁的完整 estimand/uncertainty/selection 对齐，以及它相对 candidate-bound scalar 和 absolute field 的可证伪能力差异。

### 2. 连续证书遗漏 native 采样方向

- **Finding:** 仅 challenger 是内部 (h)-net 不足；native 若含 footprint 外点，可产生假证书。
- **Repair:** 单侧界强制两条轨迹的离散采样都位于各自连续 footprint 内，且 challenger 是内部 (h_j)-net；否则只允许双边 Hausdorff 预算或拒绝认证。

### 3. 逐帧尺度归一化删除整体观测质量

- **Finding:** 每帧令 mean-log-scale 为零不能由全局 (s\leftrightarrow q) 等价辩护，并会让某些边变得更乐观。
- **Repair:** 主版本采用冻结物理尺度；learned-scale 扩展不做逐帧归一化，只做 source-frozen positivity/clipping。任何全局数值重标定只能是 calibration 前对所有样本、所有边相同的常数。

### 4. 图局部性被夸大

- **Finding:** 远处坏边仍可能通过全局 session maximum 抬高 (q)，并放大所有边。
- **Repair:** 只声称“给定最终边预算后的路径局部性”；全局 (q) 耦合和长路径累积明确列为主要失效模式，不再作为结构性已解决问题。

### 5. 最强 baseline 使用同样精确 readout

- **Finding:** 若 global gauge baseline 仍扣统一 (2q)，比较不公平。
- **Repair:** 所有 joint uncertainty baselines 必须直接求其 uncertainty set 上的 exact robust set-min；CDF-style joint field 不再是可选对照。

---

# AORTA-v8.3 — Calibrated Minitive Field Envelopes for Frozen-Planner Arbitration

## 1. Contribution Contract

### 1.1 Candidate contribution

给定一个冻结视觉 backbone 和冻结 planner，AORTA-v8.3 建立一个 candidate-free 的道路差分证据接口：先对共享物理图上的场差分误差作 session-level simultaneous calibration，再对任意自适应 swept-set pair 计算 uncertainty-set-exact 的相对 margin 下界，最后只在该下界认证 non-degradation 时替换 native。

### 1.2 What is genuinely claimed

若最终成立，论文贡献是以下对齐，而不是任何单独组件：

1. **Estimand alignment:** 只估计道路 margin 排序所需的 additive-quotient field，而非不可辨识的绝对零点；
2. **Uncertainty alignment:** 校准与候选无关的共享场事件，而非绑定某个 candidate generator 的有限 score list；
3. **Decision alignment:** 对同一 uncertainty set 直接求 worst-case set-min contrast，不把 cellwise coverage、平均 edge coverage 或松弛的 (2q) 当决策证书；
4. **Selection alignment:** 在同一覆盖事件上允许自适应候选选择，不追加未经证明的多重比较豁免。

这可以构成一个方法级系统贡献，但不是自动成立的“新数学理论”。只有当最强 joint-field baselines 在同一 exact readout 下仍不能取得等价锐度、迁移和成本时，才保留方法创新主张。

### 1.3 Standard tools explicitly not claimed

Incidence matrix、additive gauge、edgewise ℓ∞ band、split conformal、shortest-path Lipschitz extension、multi-source Dijkstra、hard minimum 与 fallback 都是标准工具。

## 2. Discrete Task Model

令 (G=(\Omega,E)) 是有限、连通、无向的固定物理网格图；incidence matrix 的方向仅为记号。边长 (ell_e>0) 来自冻结网格。GT signed-distance field (d_S:\Omega\to\mathbb R)。冻结几何接口把合格轨迹 (	au) 变为非空节点集 (A(\tau)\subseteq\Omega)，并保证全集落在认证 ROI。

\[
M_S^\Omega(\tau)=\min_{q\in A(\tau)}d_S(q)-m_{\rm buf}.
\]

主模型输出 candidate-free 节点势 (u_\theta(X,q))。共同 shift 不改变任何相对 margin。

## 3. Uncertainty Geometry

### 3.1 Primary fixed-metric version

主版本使用冻结尺度

\[
s_e=\ell_e/\ell_{ref},
\]

或在等距网格上 (s_e=1)。它避免把 learned uncertainty 的收益混入核心结构验证。定义

\[
\mathcal U_q(u)=
\{d: |D(d-u)|_e\le q s_e,\ \forall e\in E\}.
\]

### 3.2 Optional evidence-adaptive version

若 fixed-metric 版本先通过资格，允许同一 decoder 输出

\[
s_\theta(X,e)=s_{min}+\operatorname{softplus}(v_\theta(X,e)),
\]

并在 source-train 预先冻结 (s_{max}) 后截断。没有逐帧中心化。退化观测可以使全图尺度整体增大；局部尺度也可以变化。

只存在一个无关紧要的**全数据常数**重标定：若所有样本所有边的 (s) 同乘常数，calibration (q) 反向缩放而最终 (qs) 不变。每观测的整体尺度是有意义的证据信息，不能归一掉。

固定 (u,q) 时，若最终 (s'_e\ge s_e) 对每条边成立，则 uncertainty set 扩大，所有 robust lower bounds 不增。这一条件单调性不表示网络在任意 corruption 下必然输出逐边更大尺度。

## 4. Training

固定尺度主版本只训练 graph-gradient residual：

\[
\mathcal L_{fixed}
=|E|^{-1}\sum_e
\operatorname{Huber}\left(
\frac{(Du-Dd_S)_e}{s_e}
\right).
\]

learned-scale 扩展使用

\[
\mathcal L_{scale}
=|E|^{-1}\sum_e
\left[
\frac{|(Du-Dd_S)_e|}{s_e}+\log s_e
\right].
\]

网络输出节点势，故 (Du) 自动 cycle-consistent。图连通时 edge supervision 将 (u) 确定到唯一共同常数；可用 (sum_q u_q=0) 作为数值 gauge，不改变任务输出。

不能用“raw edge MAE 小于 node MAE”证明结构更可学习，因为 edge residual 随网格长度自然缩小。资格只比较按物理长度标准化的 held-out residual 与最终米制 contrast bound。

## 5. Session Calibration

对 calibration 前完全冻结的模型、图、尺度协议和 session time set：

\[
E_i=max_{t\in T(Z_i)}\max_{e\in E}
\frac{|(Du_{it}-Dd_{it})_e|}{s_{it,e}}.
\]

令 (k=\lceil(n+1)(1-\alpha)\rceil)。若 (k>n) 则不可部署；否则 (q_\alpha=E_{(k)})。在 source sessions 交换时：

\[
P\left(
\forall t,e:
|(Du-Dd)_e|\le q_\alpha s_e
\right)\ge1-\alpha.
\]

这是 calibration/test 联合随机性的 source-session 边际保证。它不覆盖：任意 appearance OOD、固定 calibration realization 的条件风险、未声明时刻、修改后的图/尺度/相机协议或真实闭环状态分布。

全局 maximum 的代价必须显式承认：一个远处且未被尺度吸收的坏边可以抬高 (q_\alpha)，进而放大所有边预算。图结构只让**最终预算 (w_e=q_\alpha s_e) 给定之后**的误差传播依赖相关路径；它不消除 calibration 的全局耦合。

## 6. Exact Robust Readout

设 (w_e=q_\alpha s_e\ge0)，(operatorname{dist}_w) 是无向图加权最短路。对非空 (A,B\subseteq\Omega)：

\[
L_G(A,B)
=\inf_{d\in\mathcal U_q(u)}
\left[\min_A d-\min_B d\right]
=\min_{a\in A}\max_{b\in B}
[u_a-u_b-\operatorname{dist}_w(a,b)].
\]

证明条件明确为：有限连通无向图、有限非负权重、非空集合。固定一个 gauge 后可行 polytope 紧，最优值可达到；distance potential 同时实现所需极值。

### 6.1 Multi-source min-plus form

对 native set (B) 定义

\[
\psi_B(v)=\min_{b\in B}
[u_b+\operatorname{dist}_w(v,b)].
\]

则

\[
\boxed{L_G(A,B)=\min_{a\in A}[u_a-\psi_B(a)]}.
\]

每个 native 只需一次 multi-source Dijkstra，初始标签为 (u_b)；统一减去 (min_Bu) 可转为非负 super-source 边。之后所有 challenger 只做集合 reduction。

参考复杂度：

\[
O((|\Omega|+|E|)\log|\Omega|)
+O(\sum_j|A_j|),
\]

内存 (O(|\Omega|+|E|))。第一版固定一个 native reference、4-neighbor 图和配置中冻结的 (|\Omega|)；在实际 profiler 证明需要前，不引入近似路径算法。

### 6.2 Structural laws

- 若 (A\subseteq B)，则 (L_G(A,B)\ge0)；
- 若 (A=B)，则 (L_G(A,B)=0)；
- 候选加入、删除、重排不改变固定 pair 的 bound；
- coverage event 同时覆盖所有自适应非空 pairs，无需按候选数修正。

inclusion 收益完全来自几何，必须作为独立 baseline，不计作视觉方法发现。

## 7. Continuous Geometry Contract

记连续 nominal swept footprint 为 (mathcal B(\tau))，连续 margin 为

\[
M^{cont}(\tau)=\inf_{x\in\mathcal B(\tau)}d_S(x)-m_{buf}.
\]

### 7.1 Qualified internal-sampling route

单侧扣项只在以下条件全部成立时允许：

1. (A_j,A_0) 的所有物理采样点分别位于 (mathcal B_j,mathcal B_0) 内；
2. challenger (A_j) 是 (mathcal B_j) 的内部 (h_j)-net；
3. (d_S) 在覆盖区域 1-Lipschitz；
4. 两个完整 footprints 均位于认证 ROI，集合非空，姿态/车体膨胀协议冻结。

于是

\[
M_j^{cont}\ge M_j^\Omega-h_j,
\qquad
M_0^{cont}\le M_0^\Omega,
\]

从而

\[
\Delta_{j0}^{cont}\ge L_G(A_j,A_0)-h_j.
\]

### 7.2 General symmetric route

若无法保证内部采样方向，只在预先验证的双边 Hausdorff budgets (eta_j,eta_0) 下使用

\[
\Delta_{j0}^{cont}
\ge L_G(A_j,A_0)-\eta_j-\eta_0.
\]

没有任一路径资格则该 pair 不可认证。ROI、nonempty、geometry qualification 是仲裁入口硬门，而非事后备注。

## 8. Arbitration and Risk

候选先通过冻结的动力学、命令一致性、意图走廊、进度与偏离预算。对通过连续几何资格的 pair 定义

\[
L_{j0}^{cert}=
\begin{cases}
L_G-h_j,&\text{internal route},\\
L_G-\eta_j-\eta_0,&\text{symmetric route}.
\end{cases}
\]

若 (L_{j0}^{cert}\ge\gamma\ge0)，才进入可替换集合；为空保留 native，否则选择 deviation cost 最小者。

正式失败事件唯一采用连续 nominal road margin：

\[
H(Z)=1
\iff
\exists t:\ I_t=1,
M_0^{cont}\ge0,
M_{sel}^{cont}<0.
\]

在 source-session field coverage event 与 geometry qualification 同时成立时，所有被授权选择 relative non-degrading，故 (H=0)。因此在几何协议确定性有效的前提下

\[
P_{D_{cal},Z}(H=1)\le\alpha.
\]

这仍不是碰撞、规则、执行误差或任意 OOD 安全保证。条件于发生干预的风险最多写为

\[
P(H=1\mid I=1)\le\min\{1,\alpha/P(I=1)\}.
\]

## 9. Exact Strong Baselines

所有结构化 baseline 都必须从各自 uncertainty set **直接求 exact worst-case set-min contrast**：

1. scalar pair contrast + candidate-process session calibration；
2. absolute node-field simultaneous band + exact robust set-min；
3. global additive-gauge oscillation set + exact robust set-min；
4. CDF-style low-rank/joint residual field + direct relative robust optimization；
5. local graph fixed physical scale + multi-source exact readout；
6. local graph learned scale + 同一 readout；
7. geometry-only continuous inclusion。

通用 difference-constraint solver 与 multi-source closed form 应在小图上逐例一致；它们是正确性互证，不是两个统计 baseline。

## 10. Qualification and Decisive Tests

### Gate A — mathematical and geometry audit

- 小图枚举/线性规划与 closed form 一致；
- (A=B)、(A\subset B)、零权边、长链和环图反例通过；
- internal 与 symmetric rasterization 条件被实现为硬检查；
- 浮点容差并入预冻结 (gamma)。

### Gate B — source structural nontriviality

只用合法 held-out source predictor residual，比较最终**米制 pair bounds**，不得比较 raw edge/node MAE。fixed graph 至少必须在同一 coverage 下对一部分非包含 pair 紧于 exact global-gauge baseline；否则 local graph 无必要性。

### Gate C — meaningful non-inclusion

成功事件必须同时满足：

- 连续 footprints 在超过网格容差的区域上不包含；
- GT relative margin 严格改善至少预注册 (delta_M>0)；
- 相同 qualification、progress 和 deviation budgets；
- 不是仅增加/删除一个边界 raster cell。

### Gate D — held-out candidate/planner transfer

同一 field calibration event 下测试未见候选策略和未参与 calibration 的 frozen planner。只检验 candidate-independence，不把该结果写成 target appearance coverage。

## 11. Two Minimal Validation Blocks

### Block 1 — uncertainty geometry and exact certificate

在同一 backbone、node predictor、teacher、grid 和 source split 下比较第 9 节全部 baselines。报告 session simultaneous coverage、米制 bound gap、meaningful non-inclusion certification、全局 (q) 对远处异常边的敏感度、路径长度分层锐度、scale clipping 与一次 multi-source latency。

**Falsifier:** local graph 在 exact-readout 公平比较下不优于 global gauge/joint field，或只在短距离/包含 pair 上有效。

### Block 2 — conservative arbitration utility

固定相同候选、资格层、选择规则和 deviation budget，比较 keep-native、geometry-only、scalar、global gauge、joint field 与 AORTA。报告 session (H)、intervention rate、meaningful non-inclusion success，以及 DAC/NC/EP/comfort/PDMS 分项。

**Falsifier:** 名义 coverage 不成立、有效干预近零、收益来自纯包含几何，或其他驾驶指标显著下降。

## 12. Innovation Boundary and Paper Position

当前可以安全表述为：

> 一个针对冻结端到端驾驶 planner 的、候选无关且选择后有效的结构化道路 margin 仲裁接口；其场级差分不确定集可用标准 min-plus envelope 精确读取。

只有在 Gate B–D 和系统查新都通过后，才升级为：

> 相比现有 conformal distance fields 与 scalar referee，首次将 additive-quotient local field coverage、uncertainty-set-exact set-min contrast 和 frozen-planner adaptive arbitration 统一，并在相同 exact readout 下取得不可替代的证书锐度与跨候选能力。

若直接联合场/CDF 可在相同成本得到同等结果，则论文应定位为“relative set-min runtime assurance 的驾驶应用与严密审计”，而不是宣称新的通用不确定性算法。

## 13. Stop Conditions

- source session simultaneous coverage 在固定协议上失败；
- global (q) 被少量无关边主导，导致 meaningful non-inclusion 证书为空；
- 路径累积使 local graph 比 exact global-gauge 更松；
- learned scale 不优于 fixed physical scale，删除 learned scale；
- graph 需按 target 或 candidate 动态删边才有效；
- CDF/joint field 在同样 exact readout 下等价；
- 跨 disconnected components 需要比较但不存在共同 gauge，限制 ROI/保持连通，否则 abstain；
- downstream 碰撞、规则、进度或舒适性抵消道路收益；
- 查新发现等价决策接口，降级贡献或停止。

## 14. Experiment Handoff Boundary

本轮只完成理论对象、架构、训练目标、校准协议、精确算法、风险量词、资格门与 falsifiers。没有执行 pilot、训练或 NAVSIM 长跑。下一阶段只能先进入 experiment-plan；未通过 Gate A/B 前，不得启动完整目标域评测。

