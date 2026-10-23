# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** 同一物理场景只发生外观变化时，完整冻结的端到端驾驶 planner 可能输出显著不同甚至违反道路约束的轨迹；我们要从视觉中学习一个可跨冻结 planner 使用的任务接口，使干预依据与物理道路约束一致，而不是依赖全局特征不变性。
- **Must-solve bottleneck:** 绝对道路裕度的安全零点在视觉退化下可能不可辨识，但候选之间的相对道路可行性仍可能保留；方法必须明确回答在不知道绝对安全锚点时，哪些替换仍能被合法授权，并对 shape/局部残差、选择后风险和几何离散误差负责。
- **Non-goals:** 不重建完整 HD map；不修改或联合训练原 planner；不宣称道路可行性等于碰撞、交通规则或闭环系统安全；不使用目标域做训练、阈值、早停或方法选择；不承诺任意 OOD 分布无关保证；不把标准化、conformal 或 DINO 单独宣称为创新。
- **Constraints:** NAVSIM v1/v2；冻结 DINOv3-H+ 与被审计 planner；source-only 学习和校准；候选生成域、动力学资格和意图偏离预算必须预先冻结；大规模实验前先通过结构资格；主要新增可训练组件不超过两个。
- **Success condition:** 在独立日志上证明：相对对比算子比绝对 margin 更稳定且更可校准；同时下界在预先声明的 source-session 风险对象上有效；它能授权绝对区间规则无法授权的、相对 native 不新增道路失败的干预；优势不由候选覆盖、参数量、目标域选择或更强 backbone 解释。

## Anchor Check

- **Original bottleneck:** 视觉外观变化破坏绝对道路 margin 判断，但同一局部道路场对多条轨迹产生共享误差；研究必须说明剩余证据能授权什么，而非伪造绝对安全。
- **Why the revised method still addresses it:** 修订版只学习道路 clearance field 的相对空间形状，并把未知共同偏移显式商掉；决策继续只认证 challenger 相对 native 的道路 margin 不下降。
- **Reviewer suggestions rejected as drift:** 不加入碰撞、动态障碍、交通规则、VLM 推理或 planner 联合训练；这些会改变当前道路约束问题。

## Simplicity Check

- **Dominant contribution after revision:** Gauge-Conformal Minitive Field（GCMF）：在固定任务网格上学习只确定到共同偏移的共享 clearance field，对其 gauge-aligned sup-norm 误差作 session 校准，并由同一个场事件统一覆盖任意候选扫掠集合的相对 margin。
- **Components removed or merged:** 删除 trajectory scalar decoder、soft-min、activity attention、pair evidence scale head、normalized conformal、候选策略相关的最大 residual 校准。
- **Reviewer suggestions rejected as unnecessary complexity:** 不加入 panel transformer、粒子分布、低秩残差生成器或额外 OOD detector。
- **Why this is the smallest adequate route:** 一个共享点场 decoder 加一个无参数解析 min 算子即可同时满足物理局部语义、候选插入不变和 planner-independent 候选读取；校准器只是场级顺序统计。

## Changes Made

### 1. 解除“最大不变量等于视觉可辨识性”的误推

- **Reviewer said:** 商空间是共同平移 nuisance 下的代数最大不变量，但遮挡可以让 contrast 同样不可辨识。
- **Action:** 将主假设改为可检验的条件不确定性假设，不再称 contrast 自动可辨识；加入遮挡反例和停止条件。
- **Impact:** 理论只说明在选定 nuisance 模型下保留了什么，不代替数据资格。

### 2. 与简单 scalar baseline 建立真正的结构差异

- **Reviewer said:** 原 GCCO 在固定尺度时就是 scalar contrast 加 session 最大 residual；异方差版只是 normalized conformal。
- **Action:** 将学习对象改为共享空间坐标上的 gauge field；校准完整任务域的 gauge-aligned sup error。一次覆盖事件对域内所有候选扫掠集合成立，不依赖 calibration 中的候选生成策略或候选数。
- **Impact:** 算法能力从“对已见候选打分”变为“对任意集合查询保持 minitive 结构并获得统一 contrast 界”。

### 3. 关闭 support 物理语义缺口

- **Reviewer said:** 候选级差值监督不能识别局部势；soft-min 甚至可反转真实 min 排序。
- **Action:** 删除 soft-min。点 decoder 直接接受固定物理坐标并用 gauge-centered GT signed-distance 场监督；候选分数使用 exact hard minimum。
- **Impact:** 同一物理 cell 在所有轨迹中共享一个值，argmin 是模型场上的真实活动 cell；与真实活动约束的一致性由场误差界而非注意力叙事支持。

### 4. 重写统计与几何合同

- **Reviewer said:** session 风险单位、target ledger、候选策略范围和重复误差预算不清楚。
- **Action:** 把随机对象、场 target、离散轨迹 margin、连续近似和失败事件全部显式定义；保证限定为 source-session 边际事件。
- **Impact:** 不再把帧风险、干预条件风险和 session 风险混写，也不把输出插入不变误当成统计覆盖。

## Revised Proposal

# Research Proposal: AORTA-v8.1 — Gauge-Conformal Minitive Fields for Appearance-Robust Trajectory Arbitration

## Problem Anchor

- **Bottom-line problem:** 同一物理场景只发生外观变化时，完整冻结的端到端驾驶 planner 可能输出显著不同甚至违反道路约束的轨迹；我们要从视觉中学习一个可跨冻结 planner 使用的任务接口，使干预依据与物理道路约束一致，而不是依赖全局特征不变性。
- **Must-solve bottleneck:** 绝对道路裕度的安全零点在视觉退化下可能不可辨识，但候选之间的相对道路可行性仍可能保留；方法必须明确回答在不知道绝对安全锚点时，哪些替换仍能被合法授权，并对 shape/局部残差、选择后风险和几何离散误差负责。
- **Non-goals:** 不重建完整 HD map；不修改或联合训练原 planner；不宣称道路可行性等于碰撞、交通规则或闭环系统安全；不使用目标域做训练、阈值、早停或方法选择；不承诺任意 OOD 分布无关保证；不把标准化、conformal 或 DINO 单独宣称为创新。
- **Constraints:** NAVSIM v1/v2；冻结 DINOv3-H+ 与被审计 planner；source-only 学习和校准；候选生成域、动力学资格和意图偏离预算必须预先冻结；大规模实验前先通过结构资格；主要新增可训练组件不超过两个。
- **Success condition:** 在独立日志上证明：相对对比算子比绝对 margin 更稳定且更可校准；同时下界在预先声明的 source-session 风险对象上有效；它能授权绝对区间规则无法授权的、相对 native 不新增道路失败的干预；优势不由候选覆盖、参数量、目标域选择或更强 backbone 解释。

## Technical Gap

标准方法可以对某一候选策略产生的 scalar contrasts 使用 session maximum conformal calibration。这在同分布、固定候选过程下是正确的，但有三个结构限制：

1. 一个 generic score \(h(X,\tau)\) 不保证同一物理位置跨轨迹具有相同含义；
2. 它不保证道路 margin 应满足的集合单调性与 union-min 法则；
3. 一旦候选生成策略或 planner 改变，原来的同时覆盖对象也改变，网络的候选插入不变不能自动延伸统计覆盖。

DriveReferee 通过完整预测地图和解析规则获得共享几何，但没有针对绝对地图偏移不可靠时的 gauge-relative 场集合与不退化证书。Conformalized Distance Fields 对绝对残差场作低维联合建模并构造绝对约束下界；其目标不是在丢弃共同 offset 后取得对任意轨迹集合的相对 margin 界。

修订版因此不把“contrast＋conformal”当贡献，而研究一个更具体的问题：

> 能否对只确定到共同偏移的视觉 clearance field 进行场级校准，使同一场事件统一认证任务域内任意候选轨迹相对 native 的道路 margin？

## Method Thesis

- **One-sentence thesis:** 学习一个共享坐标、只确定到场景共同偏移的道路 clearance potential；校准其 gauge-aligned 全域误差后，通过解析 hard-min 将一个场级覆盖事件转换为对任意候选扫掠集合的相对 margin 下界。
- **Smallest adequate intervention:** 一个点查询 decoder、一个 hard-min set operator、一个 scalar session quantile；没有候选网络、anchor head、粒子或 uncertainty head。
- **Foundation-model role:** 冻结 DINOv3-H+ 只作为图像局部证据底座；新方法的结构来自物理场、集合算子与统计合同，而非基础模型名称。

## Contribution Focus

- **Dominant contribution:** GCMF——对 clearance field 的 additive gauge quotient 进行 sup-norm session calibration，并利用 minitive set-functional 结构，把场级覆盖统一转化为候选策略无关的 relative-road-margin certificate。
- **Supporting result:** 对有限任务网格、连续扫掠体近似和 source-session 自适应仲裁给出一致的误差与风险合同。
- **Non-contributions:** DINO、SDF、hard min、split conformal、baseline fallback 都是已知部件；创新是否成立取决于 gauge-field calibration 与 minitive decision transfer 的组合是否存在未覆盖的先行工作。

## Mathematical Object

### 1. Finite task field

令 \(\Omega=\{q_1,\ldots,q_N\}\) 为固定 ego-centric 道路任务网格，cell centers、ROI、分辨率和时间索引与 source teacher 协议一起冻结。\(d_S:\Omega\to\mathbb R\) 是 GT drivable-area signed-distance field，正值在道路可行侧。

冻结 tracker 将轨迹 \(\tau\) 变为非空 swept-cell set

\[
A_\Omega(\tau)\subseteq\Omega.
\]

离散道路 margin 是

\[
M_S^\Omega(\tau)
=\min_{q\in A_\Omega(\tau)}d_S(q)-m_{\rm buf}.
\]

这是本方法直接认证的任务量。官方 DAC、连续车体和真实执行安全均不是该定义的自动同义词。

### 2. Gauge field

网络输出共享点场

\[
u_\theta(X,q)\in\mathbb R,
\qquad q\in\Omega.
\]

它只要求恢复 \(d_S\) 的相对空间结构：

\[
u_\theta(X,q)+b_X\approx d_S(q),
\]

其中 \(b_X\) 是每个观测允许不同的 nuisance offset。局部遮挡可能使右侧路肩的相对结构也不可辨识，因此这只是可失败的条件模型，不是由 quotient 代数自动得到的事实。

### 3. Minitive trajectory operator

对任意非空 \(A\subseteq\Omega\)，定义

\[
F_u(A)=\min_{q\in A}u(q).
\]

它结构性满足

\[
A\subseteq B\Rightarrow F_u(A)\ge F_u(B),
\]

以及

\[
F_u(A\cup B)=\min\{F_u(A),F_u(B)\}.
\]

因此同一 cell 跨候选共享语义，轨迹加入额外扫掠区域不可能被预测为 margin 更高。generic scalar trajectory scorer 不自动满足这两条 law。

预测 contrast 为

\[
\widehat\Delta_{j0}
=F_u(A_\Omega(\tau_j))
-F_u(A_\Omega(\tau_0)).
\]

共同 offset 在 hard min 后仍严格抵消。

## Core Theorem

### Gauge-field transfer theorem

对任意场 \(u,d:\Omega\to\mathbb R\)，定义 gauge-aligned error

\[
\varepsilon(u,d)
=\inf_{b\in\mathbb R}
\|u+b\mathbf1-d\|_\infty.
\]

若 \(\varepsilon(u,d)\le e\)，则对任意非空集合 \(A,B\subseteq\Omega\)，

\[
\left|
[F_u(A)-F_u(B)]
-[F_d(A)-F_d(B)]
\right|\le2e.
\]

证明只需使用 \(\|u+b-d\|_\infty\le e\) 对两个 infimum 各给出 \(e\) 误差；\(b\) 在差值中抵消。

这一结论与 scalar candidate calibration 的关键区别是量词：覆盖一旦在整个 \(\Omega\) 上成立，\(A,B\) 可以在看到场以后自适应选择，也可以来自未在校准阶段出现的 frozen planner，只要它们是 \(\Omega\) 的非空子集。统计随机对象不再包含候选生成策略。

### Closed-form gauge alignment

令 residual \(r(q)=u(q)-d(q)\)。有限网格上

\[
\varepsilon(u,d)
=\frac12\left[
\max_{q\in\Omega}r(q)
-\min_{q\in\Omega}r(q)
\right],
\]

最优 shift 为

\[
b^*=-\frac12
\left[
\max_qr(q)+\min_qr(q)
\right].
\]

因此校准对象不是任意粒子集合，而是可直接计算的 residual range。

## Architecture

```text
camera history + calibration + ego poses
                 │
                 ▼
       frozen DINOv3-H+ image tokens
                 │
     fixed physical query grid Ω in ego frame
                 │ project each q with deployable geometry
                 ▼
 shared point decoder uθ(X,q)
                 │ one field value per cell; candidate-independent
                 ▼
  frozen tracker rasterizes any planner trajectory to AΩ(τ)
                 │
                 ▼
 exact hard min F_u(AΩ(τ))
                 │
                 ▼
 pair contrast Δ̂j0 and field-calibrated lower bound
                 │
                 ▼
 qualification + least-deviating certified challenger
```

### Visual field decoder

对每个 \(q=(x,y)\in\Omega\)，使用已声明的 ground-plane/road-height 模型形成三维点，经相机外参、历史 pose 和裁剪后内参投影到 frozen multi-scale DINO tokens。双线性采样的局部特征与 \((x,y)\)、时间、相机 visibility mask 和全局 source token 进入共享小型 MLP/attention decoder，输出 \(u_\theta(X,q)\)。

候选轨迹不进入 field decoder，所以：

- 新增候选不会改变已有 field 或 trajectory score；
- 不同候选共享同一物理 cell 值；
- 场只计算一次，随后可供多个 frozen planner 查询。

遮挡 cell 使用可部署 visibility/missing token；模型可利用上下文推断，但不声称遮挡区域已知。退化若导致场 residual range 扩大，将直接反映在 coverage/sharpness 审计中。

## Training

教师只包含 drivable-area signed distance。动态障碍、路线命令、碰撞、舒适性和进度均不写入该场。

对 source observation 的 field residual

\[
r_\theta(q)=u_\theta(X,q)-d_S(q),
\]

先去除其加权均值

\[
\bar r_\theta(q)
=r_\theta(q)-\sum_{p\in\Omega}w_p r_\theta(p),
\qquad\sum_pw_p=1.
\]

训练目标只有一个 gauge-invariant field loss：

\[
\mathcal L_{\rm field}
=\sum_{q\in\Omega}w_q
\operatorname{Huber}(\bar r_\theta(q)).
\]

它监督每个物理 cell 的相对 clearance，并保持米制尺度；与同时归一化 scale 的方案不同，未知乘法误差不会被理论消掉，而会进入 field residual range。source-derived appearance corruption 共享同一个 \(d_S\) teacher，但不使用目标域图像。

hard-min 不承担识别局部场的训练责任；局部语义来自 field supervision。因此不再需要 soft-min、activity attention 或候选级辅助 loss。

## Source-Session Gauge Calibration

### Random experiment

- \(D_{cal}=(Z_1,\ldots,Z_n)\)：独立 source calibration sessions；
- \(Z_{n+1}\)：与它们交换的新 source session；
- \(T(Z)\)：协议预先冻结的有限评估时刻；
- 每个时刻的场覆盖整个固定 \(\Omega\)，不依赖任何候选集合。

第 \(i\) 个 session 的非符合分数为

\[
E_i=max_{t\in T(Z_i)}
\varepsilon
(u_\theta(X_{it},\cdot),d_{S_{it}}).
\]

取

\[
k=\lceil(n+1)(1-\alpha)\rceil.
\]

若 \(k>n\)，协议返回不可部署；否则 \(q_\alpha=E_{(k)}\)。由于 \(E_i\ge0\)，不存在负 quantile 造成 uncertainty 越大、下界越乐观的问题。

split-conformal 给出对 \((D_{cal},Z_{n+1})\) 联合随机性的边际 source-session 保证：

\[
P\left(
\forall t\in T(Z_{n+1}):
\varepsilon_t\le q_\alpha
\right)\ge1-\alpha.
\]

结合 gauge-field transfer theorem，对同一 session、任意时刻、任意自适应产生的非空 \(A,B\subseteq\Omega\)：

\[
\Delta_d(A,B)
\ge
\Delta_u(A,B)-2q_\alpha.
\]

这不依赖候选数量、候选生成策略或 planner 身份；依赖的是 field observation 分布、固定 \(\Omega\)、时间协议和 source-session 交换性。外观 OOD 与未见传感器设置只有实证结论。

## Geometry Target Ledger

| 层级 | 对象 | 是否被证书直接覆盖 |
|---|---|---|
| GT field | \(d_S(q),q\in\Omega\) | 是，field calibration target |
| Discrete swept set | tracker 后的 \(A_\Omega(\tau)\) | 是，hard-min operator target |
| Continuous nominal footprint | \(\mathcal B(\tau)\) | 仅在 h-net 条件下加近似界 |
| Real executed footprint | 真实控制与动力学误差后的集合 | 否，除非另有可验证 tracking bound |
| Official NAVSIM DAC | 官方 scorer 事件 | 仅作独立 fidelity audit 与下游指标 |

若 \(A_\Omega(\tau)\) 是 \(\mathcal B(\tau)\) 的内部 \(h_\tau\)-net，且 \(d_S\) 为 1-Lipschitz，则

\[
0\le M_S^\Omega(\tau)-M_S^{cont}(\tau)
\le h_\tau.
\]

因此 challenger 相对 native 的连续 margin 下界可以保守写为

\[
L_{j0}^{cont}
=\widehat\Delta_{j0}
-2q_\alpha-h_j,
\]

因为 native 的离散乐观误差对该下界方向有利。若 rasterization 不是内部 h-net，只能使用对称 Hausdorff 预算并减去 \(\eta_j+\eta_0\)。不能把同一 support 误差既放进 field calibration target 又再次扣除。

## Arbitration and Risk Contract

候选先通过冻结资格层：闭环动力学、命令一致的意图走廊以及预先声明的姿态/速度/终点偏离预算。资格层不读取 GT field。

定义

\[
\mathcal A_t=
\left\{
j:L^{cont}_{j0,t}\ge\gamma
\right\},
\]

其中 \(\gamma\ge0\) 在 source-only 开发阶段冻结。若为空，保留 native；否则选择资格候选中 \(C_{dev}(\tau_j,\tau_0)\) 最小者。

对随机 session \(Z\)：

- \(I(Z)=1\)：该 session 至少发生一次干预；
- \(H(Z)=1\)：该 session 至少一次出现 native 道路 margin 非负、替换轨迹道路 margin 负。

在场覆盖事件上，所有被授权 challenger 均满足 \(M_j\ge M_0\)，故 \(H=0\)。于是

\[
P_{D_{cal},Z}(H=1)\le\alpha.
\]

条件风险仅能写为

\[
P(H=1\mid I=1)
\le\min\{1,\alpha/P(I=1)\}.
\]

这不是逐帧风险、逐干预风险、固定 calibration realization 的 PAC 风险，也不是闭环真实车辆安全保证。

## Why This Is Not the Scalar Baseline

scalar baseline 对校准候选过程中的有限 contrasts 使用

\[
R_i=\max_{t,j}(\hat\Delta_{j0}-\Delta^*_{j0}).
\]

它可合法覆盖同一候选过程，但其随机对象包含 candidate generator。改变 planner、增加不同分布候选或改变候选预算，都改变 score maximum 的分布。

GCMF 校准的是

\[
\inf_b\|u+b-d\|_\infty
\]

这一 candidate-free field event。由于 minitive theorem 对所有 \(A,B\subseteq\Omega\) 同时成立，候选可以在看到 field 后任意插入和自适应选择。这是两者的明确算法能力差异。

代价是 field-level sup error 可能比候选特定校准更保守。若它在真实数据上导致零有效干预，理论普适性没有部署价值，方法判定失败。

## Failure Modes and Stop Conditions

- **Contrast 仍不可辨识：** 遮挡区域相对 field residual range 大，\(q_\alpha\) 使证书失效。停止“relative is sufficient”主张。
- **场级覆盖过度保守：** coverage 正确但有效干预近零，scalar candidate-specific baseline 明显更紧。停止 GCMF 主路线。
- **简单 dense SDF 等价：** absolute dense SDF 在先做最优 shift 后取得相同 sharpness、迁移和成本。承认方法只是已有 map readout 的重参数化。
- **field structure 无效：** generic scalar scorer 在 held-out candidate policies/planners 上同样可靠。撤回 minitive structure 必要性。
- **跨 planner 仍失败：** field 分布本身随 planner 不变，但资格/ROI/轨迹 rasterization 超出 \(\Omega\)。限制主张到被覆盖的轨迹域。
- **其他驾驶指标退化：** 即使道路新失败受控，碰撞、进度、舒适性或规则显著下降。方法不具备系统部署价值。

## Novelty and Elegance Argument

主张不再是“relative margin 更稳”或“conformal 更可靠”，而是三个不可拆开的对象对齐：

1. **Representation:** 只确定到共同 offset 的共享物理 clearance field；
2. **Set structure:** hard-min 使轨迹算子严格满足 inclusion monotonicity 与 union-min；
3. **Calibration-to-decision transfer:** gauge-aligned field sup error 一次覆盖任务域内任意 swept-set contrast，从而允许候选策略无关的自适应仲裁。

DriveReferee 提供绝对预测 map 并作相对比较；CDF 构造绝对 conformal distance-field envelope；GCMF 则主动放弃绝对 field offset，只校准足以支持 relative non-degradation 的 quotient field。这个差异目前是候选创新，必须继续核对 conformal ranking、implicit neural fields、set function learning 与 runtime assurance 文献。

## Claim-Driven Validation Sketch

### Claim 1 — Gauge-field calibration 提供候选策略无关的有效且非平凡 contrast bounds

- **Minimal experiment:** 固定 frozen DINO 与任务网格，在 source-calibration 后测试：同策略候选、未见候选扰动策略、未参与 calibration 的 frozen planner。
- **Baselines:** scalar contrast＋session max calibration；absolute dense SDF＋sup-norm calibration；absolute field 先最优 shift；一般 low-rank field residual；GCMF。
- **Metrics:** field session coverage、所有实际候选 contrast coverage、bound sharpness、可认证 pair 比例、planner/policy transfer degradation。
- **Decisive evidence:** GCMF 在未见候选策略上维持 field-implied coverage，并在相同 coverage 下显著紧于 absolute field；如果只在原候选策略有效或证书为空，则主张失败。

### Claim 2 — Minitive field certificate 支持低新增道路失败的有效仲裁

- **Minimal experiment:** 固定相同候选、资格层和 deviation budget，比较 keep-native、未校准 relative ranking、scalar calibrated、absolute box、DriveReferee-style relative map ranking 与 GCMF。
- **Metrics:** session-level \(H\)、intervention session rate、真实 positive contrast、oracle headroom，以及 DAC/NC/EP/comfort/PDMS 分项。
- **Decisive evidence:** GCMF 在预声明 source 风险对象上覆盖合格，产生非零有效干预，且对 held-out candidate policy/planner 的新增道路失败比 scalar candidate-specific calibration 更稳定。

## Experiment Handoff Inputs

- **Must prove:** field quotient error到任意 set contrast 的 theorem；source-session field coverage；candidate-policy transfer；非平凡 intervention。
- **Must falsify early:** q 是否导致全部 abstain；shift-only 是否足以解释残差；scalar baseline 是否在 held-out policy 上已经等价。
- **Critical controls:** 相同 DINO、任务网格、GT teacher、候选和资格层；target domain 完全冻结。
- **Highest risk:** 与 CDF/implicit field 先行工作重叠；全域 sup-norm 太保守；视觉相对场在遮挡下仍不可辨识。

## Compute & Timeline Estimate

- **T0 proofs and counterexamples:** 3–5 天，CPU。
- **Field-structure qualification:** 用已有 GT field 与 frozen features，先估计 oracle gauge residual range；约 1–2 GPU-days 提取特征，CPU 完成 field statistics。
- **Minimal decoder:** 单一 point decoder，3 seeds 约 60–180 GPU-hours；实际成本由 \(|\Omega|\)、历史帧数和相机数决定，必须在配置中冻结。
- **Calibration:** 至少 99 个真正独立 session 才能得到有限 99% split-conformal 阈值；独立验证风险声明另行计算，不把帧或 corruption 复制视为独立 session。
- **Decision:** 仅当 oracle gauge-field bound 与候选 headroom 均非平凡后进入完整 NAVSIM 评测；否则一周内停止该路线。
