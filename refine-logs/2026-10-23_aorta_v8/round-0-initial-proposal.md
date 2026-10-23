# Research Proposal: AORTA-v8 — Gauge-Calibrated Relative Operators for Appearance-Robust Trajectory Arbitration

> 中文工作标题：在不知道绝对安全时认证“不会更差”：面向外观鲁棒驾驶的规范校准相对算子

## Problem Anchor

- **Bottom-line problem:** 同一物理场景只发生外观变化时，完整冻结的端到端驾驶 planner 可能输出显著不同甚至违反道路约束的轨迹；我们要从视觉中学习一个可跨冻结 planner 使用的任务接口，使干预依据与物理道路约束一致，而不是依赖全局特征不变性。
- **Must-solve bottleneck:** 绝对道路裕度的安全零点在视觉退化下可能不可辨识，但候选之间的相对道路可行性仍可能保留；方法必须明确回答在不知道绝对安全锚点时，哪些替换仍能被合法授权，并对 shape/局部残差、选择后风险和几何离散误差负责。
- **Non-goals:** 不重建完整 HD map；不修改或联合训练原 planner；不宣称道路可行性等于碰撞、交通规则或闭环系统安全；不使用目标域做训练、阈值、早停或方法选择；不承诺任意 OOD 分布无关保证；不把标准化、conformal 或 DINO 单独宣称为创新。
- **Constraints:** NAVSIM v1/v2；冻结 DINOv3-H+ 与被审计 planner；source-only 学习和校准；候选生成域、动力学资格和意图偏离预算必须预先冻结；大规模实验前先通过结构资格；主要新增可训练组件不超过两个。
- **Success condition:** 在独立日志上证明：相对对比算子比绝对 margin 更稳定且更可校准；同时下界在预先声明的 source-session 风险对象上有效；它能授权绝对区间规则无法授权的、相对 native 不新增道路失败的干预；优势不由候选覆盖、参数量、目标域选择或更强 backbone 解释。

## Technical Gap

### 1. v7 为什么没有闭合

v7 用

\[
\mathbf m=a\boldsymbol\rho+b\mathbf 1
\]

描述候选道路裕度，但实际预测集合固定 \(\hat{\boldsymbol\rho}\)，只允许 \((a,b)\) 变化。这无法覆盖排序反转和局部边界误差。与此同时，v7 的仲裁只读取每个候选的 \([L_k,U_k]\)，因此对联合集合 \(\Gamma\) 与其矩形包络 \(\operatorname{Box}(\Gamma)\) 给出相同决策；“联合性”没有成为不可替代的决策信息。

仿射标准化本身也不降低自由度。它只有在“视觉条件误差主要落在共同尺度和偏移方向”被独立验证时才具有统计意义，不能从代数唯一性推出。

### 2. 两条候选路线

**Route A：修补完整仿射算子。** 保留 \((a,b,\rho)\)，增加非仿射残差集合 \(e\)，建立完整联合覆盖，并继续执行 native unsafe、challenger safe 的绝对证书。这条路线数学上可修复，但需要同时学习绝对锚点、shape 分布、残差集合和条件风险；它与 Conformalized Distance Fields 的低维联合集合路线接近，且决策仍强依赖不稳定的零点。

**Route B：只学习可辨识的相对道路算子。** 把场景共同安全锚点看作规范自由度，直接在 \(\mathbb R^K/\operatorname{span}\{\mathbf1\}\) 上学习候选对比；以一侧同时下界认证 challenger 相对 native 的道路裕度不下降。它不声称 challenger 绝对安全，却能在绝对零点未知时证明“不会把原本满足道路约束的 native 换成违反道路约束的轨迹”。

本提案选择 **Route B**。Route A 保留为强基线，而不是与 Route B 拼成一个大系统。

### 3. 与最近邻工作的精确边界

- DriveReferee 已经在同一预测地图上使用相对 margin 排序，并解释共同地图偏差可能抵消；因此“相对 margin”不是贡献本身。
- Conformalized Distance Fields 已经覆盖低维场系数、联合 conformal 区域、投影残差和几何约束；因此“低维集合＋残差”不是贡献本身。
- SPIBB 已经研究相对 baseline 的安全策略改进；因此“baseline-relative”也不是一般性首创。

本项目的候选空缺是：**对外观退化下的道路轨迹算子，确定共同锚点不可辨识后的最大有用对象；用插入不变的轨迹条件网络直接学习该对象；对完整候选选择过程校准一侧同时下界；把它变成冻结 planner 的道路可行性不退化合同。** 该组合是否仍有足够新颖性，需要后续系统查新，当前不得写“首次”。

## Method Thesis

- **One-sentence thesis:** 不预测视觉下不可靠的绝对安全锚点，而是学习定义在平移商空间上的轨迹裕度势函数，并对候选相对 native 的对比误差进行 session 级一侧同时校准；仅当该对比在所有校准兼容情况下非负时才替换 native。
- **Why this is the smallest adequate intervention:** 它删除 anchor 粒子、完整 margin 集合、panel transformer 和绝对 unsafe→safe gate，只保留解决“还能知道什么、还能授权什么”所需的一个候选势函数和一个证据尺度头。
- **Why this route is timely:** 冻结视觉基础模型提供通用局部证据，但系统不把基础模型表征相似度当成安全证书；学习只发生在任务查询接口，统计校准只覆盖明确的 source-session 事件。

## Contribution Focus

- **Dominant contribution:** Gauge-Calibrated Contrast Operator（GCCO）——一个以共同锚点为规范自由度、以轨迹相对道路裕度为可辨识输出、以一侧同时下界支持冻结 planner 不退化仲裁的统一数学—网络—决策机制。
- **Supporting contribution:** 由 signed-distance 场共同仿射扰动推导对比稳定条件，并把 support 离散、轨迹执行和选择后风险写入同一证书合同。
- **Explicit non-contributions:** DINOv3、轨迹 query、soft-min、conformal quantile、baseline fallback、解析 signed-distance 规则均不是单独创新。

## Proposed Method

### Complexity Budget

- **Frozen / reused:** DINOv3-H+ 多尺度视觉 tokens；原 planner；相机标定、ego pose、车辆几何；NAVSIM scorer；split conformal 顺序统计。
- **New trainable component 1:** 共享的 support-level trajectory potential decoder。
- **New trainable component 2:** 与 decoder 共享 trunk 的 pair-evidence scale head；它只估计对比残差尺度，不输出安全概率。
- **Explicitly removed:** anchor particle generator、energy-score 粒子分布、双向 panel transformer、完整 BEV/SDF 解码器、绝对安全分类头、unsafe-only 主门控。

### Mathematical Object

物理场景为 \(S\)，外观为 \(A\)，观测为 \(X\sim p(X\mid S,A)\)。道路可行域的 signed distance 为 \(d_S(q)\)，正值表示在建模的可行域内。轨迹 \(\tau\) 的连续扫掠体为 \(\mathcal B(\tau)\)，道路裕度定义为

\[
m_S(\tau)=\inf_{q\in\mathcal B(\tau)}d_S(q)-m_0.
\]

其中 \(m_0\) 是预先冻结的道路缓冲。该量只描述道路几何约束。

给定 native \(\tau_0\) 与 challenger \(\tau_j\)，定义对比算子

\[
\Delta_S(\tau_j,\tau_0)
=m_S(\tau_j)-m_S(\tau_0).
\]

对于候选向量 \(\mathbf m\)，共同平移群作用为

\[
g_b(\mathbf m)=\mathbf m+b\mathbf1.
\]

对比矩阵 \(B\) 是以 native 为中心的 star-graph incidence matrix，故 \(B\mathbf1=0\)，\(B\mathbf m\) 给出全部 \(\Delta_{j0}\)。模型学习的是等价类

\[
[\mathbf m]=\{\mathbf m+b\mathbf1:b\in\mathbb R\},
\]

而不是任意选择一个绝对锚点。

### Core Theoretical Statements

#### Proposition 1 — 平移 nuisance 下的最大不变量

若候选图连通，则 \(B\mathbf m=B\mathbf m'\) 当且仅当 \(\mathbf m'=\mathbf m+b\mathbf1\)。因此 \(B\mathbf m\) 是共同平移群的最大不变量；任何对共同锚点不变的决策都只能依赖该商空间坐标。该命题说明为什么删除 \(b\) 不损失“相对不退化”问题所需的信息。

#### Proposition 2 — 从共同几何误差到对比稳定

若所有候选扫掠体并集上

\[
\tilde d(q)=a,d_S(q)+c+r(q),
\qquad a>0,
\qquad |r(q)|\le\varepsilon,
\]

则对任意两条候选

\[
\left|
[\tilde m(\tau_i)-\tilde m(\tau_j)]
-a[m_S(\tau_i)-m_S(\tau_j)]
\right|\le2\varepsilon.
\]

因此当 \(a|\Delta_{ij}|>2\varepsilon\) 时，排序符号保持。该命题不是对真实视觉误差结构的证明，而是 H3-v8 的可检验充分条件。

#### Proposition 3 — 道路失败不退化

若真实对比满足 \(\Delta_{j0}\ge0\)，则

\[
\mathbf1[m_S(\tau_j)<0]
\le
\mathbf1[m_S(\tau_0)<0].
\]

即 challenger 可能仍然不安全，但不会把一个道路可行的 native 变成道路违规。这一结论不涉及碰撞、规则、舒适性或进度。

#### Proposition 4 — 自适应选择下的同时覆盖

设 \(\mathcal Q(X)\) 是通过冻结动力学与意图资格的候选集合。若构造下界 \(L_{j0}(X)\)，满足 source-session 交换性下

\[
P\left(
\forall t,\forall j\in\mathcal Q(X_t):
\Delta_{j0,t}\ge L_{j0,t}
\right)\ge1-\alpha,
\]

那么任何从 \(\{j:L_{j0}\ge0\}\) 中自适应选择的 challenger，其“native 道路可行但替换后道路违规”事件的无条件 session 概率至多为 \(\alpha\)。条件于发生干预的风险只能写成

\[
P(\text{new road failure}\mid I)
\le\min\{1,\alpha/P(I)\},
\]

不能宣称每次干预错误率小于 \(\alpha\)。

#### Proposition 5 — 几何与执行误差预算

signed distance 为 1-Lipschitz 时，扫掠体 Hausdorff 误差 \(\eta_i\) 使单轨迹 margin 最多变化 \(\eta_i\)。因此部署证书必须使用

\[
L^{\rm deploy}_{j0}
=L^{\rm model}_{j0}-\eta_j-\eta_0,
\]

其中 \(\eta\) 分解记录 support 空间采样、时间采样、车辆 footprint 近似和名义—执行轨迹偏差；直接预测部署候选时不再重复扣除 probe 最近邻误差。

### System Overview

```text
multi-camera history X + calibration + ego poses
                    │
                    ▼
          frozen DINOv3-H+ tokens
                    │ fixed visual context (candidates cannot modify it)
       ┌────────────┴────────────────┐
       │                             │
 native swept tube τ0         challenger swept tube τj
       │ 3-D support projection      │ 3-D support projection
       ▼                             ▼
 shared support-level potential decoder fθ(X, τ)
       │ local potential u(t,r), evidence v(t,r), visibility mask
       ▼
 support soft-min + activity weights
       │
 candidate potential s0,sj and evidence features e0,ej
       └────────────┬────────────────┘
                    ▼
 Δ̂j0 = sj - s0 ; σ̂j0 = evidence_head(ej,e0,eX)
                    │
     source-session one-sided simultaneous calibration qα
                    ▼
 Ldeploy_j0 = Δ̂j0 - qα σ̂j0 - ηj - η0
                    │
 frozen dynamics + intent qualification
                    ▼
 keep native unless some Ldeploy_j0 ≥ γ; choose least-deviating certified challenger
```

### Representation and Architecture

#### Fixed visual context

DINOv3-H+ 完全冻结。候选不能进入视觉 token 的生成过程，也不存在候选间双向 self-attention。这样新增候选不会改变已有候选的势函数：

\[
f_\theta(X,\tau;\mathcal C)
=f_\theta(X,\tau;\mathcal C\cup\mathcal D)
=f_\theta(X,\tau).
\]

排列等变因此提升为候选插入不变。跨 planner 的主张仍限定在相同轨迹域、传感器与资格策略内。

#### Three-dimensional swept-tube queries

每条候选通过闭环 tracker/kinematic model 得到名义 poses，在每个时间点采样车辆 footprint 的内部覆盖网格而非只取四角。利用 ground-plane 或已声明的道路高度模型、ego-to-camera 外参、历史 pose 与裁剪后内参，将三维 support 投影到多相机特征。超出视野或无法由部署信息判定可见的 support 标记为 missing；模型不能使用训练期特权可见性。

#### Support-level active constraint localization

共享 decoder 对每个 support \(q_{i\ell}\) 输出局部势 \(u_{i\ell}\) 与证据特征 \(v_{i\ell}\)。候选势函数使用温度固定的 soft-min：

\[
s_i=-T\log\sum_\ell \pi_{i\ell}
\exp(-u_{i\ell}/T),
\]

活动权重

\[
\alpha_{i\ell}
=\operatorname{softmax}_\ell(-u_{i\ell}/T)
\]

明确定位决定最小裕度的时间—车体 support。证据头读取 \(\{\alpha_{i\ell},v_{i\ell},\text{mask}_{i\ell}\}\)，而不是仅在候选级重加权。

#### Task content and evidence state

势函数 \(s_i\) 表示任务内容；\(e_i\) 与 \(\hat\sigma_{j0}\) 表示当前观测对该 pair 的证据质量。训练只要求同场景外观增强共享对比真值，不要求 \(\sigma\) 对外观不变。遮挡或模糊是否导致 \(\sigma\) 增大是待验证性质，不由网络结构自动保证。

### Training Signal

教师严格限定为道路可行性 signed-distance margin。只改变路线或动态障碍而不改变道路约束真值时，不要求教师变化；若将来扩展碰撞或路线走廊，必须建立独立算子，不得隐式混入本任务。

训练候选来自预先冻结的轨迹域：native、受控横向/曲率/速度扰动以及多个 source planner 的真实输出。每个训练 appearance—including source-derived corruption—使用同一物理 \(\Delta^*_{j0}\)。

中心损失使用加权 Huber：

\[
\mathcal L_{\Delta}
=\frac1{|\mathcal P|}
\sum_{(j,0)\in\mathcal P}
\operatorname{Huber}
(\hat\Delta_{j0}-\Delta^*_{j0}).
\]

证据尺度使用带下限 \(\sigma_{\min}\) 的 Laplace NLL：

\[
\mathcal L_{\rm evid}
=\frac{|\hat\Delta_{j0}-\Delta^*_{j0}|}
{\hat\sigma_{j0}}
+\log\hat\sigma_{j0},
\qquad
\hat\sigma_{j0}\ge\sigma_{\min}.
\]

总损失只有

\[
\mathcal L=\mathcal L_\Delta+\lambda_{\rm evid}\mathcal L_{\rm evid}.
\]

NLL 只用于学习排序误差的相对尺度；最终覆盖来自独立 conformal 校准，不从 NLL properness 推出。

### Source-Session Calibration

训练结束后冻结网络。对第 \(i\) 个校准 session，定义一侧标准化非符合分数

\[
R_i=
\max_{t\in i}
\max_{j\in\mathcal Q(X_{it})}
\frac{\hat\Delta_{j0,it}-\Delta^*_{j0,it}}
{\hat\sigma_{j0,it}}.
\]

令

\[
k=\lceil(n_{cal}+1)(1-\alpha)\rceil.
\]

只有当 \(k\le n_{cal}\) 时才报告有限阈值 \(q_\alpha=R_{(k)}\)；否则协议返回不可部署，而不是裁剪索引。模型下界为

\[
L^{\rm model}_{j0}
=\hat\Delta_{j0}-q_\alpha\hat\sigma_{j0}.
\]

该保证要求校准与测试 session 交换，并同时依赖冻结的候选生成/资格策略与最大候选预算。对 OOD appearance 只报告实证 coverage，不声称分布无关保证。

### Inference and Arbitration

候选首先通过与视觉模型无关的资格层：

1. 闭环动力学可执行；
2. 与命令一致的固定意图走廊；
3. 最大横向、航向、速度和终点偏离预算；
4. 候选数量不超过校准协议的 \(K_{max}\)。

定义

\[
\mathcal A(X)=
\{j\in\mathcal Q(X):
L^{\rm deploy}_{j0}\ge\gamma\},
\]

其中 \(\gamma\ge0\) 在 source-only 开发阶段冻结，用于数值与实际改善裕量。若 \(\mathcal A=\varnothing\)，保留 native。否则选择

\[
j^*=\arg\min_{j\in\mathcal A}
C_{\rm dev}(\tau_j,\tau_0),
\]

即在已认证不退化的候选中选择对 native 意图偏离最小者。这里不使用“绝对安全”标签，也不声称 rescue；若需要绝对 rescue，必须作为独立扩展和独立风险合同。

### Why Joint Structure Is Now Decision-Relevant

绝对 margin 区间的矩形包络给出

\[
\inf_{\operatorname{Box}(\Gamma)}(m_j-m_0)
=L_j-U_0,
\]

它把共同锚点不确定性重复计入。GCCO 直接校准 \(B\mathbf m\)，共同平移被严格消掉。对于

\[
\Gamma=\{(m_0,m_j)=(\beta,\beta+0.3):
\beta\in[-1,1]\},
\]

绝对区间均跨零且矩形差值下界为 \(-1.7\)，而真实联合对比恒为 \(0.3\)。v8 的决策对象正是这个对比，而不是先构造绝对区间再丢弃相关性。

强基线“相对分数＋共享阈值＋联合校准”与 GCCO 极其接近，因此不回避：它就是必须击败或被证明等价的最简基线。若 support-level 势函数、证据尺度和几何误差合同不能在相同校准下优于该基线，则网络设计不构成必要贡献，应退化为更简单的方法论文。

### Failure Modes and Diagnostics

- **局部误差不共享：** 近零 pair 的 contrast residual 大、排序反转集中。诊断：对比 shared-shift、affine、GCCO residual 和一般低秩误差。失败时撤回共同误差机制主张。
- **证据头不表达退化：** corruption 加重而 \(\sigma\) 不增或 coverage 恶化。诊断：按可见 support 比例和 corruption severity 分层 reliability；不得用 Proposition 4 代替该实证。
- **候选域泄漏 planner 指纹：** held-out planner 或候选政策上 coverage 失效。诊断：固定轨迹域重采样与候选插入测试；降低 planner-independent 表述。
- **证书过度保守：** coverage 达标但 intervention 接近零。诊断：报告 sharpness、可认证 pair 比例和 oracle contrast headroom；不得只报告 coverage。
- **道路改善损害其他目标：** PDMS、碰撞、规则、舒适性或进度下降。处理：资格预算与完整下游指标；道路不退化证书不得扩写为整体驾驶不退化。
- **有限 support 乐观：** 对非凸边界或中间时刻漏检。处理：覆盖网格、时间 Lipschitz/稠密采样界和 \(\eta\) 预算；用连续或高分辨率教师审计。

### Novelty and Elegance Argument

v8 不再声称“发现相对 margin”或“提出 conformal 集合”。它提出的机制级论点是：**当视觉道路证据含有不可可靠恢复的共同安全锚点时，应该在该 nuisance 的商空间中定义学习对象、校准对象和决策对象，并让三者完全一致。**

这种一致性带来四个可审查差异：

1. 网络只通过差值监督识别到 additive gauge，不伪造绝对零点；
2. 候选势函数插入不变，避免 panel 组成反向改变 reference；
3. 校准直接覆盖自适应选择所需的一侧同时对比事件；
4. 决策只承诺道路失败不新增，并准确声明无条件与干预条件风险的区别。

方法是否达到顶会创新仍取决于完整查新与资格实验。若强基线在同样的冻结 backbone、候选、校准和资格层下等价，则应承认 GCCO 是一个清晰但非独立的方法贡献。

## Claim-Driven Validation Sketch

### Claim 1 — 商空间对比是外观退化下更稳定、更紧的任务接口

- **Minimal experiment:** 在同物理 scene 的受控 appearance pairs 与未参与开发的 source sessions 上，比较 absolute margin、标准化 shape、native-centered contrast、一般低秩 residual 的误差分解与一侧 bound sharpness。
- **Baselines / ablations:** absolute margin＋同一 session calibration；DriveReferee-style dense SDF＋解析 relative margin；scalar relative score＋全局阈值；GCCO 去掉 support soft-min；GCCO 去掉 evidence scale；一般低秩 joint predictor。
- **Metric:** near-zero pair 的 sign error、session simultaneous coverage、平均/尾部 lower-bound gap、可认证 pair 比例；全部以 scene/session 为统计单位。
- **Expected evidence:** GCCO 在同 coverage 下给出更紧的 contrast 下界，且优势不能仅由标准化、参数量或候选域解释。

### Claim 2 — 相对证书支持绝对区间无法支持的道路不退化干预

- **Minimal experiment:** 固定候选与资格层，比较 GCCO joint contrast、由 absolute joint set 导出的 contrast、absolute marginal box、未校准 relative ranking；直接计算每种规则授权的独有干预及其真实 \(m_j-m_0\)。
- **Baselines / ablations:** 原 v7 unsafe→safe；矩形包络；仅 shared threshold；oracle contrast；保留 native。
- **Metric:** source-session 新增道路失败率、干预率、认证改善率、条件 false-improvement、oracle headroom，以及 NAVSIM DAC/PDMS/EP/comfort/NC 的分项变化。
- **Expected evidence:** GCCO 在不超过预声明新增道路失败预算时，存在非零且有实际效用的独有干预；若只有 interval-tightening 而没有额外有效决策，弱化决策创新主张。

### Claim 3 — 网络复杂性是必要的而非装饰

- **Minimal experiment:** 在相同 frozen DINO tokens、候选和校准器下，删除 support-level activity localization 或 evidence head。
- **Baselines / ablations:** global pooled scalar、centerline-only query、four-corner query、support soft-min、固定 \(\sigma=1\)。
- **Metric:** Claim 1 的 sharpness/coverage 与端到端延迟、显存。
- **Expected evidence:** 若最简单 scalar baseline 等价，则删除复杂 decoder 并把论文收敛为商空间校准算法。

## Experiment Handoff Inputs

- **Must-prove claims:** C1 对比接口更稳定；C2 同时证书支持低新增道路失败的有效干预；C3 support/evidence 架构在同校准下不可替代。
- **Must-run ablations:** absolute＋同校准、relative scalar＋shared threshold、DriveReferee-style dense SDF、一般低秩 residual、fixed sigma、centerline/four-corner/support grid。
- **Critical datasets / metrics:** NAVSIM source session split；严格冻结的 appearance OOD confirmation；session coverage、sharpness、intervention、new road failure、完整 PDMS 分项。
- **Highest-risk assumptions:** contrast error确实比 absolute anchor 稳定；候选域有足够 oracle headroom；source evidence scale 对 OOD 退化有实证价值；意图资格不会吞掉全部可认证替换。

## Compute & Timeline Estimate

- **Theory/T0:** 约 3–5 天，完成命题、反例、数值 property tests，不使用 GPU。
- **Structure qualification:** 先用已有 GT geometry/margin logs；若需重算，估计 1–2 GPU-days 用于 frozen feature extraction，CPU 负责 margin teacher 与统计审计。
- **Minimal model:** 1 个 frozen-backbone feature cache，decoder 训练约 20–60 GPU-hours/seed；3 seeds 约 60–180 GPU-hours，具体以样本量和候选 support 数为准。
- **Calibration/decision:** CPU 为主；99 个独立 calibration sessions 只解决 99% split-conformal 非退化索引，若要以零失败和 95% 置信度支持 1% 风险上界，另需至少 299 个独立验证单位。
- **Timeline:** 在结构资格通过后约 4–6 周完成最小模型与两项核心验证；未通过资格时停止，不进入全基准长跑。
