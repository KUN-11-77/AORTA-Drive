# Round 2 Review

<details><summary>Raw reviewer response</summary>

# AORTA-v8.1 Round 2 原始方法审稿

审查范围：完整阅读了 `round-1-refinement.md`，并沿用上一轮已读取的项目问题定义和 GROUNDING_NOTES。以下只审查方法与论证，没有执行实验或修改文件。

## 1. Problem Anchor 复述与检查

原问题仍是：同一物理场景仅发生外观变化时，冻结 planner 可能输出道路违规轨迹；需要一个与物理道路约束一致的视觉任务接口，为干预提供依据，而不是依赖全局特征不变。

修订稿选择的具体目标是：学习一个只确定到共同偏移的道路 clearance field，通过场级误差校准，对任意域内候选相对 native 的道路裕度建立下界，从而控制“native 原本可行、替换后违规”的 source-session 风险。

这个目标仍解决原问题中的有效子问题。它没有扩展到碰撞、规则、真实闭环车辆安全或任意 OOD 保证，也没有重新训练原 planner。

本轮最重要的进步是：**学习对象、校准对象和决策对象现在基本一致，而且候选策略无关的量词确实成立。** 这不是上一轮 scalar contrast 校准的换名。

但当前路线已经转为全任务网格的道路距离场预测。它不等于完整 HD map，因此不直接违反该 non-goal；然而，它确实接近原本希望避免的 map-first 路线，也显著靠近 Conformalized Distance Fields。不能仅因采用 point decoder，就把它与 dense scalar distance-field prediction 描述为不同范式。

## 2. 评分与结论

| 项目 | 权重 | 分数 | 加权贡献 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 8/10 | 1.20 |
| Method Specificity | 25% | 8/10 | 2.00 |
| Contribution Quality | 25% | 6/10 | 1.50 |
| Frontier Leverage | 15% | 7/10 | 1.05 |
| Feasibility | 10% | 6/10 | 0.60 |
| Validation Focus | 5% | 8/10 | 0.40 |
| Venue Readiness | 5% | 6/10 | 0.30 |
| **OVERALL SCORE** | **100%** | **7.05/10** | **7.05** |

**Verdict：REVISE。**

本轮已经修复多数上一轮的数学对象错配。当前主要阻塞从“方法合同不闭合”转移为两个问题：

1. 与标准均匀距离场置信带、特别是允许共同 offset 的 CDF 变体相比，是否还有足够的方法新颖性；
2. 全域 residual range 是否保守到没有超出几何简单规则的有效干预。

**Drift Warning：AMBER。** 问题语义没有明显漂移，但表示范围从候选任务接口扩展到了整个 ROI 的距离场。需要证明这是获得所需能力的必要代价，而不能把“candidate-policy-independent”本身当作免除该代价审查的理由。

## 3. 七项评分依据与低分修复要求

### Problem Fidelity：8/10

道路裕度、source-only、冻结 planner 和相对不新增道路失败的目标保持清楚。撤回“商空间自动带来可辨识性”是正确修复。

主要保留意见是：主成功条件仍写为“相对对比算子比绝对 margin 更稳定且更可校准”，而实际学习对象已经是全场相对空间结构。候选 contrast 稳定，不必意味着整个 ROI 的 quotient field 容易预测；反过来，整个场覆盖合格，也不必带来有用的实际候选 contrast。

应将这两层条件区分，避免用前者的动机默认支持后者。

### Method Specificity：8/10

核心 transfer theorem、闭式 gauge alignment、正的校准分数、source-session 随机性和连续近似方向均清楚。本轮也真正关闭了 soft-min 与未监督局部势的物理语义缺口。

尚缺的主要是部署资格的可执行定义：完整扫掠体是否位于 ROI、内部 h-net 是否对 native 和每个 challenger 成立，以及失败时必须拒绝连续证书。这些不能只出现在 failure mode 中。

### Contribution Quality：6/10

相对于候选策略相关的 scalar calibration，当前方案确实具有新的能力：一次全域事件覆盖所有域内集合查询。

但这种能力来自 uniform field coverage，并非 GCMF 独有。标准绝对距离场的均匀置信带同样能支持任意域内 swept-set 查询。GCMF 相对它的区别主要是：将共同常数方向商掉，并用 residual oscillation 代替绝对 sup error。

这是清楚的差异，但目前仍像标准场级 conformal 的 quotient 版本，加上距离场已有的解析 min readout。新名称和对象对齐没有自动使它达到方法创新门槛。

**方法级修复，P0：** 加入“同一 absolute field predictor＋相同 residual-range calibration＋相同 hard-min 仲裁”的强基线，并将其与 gauge-centered training 分开。否则无法判断收益来自新训练对象，还是仅来自改变非符合分数。

**方法级修复，P0：** 明确与 CDF 的差别不能只写“它研究绝对场、我们研究相对场”。需要说明把 CDF 的共同常数模式视为 nuisance 并投影掉后，究竟仍缺少哪一种能力。若没有，应收缩为已有场校准方法的任务化简或研究发现。

### Frontier Leverage：7/10

冻结 DINO 的角色自然，也不需要追加 VLM 或更大的基础模型。一个共享点 decoder 可以利用局部视觉证据，物理坐标和 hard min 提供结构。

但现在对遮挡与远端道路形状提出了更强的全域预测要求。基础模型不会自动使这些位置的相对结构可辨识；它们反而可能决定整个 session 的阈值。

### Feasibility：6/10

训练与推理结构简单，但统计与非平凡性风险仍高。全 ROI、所有 cell、所有评估时刻的最大 residual range，可能由少数不可见或与实际决策无关的位置控制。

**方法级修复，P0：** 在任何完整评测之前，给出实际预测场对应的 held-out session 阈值与候选改善幅度之间的可行性判据。不能用 GT field 与自身比较产生的“oracle error”替代视觉预测误差。

**方法级修复，P1：** 固定网格、时间协议、可达域、输入相机和 decoder 查询成本，并给出实际内部 h-net 或 Hausdorff 验证方法。若没有非平凡证书，按当前停止条件停止，而不是事后缩小 ROI 或改变风险单位。

### Validation Focus：8/10

两项主 claim 比上一轮更集中，且明确允许“场级保证正确但没有部署价值”的失败结果。候选策略迁移也是正确的区分点。

仍缺两项决定性控制：允许共同 offset 的场级校准基线，以及完全不读取视觉的 swept-set inclusion baseline。缺少它们，正结果可能被错误解释为 gauge field 的必要性。

### Venue Readiness：6/10

这是一个数学上明显更完整的提案，但尚不是 READY。主要问题不是缺少更多模块，而是最接近的场级方法可以通过很直接的改动取得相同保证。

**方法级修复，P0：** 完成 quotient field 与既有 uniform field calibration 的明确对照，判断是否需要重写主贡献。

**方法级修复，P1：** 若仍保留方法论文定位，需要展示场级统一保证产生了简单场基线和几何规则都无法提供的非平凡决策能力。只优于候选策略相关 scalar baseline 不够。

## 4. 核心数学审查

### 4.1 Gauge-field transfer theorem 正确，而且可以写成精确等式

令

\[
r(q)=u(q)-d(q).
\]

任意非空集合 \(A\) 都满足

\[
\min_q r(q)
\le F_u(A)-F_d(A)
\le \max_q r(q).
\]

因此，任意 \(A,B\) 的 contrast error 不超过 residual range：

\[
\left|
[F_u(A)-F_u(B)]-[F_d(A)-F_d(B)]
\right|
\le \max r-\min r.
\]

由于允许 singleton sets，分别取 residual 最大点和最小点，得到精确等式：

\[
\sup_{\varnothing\ne A,B\subseteq\Omega}
\left|
[F_u(A)-F_u(B)]-[F_d(A)-F_d(B)]
\right|
=
\max r-\min r
=
2\varepsilon(u,d).
\]

这说明两件事：

1. 当前定理的全子集量词真实成立，系数 2 也不能在当前信息下普遍改小。
2. 它本质上是 residual oscillation seminorm 对所有集合 contrast 的精确控制，而不是一个特殊神经网络定理。

建议写出该等式，使能力与保守性的来源都透明。它能加强严谨性，但仍不应被当成复杂的新理论。

### 4.2 per-frame 的 \(\inf_b\) 不会让相对错误结构逃逸

这是本轮成立的设计。

若某帧所有 residual 都等于任意大的常数 \(c_t\)，则分数为零；该常数确实会在同帧的所有 hard-min contrasts 中消失。不同帧允许不同 \(c_t\)，也不会破坏逐帧 contrast 的保证。

反之，只要空间上存在非共同误差，例如右侧路肩错、左侧不变，其 residual range 就会增加。一个全局标量 \(b_t\) 不能抹掉局部错误、左右偏差、局部符号反转或乘法尺度错误。

因此，不应继续把“per-frame gauge 太自由”作为当前方法的漏洞。

适用范围也必须保持清楚：这个校准不支持跨帧绝对 margin 比较、绝对安全判定或依赖 offset 的时间累计量。当前方法没有这些目标，所以不构成问题。

前提是校准确实覆盖整个声明的 \(\Omega\)。visibility mask 可以影响网络输入，不能让难预测的 cell 从校准 target 中消失。

### 4.3 candidate-policy-independent 量词成立，但独立的是查询策略

对固定观测 \(X\)、固定任务域 \(\Omega\) 和完整场覆盖事件，集合 \(A,B\) 可以依赖预测场，可以在校准后才出现，也可以来自不同的 frozen planner。候选数不进入该事件。

因此，当前方法确实超越了只校准某个候选生成过程的 scalar baseline。

但应保留以下限定：

- 新候选的全部待认证几何必须落在被覆盖的任务域内；
- 场覆盖必须包括 native 和 challenger 使用的所有 cell；
- 连续证书还需要相应几何近似资格；
- 换 planner 不能同时改变所讨论的 observation distribution 后仍沿用原保证；
- 这支持在相同 source 观测分布上的跨 planner 查询，不自动支持不同 planner 导致的新闭环访问状态分布。

另一个应修正的叙述是：scalar calibration 并不是只能覆盖“已见候选”。它可以覆盖由同一冻结随机过程产生的新候选。GCMF 的优势是更强的统一查询量词，而不是前者只能记忆 calibration 候选。

### 4.4 全域 sup norm 不必然空泛，但存在不可回避的代价

“全域校准必然没有有效干预”是不正确的。若预测误差主要是共同常数，residual range 可以接近零，即使绝对 field error 很大，GCMF 仍可给出紧的 contrast 界。

但以下反例说明风险很现实：

- 所有实际候选位于道路中间；
- 一个不被这些候选经过的不可见网格点产生大小为 \(M\) 的 residual；
- 其他位置 residual 近零。

此时 \(2\varepsilon=M\)。如果实际候选间最大的可实现预测裕度差低于 \(M\)，所有证书都会失败，尽管该错误点与这些决策无关。

这种保守性是当前“任意非空子集”量词的代价。上面的精确等式说明：如果坚持该量词、只保留一个 global radius，就不能声称有一个更聪明的简单证明能普遍消除它。

因此，非平凡性必须作为主资格条件。不能只报告 coverage 后，把零干预解释为安全系统的保守优点。

### 4.5 finite-grid 到连续 margin 的单侧界正确

如果

\[
A_\Omega(\tau)\subseteq\mathcal B(\tau)
\]

且它确实是内部 \(h_\tau\)-net，那么由 1-Lipschitz 性有

\[
0\le M^\Omega(\tau)-M^{cont}(\tau)\le h_\tau.
\]

令该离散乐观误差为 \(\delta_\tau\)，则

\[
\Delta^{cont}_{j0}
=
\Delta^\Omega_{j0}-\delta_j+\delta_0
\ge
\Delta^\Omega_{j0}-h_j.
\]

所以连续下界只扣 \(h_j\) 是正确的，不需要为对称性额外扣 \(h_0\)。

但 internal h-net 不能由“rasterize trajectory”自动推出。常见的“选取与 footprint 相交的所有 cell centers”可能包含 footprint 外的中心，不满足内部条件；有的 rasterization 也会遗漏细小区域或中间运动。

需要把以下条件放进资格层：

1. native 与 challenger 的完整连续扫掠体均在可认证 ROI 内；
2. swept sets 非空；
3. internal-net 方向及覆盖半径可验证；
4. 否则改用有效的 Hausdorff 预算，或只报告离散证书。

几何界是以同一个连续 GT signed-distance 函数为基础。有限网格上的任意 teacher 数值并不自动拥有该连续扩展。

### 4.6 source-session 风险公式本轮基本正确

当前对 \(Z,T(Z),I(Z),H(Z)\) 和 calibration 随机性的区分，修复了上一轮主要问题。

只要覆盖事件推出所有被授权 challenger 的对应真实 margin 不低于 native，就有

\[
P_{D_{cal},Z}(H=1)\le\alpha,
\]

并且在同一联合随机实验中，

\[
P(H=1\mid I=1)
\le \min\{1,\alpha/P(I=1)\}.
\]

这里没有额外的 selection-after-calibration 漏洞。

最后需要明确 \(H\) 使用离散 margin 还是连续 nominal margin。若仲裁使用已满足资格的 \(L^{cont}\)，就应把 \(H\) 定义为连续 nominal 事件。真实执行、官方 DAC 和闭环风险继续不在该定理内。

## 5. 与 Conformalized Distance Fields 的剩余重叠

这是当前最重要的贡献问题。

一个普通的均匀距离场事件

\[
\|u-d\|_\infty\le e
\]

同样支持域内任意非空集合的 min readout 和 contrast 界。因此，“候选策略无关”并不能区分 GCMF 与场级 conformal 方法，只能区分它与特定候选过程上的 scalar calibration。

GCMF 的集合可以写成

\[
\mathcal C_q(u)
=
\left\{
d:\inf_b\|u+b-d\|_\infty\le q
\right\}
=
\left\{
d:\operatorname{osc}(u-d)\le2q
\right\}.
\]

它是允许共同常数方向自由移动的均匀场误差管。

据现有 grounding，我不能断言 CDF 原文已经明确研究这个相同目标。但从方法上，把共同常数模式视为 nuisance、对其余场形状做统一校准，再读取 shift-invariant 函数，是非常直接的场级变体。提案尚未解释这一步为什么构成足够新的方法。

此外，一个保留场相关性的绝对联合集合，可以直接计算

\[
\inf_{d\in\mathcal C}
\left[F_d(A)-F_d(B)\right].
\]

共同 offset 在这个目标里也会自动抵消。不能只把 GCMF 与绝对 marginal box 比较，再据此断言联合场方法必须为共同 offset 支付两次代价。

最低限度应有三层控制：

1. 相同 absolute field predictor，使用普通绝对 sup-norm calibration；
2. 相同 predictor，改用 residual-range calibration 和相同 contrast 仲裁；
3. gauge-centered training 加 residual-range calibration，即完整 GCMF。

这样才能把 calibration target 的收益与新训练目标的收益拆开。还需要一个允许共同 offset 的 CDF/联合场变体，作为最近邻方法控制。

## 6. 一个新缺失基线：无需视觉的集合包含规则

如果 challenger 的扫掠集合满足

\[
A_j\subseteq A_0,
\]

则对任意道路场都有

\[
F_d(A_j)\ge F_d(A_0).
\]

因此，这种相对不退化无需图像、DINO、GT field 或 conformal calibration。

在静态道路任务中，减速、缩短行驶距离或保留相同路径前段，可能产生这种或近似这种结构。意图预算不自动排除它们。

这会产生一个重要混淆：若最终有效干预主要来自缩短扫掠体，不能据此证明视觉 gauge field 的必要性。

必须加入 geometry-only inclusion 仲裁基线，并单独报告非包含候选上的有效干预与收益。

同一观察也暴露了当前下界的松弛：当 \(A_j\subseteq A_0\) 时，真实 contrast 已知非负，但

\[
\widehat\Delta_{j0}-2q
\]

仍可能为负。当前决策器尚未充分利用它强调的 minitive 结构。它主要利用的是 field uniformity，而不是集合关系带来的更精细决策能力。

## 7. Simplification Opportunities

1. **直接用 residual range 表述算法。** \(\inf_b\) 和闭式 alignment 用于解释即可；实现只有每帧 residual range、session maximum 和分位数，不需要显式构造 gauge 变量。
2. **将 minitivity 保留为解析结构性质。** 不应把 hard-min 的 inclusion 与 union-min 定律另外包装成新模型贡献。
3. **收窄主证据链。** 先判断 quotient-field calibration 相对最强场级基线是否有独立收益，再讨论全面跨 planner 实证。不要通过增加 planner 数量掩盖方法等价性。

## 8. Modernization Opportunities

1. **利用集合关系构造更紧的 robust contrast 下界。** 可从同一个 gauge-field uncertainty set 直接求 pair-specific 的最坏 contrast，而不是所有 pair 统一减 \(2q\)。至少应恢复集合包含时的确定性非负结论。该改动应替换现有 readout，不增加新网络。
2. **以预先声明的可达道路域减少无关极值。** ROI 可以由固定的动力学、时间和意图预算定义，保持候选策略无关。不能在看到 calibration residual 或最终效果后删去难点；所有被认证轨迹仍须满足域资格。

这些是改善当前机制实用性的方向，不应被提前当作已经取得的新颖性。

## 9. 排序行动项

### CRITICAL

1. **P0：正面对照场级最近邻。** 加入 absolute predictor＋residual-range calibration，以及允许共同 offset 的联合距离场方法；明确哪些差异仍不是直接重参数化。
2. **P0：将非平凡性作为首要资格。** 检查实际预测场的 session radius 是否小于可实现的候选改善幅度。全域正确但全部 abstain，应按当前停止条件判失败。
3. **P0：加入 geometry-only inclusion baseline。** 证明有效干预不是仅由缩短或减速扫掠体解释，并报告非包含关系上的独有效益。

### IMPORTANT

1. **P1：把连续几何资格写进仲裁入口。** 包括完整 ROI 覆盖、非空集合、internal h-net 方向和可验证半径。
2. **P1：明确跨 planner 保证的 observation-distribution 前提。** 候选查询策略独立，不等于对不同闭环访问状态分布独立。
3. **P1：定义“oracle gauge-field qualification”。** 必须说明实际预测器、合法输入和 held-out 单位；GT field 自身没有视觉预测误差，不能替代该资格。

### MINOR

1. **P2：将 transfer upper bound加强为精确 supremum 等式。** 同时说明最坏常数为何不能普遍改小。
2. **P2：明确 \(H\) 使用哪一层 margin。** 离散、连续 nominal、官方 DAC 和实际执行必须保持不同命名。
3. **P2：修正“scalar baseline 仅覆盖已见候选”的描述。** 它覆盖同一候选过程的新样本；GCMF 的优势是统一量词。

本轮已经形成一个正确且简洁的场级风险合同，也确实超越了候选策略相关的 scalar baseline。尚未成立的是：这种进步是否超过已有均匀距离场校准的直接 quotient 变体，以及它是否能在真实视觉误差下产生超出几何规则的有效决策。

</details>
