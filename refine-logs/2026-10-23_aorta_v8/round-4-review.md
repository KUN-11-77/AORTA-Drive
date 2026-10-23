# Round 4 Review

<details><summary>Raw reviewer response</summary>

# AORTA-v8.3 Round 4 原始方法审稿

审查范围：完整阅读了 `round-3-refinement.md`，沿用此前已读取的项目问题定义与创新边界。本轮只审查理论和方法规范，没有运行实验、修改方法文件或宣称任何资格门已经通过。

## 1. Problem Anchor 复述与检查

原问题是：同一物理场景发生外观变化后，冻结端到端 planner 可能输出道路违规轨迹；需要一个与物理道路约束一致的任务接口，在绝对道路安全零点不可靠时，为 challenger 相对 native 的替换提供有明确风险范围的证据。

v8.3 的具体解法是：学习 candidate-free 的道路节点势，校准固定物理图上的场差分误差，再在同一 uncertainty set 上精确计算 swept-set contrast 的最坏下界。只有在连续 nominal 几何资格成立、下界非负时才替换 native。

这一目标保留了原问题，没有扩展为地图语义重建、planner 联合训练或其他驾驶安全任务。source-only、冻结 planner、冻结 backbone、无任意 OOD 保证的限制也明确保留。

**Drift Warning：NONE。** 当前设计解决的是原问题中的相对道路不退化子问题。标准工具的使用不构成漂移，也不需要为了前沿感增加 VLM 或新的生成模型。

## 2. 评分与结论

| 项目 | 权重 | 分数 | 加权贡献 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9/10 | 1.35 |
| Method Specificity | 25% | 9/10 | 2.25 |
| Contribution Quality | 25% | 6/10 | 1.50 |
| Frontier Leverage | 15% | 7/10 | 1.05 |
| Feasibility | 10% | 7/10 | 0.70 |
| Validation Focus | 5% | 8/10 | 0.40 |
| Venue Readiness | 5% | 6/10 | 0.30 |
| **OVERALL SCORE** | **100%** | **7.55/10** | **7.55** |

**Verdict：REVISE。** 该 verdict 针对 research-refine 的顶会方法门槛，不等于否定首版系统实现资格。

**对于“第一版理论扎实的系统”的判断：可以接受为 V0 的条件理论规范，足以进入最小实现与 Gate A/B。** 这并不表示已经存在通过几何检查、校准验证或实时测量的系统。当前没有这些执行证据，也不需要在本轮制造这些证据。

上一轮的主要数学与设计缺陷已经实质修复。接下来不建议继续靠增加数学工具或模块追求 9 分；应冻结一个小版本，验证其条件是否可实现、证书是否非平凡。论文原创性与 V0 理论正确性应分开判定。

## 3. 上一轮 CRITICAL 的修复状态

| 上一轮事项 | 当前状态 | 结论 |
|---|---|---|
| continuous bound 遗漏 native 方向条件 | 两条轨迹采样均要求内部，challenger 另有 h-net；一般情形另列对称预算 | 已修复 |
| 逐帧尺度归一化删除整体证据幅度 | 固定物理尺度成为主版本，learned-scale 无逐帧中心化 | 已修复 |
| 多源归约与运行合同不明确 | 每个 native 一次 min-plus 变换，给出复杂度，第一版不用近似 | 已修复 |
| 远处坏边被误称与局部证书无关 | 明确承认全局 q 耦合，只保留给定最终预算后的路径局部性 | 已修复 |
| global-gauge/CDF 使用较弱读出 | 所有结构化对照都要求直接 relative robust readout，CDF 不再可选 | 设计上已修复；训练公平性仍需明确 |
| 将标准工具组合直接当作原创方法 | 承认标准归约，将创新升级设为条件 | 过度主张已修复；原创性本身尚未建立 |

## 4. Continuous geometry：两条路线的符号均正确

### 4.1 Internal route 正确

当前要求 native 与 challenger 的物理采样点分别位于各自连续 footprint 内，并要求 challenger 是内部 h-net。因此

\[
M_j^{cont}\ge M_j^\Omega-h_j,
\qquad
M_0^{cont}\le M_0^\Omega,
\]

推出

\[
M_j^{cont}-M_0^{cont}
\ge
M_j^\Omega-M_0^\Omega-h_j
\ge L_G-h_j.
\]

这一方向正确。native 不需要再有一个 h-net 覆盖半径，只需要它的离散采样确实在连续 footprint 内，保证其离散 margin 乐观。要求两者采用统一内部采样协议，是合理的实现选择。

### 4.2 Symmetric route 正确

若两个采样集合与相应连续 footprint 的 Hausdorff 距离分别至多为 \(\eta_j,\eta_0\)，同一个 GT signed-distance 函数为 1-Lipschitz，则

\[
|M_j^{cont}-M_j^\Omega|\le\eta_j,
\qquad
|M_0^{cont}-M_0^\Omega|\le\eta_0,
\]

从而

\[
\Delta^{cont}_{j0}\ge L_G-\eta_j-\eta_0.
\]

这也正确。提案现在没有通过不合法的方向假设少扣误差。

### 4.3 剩余的是资格判定的可靠性，不是上述代数

“做了一个 hard check”与“该检查能可靠证明几何上界”需要区分。第一版应明确 checker 返回什么证明对象，例如保守解析半径、经过验证的几何包络或其他确定性上界。

若只是从一批离散测试点测得最大误差，它不自动证明整个连续 swept body 的覆盖半径。当前稿件没有声称已实现这种检查，所以这不是新的反例；但它是 Gate A 必须落实的接口要求。

不用同时实现两条路线。V0 可以先实现一种具有可靠上界的路线，另一种保留为可选优化。

## 5. 几何资格失败是否需要加入 H 的风险预算

答案取决于“失败”的含义，不能把它们混成一类。

### 5.1 几何资格被拒绝：不需要额外风险预算

如果某 pair 的 ROI、非空、内部采样或 Hausdorff 资格无法确认，算法不授权这个 pair。若没有其他合格 pair，则保持 native。此时没有因该 pair 新增干预；几何资格拒绝本身不产生 H。

因此，没有必要把“多少候选未通过几何门”加到 \(\alpha\) 中。它影响干预率与效用，应单独报告，但不是新增道路失败概率。

### 5.2 几何资格被错误接受：需要保证其不发生，或为其单独计账

令 \(\mathcal F\) 表示 field coverage event，令 \(\mathcal G\) 表示该 session 中所有实际被授权 pair 的几何上界真实有效。则

\[
H\subseteq\mathcal F^c\cup\mathcal G^c.
\]

如果实现协议对建模的 nominal footprint 具有确定性 soundness，即所有通过的 pair 都满足所声明几何条件，那么 \(P(\mathcal G^c)=0\)，原式

\[
P_{D_{cal},Z}(H=1)\le\alpha
\]

成立。这正是当前稿件明确采用的前提，不需要人为追加一个非零几何风险项。

如果几何上界本身只有 \(1-\beta\) 的概率资格，或可能发生未控制的错误接受，则只能写

\[
P(H=1)\le\min\{1,\alpha+\beta\},
\]

无需假设二者独立。条件于干预的相应上界也应使用 \((\alpha+\beta)/P(I=1)\)。如果 \(\beta\) 未知，就不能给出无条件的 \(\alpha\) 风险结论。

还需避免把“条件于几何通过”直接写成 \(P(H\mid\mathcal G)\le\alpha\)：一般 conditioning 可能改变 field error 的分布。当前的正确路线是 checker soundness 加全 session 的联合事件包含关系。

### 5.3 建议补齐的正式记号

将 \(I(Z)=\max_{t\in T(Z)}I_t\) 明确写出，并把 H 中的 existential time 限制为同一个 \(T(Z)\)。再定义“授权”已经包含 sound geometry gate。这样 field、geometry、selection 的量词完全闭合。

这是规范补齐，不要求扩大风险模型到真实执行或闭环状态分布。

## 6. Fixed-scale primary 与 learned-scale optional 自洽

固定物理尺度

\[
s_e=\ell_e/\ell_{ref}
\]

配合相同尺度下的训练残差和 calibration residual，是自洽的。它使主版本成为一个清楚的 graph-gradient 估计与校准问题，不再依赖一个尚未证明必要的 uncertainty head。

learned-scale 扩展也修复了上一轮的问题：整体观测质量可以保留，source-frozen 的正值下界与上界可落实；固定 \(u,q\) 后，最终尺度逐边增加只会扩张 uncertainty set、使最坏下界不增。这一单调性与网络对 corruption 的实际响应已经正确区分。

全数据统一常数重标定的表述应始终理解为：固定同一个预测器，对全部 scale 作同一重标定后重新计算 calibration 分数，最终 \(qs\) 不变。它不是允许不同样本使用任意 normalization，也不是改变训练 loss 后仍保证得到同一个预测器。

第一版只做 fixed-scale 完全合理。learned-scale 是否改善最终米制锐度应由后续必要性检查决定，不应该为了让系统看起来完整而提前纳入主模型。

## 7. Multi-source formula 与复杂度正确

\[
\psi_B(v)=\min_{b\in B}[u_b+\operatorname{dist}_w(v,b)],
\qquad
L_G(A,B)=\min_{a\in A}[u_a-\psi_B(a)]
\]

与原 exact robust expression 等价。非负有限边权允许一次 multi-source Dijkstra；所有 challengers 只需集合 reduction。给出的时间与额外图存储复杂度合理。

若实现中把初始标签统一减去 \(c=\min_Bu\)，算法返回的是 \(\psi_B-c\)，在计算最终 contrast 前必须恢复这个常数，或对整个表达式采用一致的 gauge。Gate A 应明确覆盖非零初始标签和共同 shift，不要只测试已经零中心化的例子。

第一版固定一个 native、4-neighbor 图、不引入路径近似，是合适的收敛。边权随观测变化，不能将固定拓扑误当成可以预计算所有加权距离；当前一次变换已经给出了正确运行入口。

在没有实际 profiler 结果之前，能够声称的是算法复杂度已清楚，而不是已经实时。

## 8. Strong baselines：读出公平性已修复，训练公平性仍需明确

### 8.1 不能让 absolute baseline 继承一个任意 gauge 的预测器

Block 1 要求相同 node predictor，目的是隔离 uncertainty geometry，这是好的。但如果这个 predictor 只接受 edge loss 训练，其绝对 offset 可以任意变化。让 absolute baseline 直接对这个未经合法 anchoring 的场构造绝对 band，会人为放大它的校准误差。

需要把两种问题拆开：

1. **Uncertainty-only control：** 先固定一个合法 absolute-trained field predictor，再对同一组预测分别使用 absolute band、global gauge set 和 local graph set，各自采用 exact relative readout。这隔离 uncertainty geometry。
2. **End-to-end method control：** absolute loss、gauge/edge loss 使用相同 backbone、field architecture、数据与容量预算，各自合法训练和 source selection；之后再比较相应校准与仲裁。这检查训练 target 的收益。

absolute baseline 不应使用 test GT 来对齐 offset；它也不应因为被禁止学习 offset 而处于不合理劣势。反过来，graph model 的自由 offset 也不是额外信息优势。

无需扩展成新的实验块，在现有 Block 1 中把这两个控制层说明即可。

### 8.2 global gauge exact baseline 可以很便宜

对 \(\operatorname{osc}(d-u)\le R\) 的 global gauge set，它等价于完整图上统一的 pairwise difference budget。令 \(m_B=\min_Bu\)，则相应 min-plus envelope 可写为

\[
\psi_B^{global}(v)=
\begin{cases}
\min\{u_v,m_B+R\},&v\in B,\\
m_B+R,&v\notin B.
\end{cases}
\]

再用相同的 \(\min_A[u-\psi_B^{global}]\) 得到 exact robust contrast。它不需要人为使用昂贵的通用求解器。因此运行成本比较也不能把 global gauge 做成一个不必要的低效实现。

### 8.3 CDF/joint field 比较应是具体模型，而非无限大的类别

要求一个具体 CDF-style uncertainty construction 直接优化 relative set-min 目标，是公平的。要求 GQCF 击败“所有可能的 joint field methods”则没有意义，因为一般联合场类别可以直接包含本方法的 graph polytope。

因此，应在后续 experiment-plan 中固定具体最近邻的区域构造、训练/校准数据与直接读出方法。若其 set-min 优化采用数值求解，必须使用可靠的下界或保守优化误差预算；infimum 问题上的一个可行 primal objective 是上界，不能自动当作安全下界。图解与联合场解都应按同一保守标准比较，并分别报告优化误差和统计误差。

这不是要求本轮运行或实现 CDF，而是使“最强联合场基线”成为可审查的对象。

## 9. 贡献定位：现在诚实，但尚不能升级原创性

v8.3 正确承认 incidence、gauge、edgewise band、conformal、Lipschitz extension、Dijkstra 和 hard minimum 都是标准工具。整个系统也保持一个主机制，没有为修复审稿问题堆出并行贡献。

但将 estimand、uncertainty、decision 和 selection 对齐，是一个严谨系统的必要条件，不必然是新的方法原理。当前主版本仍可精确描述为：在 graph-gradient residual 上做 session split conformal，然后对 graph-Lipschitz uncertainty polytope 采用标准 min-plus robust readout。

其可能价值在于一个适合道路证据的结构化建模选择，以及经过严密控制后的收益和计算成本。这个价值值得验证，也足以支持一个有研究价值的 V0。它是否足以成为顶会方法，需要具体最近邻边界和非平凡结果，不能靠更多“对齐”措辞完成。

尚未取得原创性不等于系统不值得实现；反之，正确实现一个标准工具组成的系统，也不自动取得“首次统一”的资格。

## 10. 七项评分说明及低分修复

### Problem Fidelity：9/10

目标明确保留为 appearance-driven road arbitration，且风险、non-goals 和 source-only 范围一致。没有为了追求分数扩展任务。道路场的建模成本仍需必要性验证，但不构成方向漂移。

### Method Specificity：9/10

主版本的表示、loss、calibration、exact solver、geometry 两路线、仲裁与风险对象已可直接转成实现合同。剩余的 checker soundness、标签恢复和失败事件记号属于必要规范补齐，不再需要重构方法。

### Contribution Quality：6/10

聚焦、简洁、数学对象对齐，但仍是标准结构化预测与鲁棒读取的专门实例。

**CRITICAL 方法级修复：** 明确区分 uncertainty-only 与 end-to-end training 控制，固定具体 joint-field 最近邻。把可能的贡献限定为一个可检验的建模/系统差异，不把对齐本身当作新的普适算法。若公平比较后等价，应执行稿中已有的应用定位或停止条件，不继续追加模块。

### Frontier Leverage：7/10

冻结 DINO 作为局部证据底座是自然的。解析图求解保留保守性和可解释的复杂度，没有必要替换成一个流行但难认证的神经求解器。当前是否能充分使用 foundation features 仍是后续资格问题，不由模型名称保证。

### Feasibility：7/10

固定尺度、单个共享 field decoder、一次 native 变换，使第一版可实现。非平凡性、独立 session 数量、最终图尺寸和几何 checker 仍需后续验证。现在可以开始小规模资格实现，但不能从理论复杂度直接声称已有实时部署能力。

### Validation Focus：8/10

两个验证块和 Gate A–D 已覆盖核心假设。meaningful non-inclusion 的连续定义优于只检查 raster set。主要剩余问题是 absolute predictor 训练公平性和具体 joint baseline 的可审查性，不需要继续增加 benchmark 或新的模块消融。

### Venue Readiness：6/10

理论规范已明显成熟，但顶会原创性未被当前材料证明。

**CRITICAL 方法级修复：** 将 V0 系统目标与论文原创目标分开。冻结并验证最小版本，完成具体最近邻的公平读出/训练比较；若最后只有结构化应用价值，就据此定位。不能仅因 Gate A 正确、coverage 合格或跨候选量词成立而升级“顶会新方法”。

## 11. 对第一版系统的明确裁定

**V0 理论设计：可接受。** 已有一条闭合的条件论证链：

\[
\text{source-session edge event}
\Rightarrow
\text{all-pair discrete robust bounds}
\Rightarrow
\text{sound geometry correction}
\Rightarrow
\text{adaptive nominal-road non-degradation}.
\]

这条链不需要 learned scale，不需要新的基础模型，也不需要新的图算法。剩余规范补齐应进入同一版本的实现前合同，不需要再发明 v8.4 的新核心机制。

**系统验证状态：尚未通过。** 本轮没有证明实际 decoder、数据 split、geometry checker 或 floating-point implementation 满足这些条件。Gate A/B 的结果必须决定是否值得继续，不能将设计审查当成运行验收。

**论文状态：REVISE。** 原创性仍需独立裁定，不能用“可以实现第一版”替代。

## 12. Simplification Opportunities

1. **V0 完全删除 learned-scale 的实现依赖。** 保留文档中的可删除扩展即可，先用一个固定尺度 decoder 和一个 calibration quantile。
2. **V0 只实现一种有可靠上界的连续几何路线。** 不需要同时建设 internal 和 symmetric 两套 checker；不能证明资格的 pair 一律拒绝。
3. **固定一个 native、精确多源读出与最小基线组。** 不提前引入路径近似、多 native 并行求解或新的视觉 backbone。

## 13. Modernization Opportunities

**NONE。** 当前已经适当地复用冻结基础模型。需要的是落实理论资格和控制对照，不是追加前沿模型。解析 min-plus readout 应保留。

## 14. 剩余必须修改项

### V0 规范与实现前必须补齐

1. **明确 geometry gate soundness。** 区分拒绝资格与错误接受；提供内部/对称上界的可验证定义。只有后者存在概率失败时才需额外 \(\beta\)；若是确定性 nominal 几何合同，原 \(\alpha\) 界可以保留。
2. **固定唯一风险记号。** H 与 I 都使用同一有限 \(T(Z)\)，授权明确包含几何资格；不在离散、nominal continuous 和真实执行之间切换。
3. **写清多源标签恢复。** 如初始标签减去 \(\min_Bu\)，最终 envelope 需要相应恢复；Gate A 检查共同 shift、非零标签、零权边和集合相等。
4. **固定一条最小运行配置。** 单 native、4-neighbor、fixed physical scale、图尺寸和一条连续几何路线。该事项属于下一阶段的实现/实验规划，不要求本轮执行。

### 任何正结果与论文主张前必须补齐

1. **修复 baseline 的训练公平性。** uncertainty-only 控制使用合法 absolute-trained 固定 predictor；end-to-end 比较允许各方法合法训练并匹配容量。不能让 absolute baseline 承担人为的任意 offset。
2. **具体化 joint-field/CDF 对照。** 定义区域、校准对象与可靠 relative robust readout，不要求击败一个可以包含本方法的无限泛化类别。
3. **保持优化误差保守。** 不能把数值 infimum 的可行 primal 值直接当下界；baseline 与主方法采用一致标准。
4. **保持原创性表述有条件。** source coverage、candidate independence 和正确标准归约本身不支持“首次”；等价结果应触发既有降级条件。

本轮不再存在必须靠增加新方法组件解决的核心理论缺口。可以将 v8.3 作为首版系统的理论规范继续推进；不能将这一接受判断解释为已验证系统，更不能解释为已经达到顶会原创性门槛。

</details>
