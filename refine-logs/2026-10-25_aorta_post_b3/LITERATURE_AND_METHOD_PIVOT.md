# B3 后文献约束与方法转向候选

状态：`LITERATURE_GROUNDED_HYPOTHESIS__NOT_ADOPTED__NO_EXPERIMENT_AUTHORIZED`。

## 1. 文献给出的硬边界

| 工作 | 状态 | 核心对象 | 对 AORTA 的约束 |
|---|---|---|---|
| Zeng et al., *End-to-end Interpretable Neural Motion Planner*, CVPR 2019 Oral | peer-reviewed | 时空 cost volume，轨迹代价为 \(\sum_t c^t(s^t)\) | 将 occupancy/cost field 沿轨迹积分不是新贡献；它也是 B3 后 signed occupancy idea 的直接先例 |
| Casas et al., *MP3*, 2021 | arXiv / robotics | online map、动态 agent future 与 neural planner | 中间场 + planner 的系统闭环已有成熟先例，不能把“视觉场接规划”本身作为创新 |
| Li et al., *GTRS*, 2025 | CVPR 2025 challenge report / preprint | trajectory tokenizer + Transformer scorer；超密词表、vocabulary dropout、sensor augmentation、top-k refinement | 是 NAVSIM scorer generalization 与外观退化上的强直接基线；普通 candidate scorer、dropout、augmentation 均不可宣称原创 |
| Zhong et al., *DA-WAM*, 2026 | preprint | candidate-conditioned future latent，多头 factor/utility scorer，hard-negative ranking | “每候选一个视觉条件表示 + safety hard negative”已被直接提出；AORTA 若走 candidate-conditioned network，必须有不同的数学估计对象和验证命题 |
| Xu et al., *TOAD*, 2026 | preprint | 将冻结 scorer 当 trajectory reward，在 trust region 内 CEM 搜索 | scorer 的 off-proposal generalization、reward hacking、跨 planner 和时延都已有明确门；AORTA 必须与其公平比较而非只做 re-ranking |
| Mandi et al., *Decision-Focused Learning to Predict Action Costs for Planning*, 2024 | preprint | 用 downstream plan quality 而非 prediction MSE 训练 cost prediction | 支持“field MAE 不等于 decision quality”，但 decision-focused loss 本身不是创新 |
| Dixit et al., *Risk-Averse Receding Horizon Motion Planning using Coherent Risk Measures*, AIJ 2023 | peer-reviewed | CVaR/EVaR/g-entropic risk + MPC | CVaR/EVaR 只能作为标准 risk baseline/工具；引入风险度量不自动形成视觉方法创新 |
| Hakobyan & Yang, *Distributionally Robust Risk Map...*, 2021 | preprint | ambiguity set 下 worst-case CVaR risk map | “risk field + worst-case uncertainty”已有成熟路线；AORTA 的 novelty 不能停在分布鲁棒风险地图 |

直接来源与版本见 `references/2026-10-25_b3_decision_operator_sources.md`。

## 2. 被文献否决的三个快捷方案

1. **signed occupancy integral 单独作为贡献：否。** NMP 已使用时变 cost volume 的轨迹累加，
   数学上就是对轨迹 occupation measure 的线性泛函；共同质量抵消虽可用于分析，但不足以构成
   ICCV 方法贡献。
2. **把 min 换成 CVaR/EVaR：否。** coherent/entropic risk 已有完整控制与规划文献；若没有
   appearance-specific identifiability 和 candidate decision evidence，只是标准风险模块迁移。
3. **candidate cross-attention scorer：否。** GTRS 与 DA-WAM 已覆盖 trajectory tokenizer、
   candidate-conditioned representation、factor utility 和 hard-negative refinement；普通 architecture
   variation 很难通过 novelty rebuttal。

## 3. 当前唯一保留的核心缺口候选

现有 AORTA 证据和这些文献共同指向一个更窄的问题：

> 在保持物理场景与候选轨迹不变、只改变视觉外观时，应该保持的是候选之间的任务偏好关系；
> feature、dense field 或单候选绝对 score 的稳定都不是该关系的充分条件。

候选数学对象是 **appearance-quotient preference operator**。由于当前 NAVSIM evaluator 的
progress normalization 依赖 native-candidate 评分对，共享 scorer 不能被错误解释为候选的绝对效用；
它对每个候选给出 planner/native-conditioned 节点分数 \(s_\theta(X,\tau_0,\tau_i)\)，学习与
校准对象是同一 native context 下的差异

\[
\Delta_{ij}^{\theta}(X,\tau_0)
=s_\theta(X,\tau_0,\tau_j)-s_\theta(X,\tau_0,\tau_i).
\]

它具有三项精确结构：

\[
\Delta_{ij}=-\Delta_{ji},\qquad
\Delta_{ij}+\Delta_{jk}+\Delta_{ki}=0,\qquad
\Delta_{ij}(s+c(X)\mathbf 1)=\Delta_{ij}(s).
\]

即候选完全图上的 preference 1-form 由节点 score 的 coboundary 产生，因此天然 antisymmetric、
cycle-consistent 且对 scene-dependent common gauge 不变。推理只需 \(O(K)\) 个共享 score 和
\(O(K)\) 个 native contrasts，不需要 \(O(K^2)\) 独立 pair heads。

这不是已成立创新。审稿人完全可能认为它只是“scores 相减”。只有容量匹配的 direct-pair head
出现 cycle inconsistency / candidate-policy transfer failure，而 exact preference head 显著改善外观下
ranking flip、regret 和跨 planner 泛化时，Hodge/coboundary 结构才获得贡献资格。

### 3.1 估计对象、干预与可识别性

令 \(S\) 表示物理场景，\(A\) 表示只改变成像外观的干预，观测为 \(X=g(S,A)\)，冻结 planner
\(\pi\) 产生 native \(\tau_0^\pi\) 与候选面板
\(\mathcal T(S,\pi)=\{\tau_0^\pi,\ldots,\tau_{K-1}\}\)。冻结 evaluator 的 task utility 应写成
\(U(S,\pi,\tau_i)\)，并在合法外观干预下保持不变。目标不是恢复跨 planner 唯一的绝对 score，
而是估计给定 native context 的商空间偏好：

\[
\Delta_{ij}^{*}(S,\pi)=U(S,\pi,\tau_j)-U(S,\pi,\tau_i).
\]

这里必须显式登记三项假设，而不能把它们写成结论：

1. **外观干预有效性：**变换不改变场景几何、candidate panel 与 evaluator outcome；
2. **标签合法性：**训练可访问的 \(U\) 不含 development/test/target 信息，也不是由待评模型生成；
3. **效用可排序性：**给定 \((S,\pi,\tau_0^\pi)\) 后，冻结 evaluator 的偏好可由标量 utility
   表示；若存在实质非传递偏好，下面的
   exact-gradient 参数化就是错误归纳偏置。

绝对分数只在等价类

\[
[s]=\{s+c(S,\pi)\mathbf 1:c(S,\pi)\in\mathbb R\}
\]

中可识别；偏好差分是该等价类上的 well-defined quantity。这解释了为什么 common scene bias
不应进入训练目标，也解释了为什么只比较绝对 field/score 稳定性并不足够。

### 3.2 可证明结构与可证伪边界

**命题 1（精确图结构）。** 对任意固定的 \((X,\tau_0)\) 与共享标量 scorer，
\(\Delta_{ij}=s_j-s_i\) 在候选完全图上是
0-cochain 的 coboundary，因此严格满足反对称、所有三角环路和为零，以及 scene-gauge 不变性。
这不是经验正则化效果，而是参数化恒等式。

**命题 2（不变性不足）。** 仅最小化 \(\mathcal L_{app}\) 时，常数 scorer
\(s_\theta(X,\tau_0,\tau_i)=c\) 取得零 appearance loss，却不包含任何任务排序信息。因此外观一致性不能
单独识别 \(\Delta^*\)，必须由合法 task supervision 或等价的 decision signal 排除坍缩。

**命题 3（决策遗憾界）。** 若对一个固定 scene/planner context 的全部候选存在某个不可识别
common gauge \(c(S,\pi)\)，使

\[
\max_i\left|s_\theta(X,\tau_0,\tau_i)-U(S,\pi,\tau_i)-c(S,\pi)\right|
\leq\varepsilon,
\]

且 \(\hat i=\arg\max_i s_\theta(X,\tau_0,\tau_i)\)、
\(i^*=\arg\max_i U(S,\pi,\tau_i)\)，则

\[
U(S,\pi,\tau_{i^*})-U(S,\pi,\tau_{\hat i})\leq 2\varepsilon.
\]

证明只使用两次一致误差界与 \(s_{\hat i}\ge s_{i^*}\)。它给出了从表示误差到 decision regret
的最小桥梁，也明确说明普通平均 MAE 不足：需要候选内一致误差、top-margin 与 regret 共同报告。

**命题 4（评分面板依赖）。** 当前 NAVSIM scorer 在 progress 超过冻结阈值时使用

\[
\widetilde p_i(\mathcal P)=
\frac{p_i}{\max_{k\in\mathcal P}p_k}.
\]

因此对一般评分面板 \(\mathcal P\)，加入一个更高 progress 的无关候选会改变原候选的分数；
absolute candidate utility 不满足 independence of irrelevant alternatives。官方单轨迹评价实际使用
\(\mathcal P_i=\{\tau_0,\tau_i\}\)，所以当前合法标签只能写成 native-conditioned
\(U(S,\pi,\tau_i)\equiv U(S;\tau_0^\pi,\tau_i)\)。这直接决定网络必须显式读取 native context，
也禁止把 7 个候选一次批量评分后当作训练真值。若未来 evaluator 改变，estimand 必须重新审查。

**失效条件。** 上述命题不保证跨场景泛化、外观干预真实性、置信度校准或 evaluator 与真实驾驶
安全的一致性；也不证明 exact-gradient 模型优于自由 pairwise 模型。若 oracle preference 大量为
零、真实标签呈非传递性，或 matched direct-pair head 没有出现结构性缺陷，则停止该方向。

### 3.3 Hodge 解释的使用边界

任意反对称 pairwise 预测 \(p\) 可在候选图上分解为 gradient component 与 cyclic residual：

\[
p=B^\top s+r_{cycle},\qquad B r_{cycle}=0,
\]

其中 \(B\) 为定向 incidence matrix（具体转置约定在实现合同中冻结）。共享 scorer 直接令
\(r_{cycle}=0\)。这只在标签来自标量 utility、因此真值位于 gradient subspace 时具有合理性；
Hodge 术语不能被用来掩盖这个可检验前提。

## 4. 候选网络与 loss（只作可证伪设计）

### 4.1 Architecture

\[
Z=E_{vision}(X),\quad a_0=E_{traj}(\tau_0),\quad a_i=E_{traj}(\tau_i),
\]

\[
h_i=H_\theta(a_0,a_i,Z),\quad s_i=W_s h_i,\quad \Delta_{ij}=s_j-s_i.
\]

- frozen/shared visual backbone 只运行一次；
- trajectory encoder 显式编码完整 4 s native、candidate 及其相对 pose/yaw/time，不再通过固定
  dense grid 聚合；
- shared cross-attention scorer 对候选 permutation equivariant；
- multi-factor heads 仅在 NAVSIM labels 合法、且容量匹配 ablation 需要时加入；
- 不恢复 B1 已失去必要性的 graph-correlated uncertainty。

该结构与 GTRS/DA-WAM 形似，因此 architecture 本身不能单独作为 novelty。跨 planner 也不是
“同一绝对 score 无条件迁移”，而是以新 planner 的 native 作为显式 context 做条件泛化。候选创新只可能来自
**appearance intervention 上的 exact preference estimand、cycle consistency 与 selective
native-relative decision rule 的统一**。

### 4.2 Loss

设冻结 evaluator 给出的候选 utility 为 \(y_i\)，同一物理场景的外观变换为 \(T_aX\)：

\[
\mathcal L_{task}
=\sum_{i\ne j}w_{ij}\,
\rho\!\left(\Delta_{ij}^{\theta}(X,\tau_0)-(y_j-y_i)\right),
\]

\[
\mathcal L_{app}
=\sum_a\sum_{i\ne j}w_{ij}\,
\rho\!\left(\Delta_{ij}^{\theta}(T_aX,\tau_0)-
\operatorname{sg}[\Delta_{ij}^{\theta}(X,\tau_0)]\right).
\]

\(w_{ij}\) 必须在结果前由 task-critical / hard-negative 合同冻结，不能按模型误差事后挑 pair。
总损失候选为

\[
\mathcal L=\mathcal L_{task}+\lambda_{app}\mathcal L_{app}
+\lambda_{factor}\mathcal L_{factor}.
\]

不加入 cycle loss，因为 score-difference parameterization 已精确满足 cycle；额外 cycle penalty 只会
制造虚假复杂度。mismatch/removed-image 应保留为测试时视觉必要性反证，不宜直接训练进去。

任务权重与稳健函数也不能凭直觉选择。预注册前必须在 source-only oracle 上比较：uniform pairs、
top-margin pairs 与 task-critical hard negatives 是否对应不同的决策 estimand；\(\rho\) 的尺度由
source label noise/utility range 冻结。若无法说明 \(w_{ij}\) 与最终 regret 的关系，就退回 uniform
weight 基线，而不是增加人为超参数。

## 5. 必须先通过的五个资格问题

1. **标签合法性：**能否在不读取 test/target 的前提下，为冻结 candidate panel 生成 task utility /
   factor labels？它们是否真实区分当前 B3 的 common-min pairs？
2. **Oracle ceiling：**perfect evaluator 上，candidate preference 是否有足够非零差异、低 tie rate 与
   可接受的 cross-planner support？否则停止。
3. **结构必要性：**exact score-difference head 是否胜过同参数 direct-pair、GTRS-style scorer 与
   DA-WAM-style factor scorer；若否，删除 Hodge 叙事。
4. **外观命题：**source controlled appearances 只用于训练/选择；NavDream 保持最终一次性 OOD，
   同时报告 correctness、preference drift、ranking flip、regret、correct-vs-mismatch。
5. **时延：**同一 GPU、batch=1、同步测量 backbone/head/candidate scaling 的 P50/P95/P99；
   TOAD 提供约 1.9–20.4 ms search-overhead 参考，但 AORTA 必须测自己的端到端链。

## 6. 顶会叙事候选（尚未获准）

如果上述门通过，30 秒故事才能写成：

> 外观鲁棒驾驶通常稳定像素、feature 或稠密场，但我们的 oracle 实验发现，即使 perfect road
> field 也会因 candidate readout 丢失决策差异。AORTA 因而直接学习候选图上的 appearance-quotient
> preference：它对外观干预保持 native-relative task margins，并以 exact coboundary 保证反对称与
> cycle consistency；证据不足时保持原 planner。这样，视觉鲁棒性第一次被定义和训练在真正决定
> 轨迹选择的关系上，而不是代理表示上。

当前只能使用前两句作为问题故事。后半段必须由 oracle、网络消融、NavDream、跨 planner、
decision gain 和 latency 共同赢得。
