# Research Proposal: AORTA G1 Failure-Driven Decision Learning Qualification

## Problem Anchor

- **Bottom-line problem:** 在冻结planner给出的合法ragged候选集合上，模型必须利用决策时可获得的
  \(X=(I,B,h)\) 选择相对native具有正条件期望效用的候选，而不是退化为native-only或有害干预。
- **Must-solve bottleneck:** R005E3A/B表明：总体上decision-aligned的soft selection-regret在当前有限
  网络和优化条件下把所有challenger压入softmax低概率尾部，产生近吸收的零纠错梯度区。
- **Non-goals:** 本阶段不训练Differential Query，不读取NavDream/development/calibration/test/target，
  不扩大网络，不把source-fit成功写成外观鲁棒或驾驶安全收益。
- **Constraints:** 保持候选、\(U_{ctx}\)、DINO cache、模型容量、初始化、optimizer、训练步数和角色封存；
  每次只修改一个可归因因素；42 recording groups是后续统计单位。
- **Success condition:** 先区分机会稀疏、参数梯度冲突、函数类不可实现与优化失败；随后最小surrogate在
  冻结outer-train资格实验中得到 \(G_{model}>0\)、非零有益nonnative选择和至少10% absolute oracle
  capture，并在独立OOF确认中保持组级正收益，才允许进入视觉增量检验。

## Technical Gap

当前失败不是“loss下降不够”，而是四个不同问题尚未分离：

\[
\underbrace{G_{hindsight}>0}_{\text{机会存在}}
\not\Rightarrow
\underbrace{G_{Bayes}(X)>0}_{\text{信息可辨识}}
\not\Rightarrow
\underbrace{f_\theta\text{可表达}}_{\text{函数类可实现}}
\not\Rightarrow
\underbrace{\operatorname{Opt}(L,\theta_0)\text{可找到}}_{\text{可优化}}.
\]

soft selection-regret

\[
L_{SR}=u^*-\sum_i p_i u_i,
\qquad p_i=\operatorname{softmax}(s_i/T)
\]

在总体函数空间中方向合理，但其score梯度含 \(p_i\)。R005E3B已经给出明确反例：优秀候选被错误压低后，
\(p_i\to0\) 使纠错梯度趋于零。删除`tanh`只删除了一个Jacobian饱和因子，不能解决softmax尾部。

直接扩大网络、调温度或增加训练时长都无法区分上述四个问题，因此不是合格的下一步。

## Method Thesis

- **One-sentence thesis:** 用“只读梯度因果诊断 → 冻结表示可实现性probe → 具有错排梯度下界和hard-regret
  上界的单因素surrogate”三步，把decision objective的统计正确性与有限神经网络可训练性闭合。
- **Smallest adequate intervention:** 不增加新视觉模块；先修复训练信号，使generic scorer获得公平的
  正收益资格，再检验视觉与Differential Query。
- **Timeliness:** 该路线把decision-focused learning、cost-sensitive ranking和foundation visual features
  放在同一可证伪链条中，但不把现成排序loss包装成视觉创新。

## Contribution Focus

- **Dominant contribution候选（尚未成立）:** candidate-difference-conditioned视觉证据是否能在外观干预下
  更样本高效地恢复decision-sufficient relation。
- **Supporting contribution:** 从native-relative utility到可优化surrogate、hard regret和保守干预的严格桥。
- **Explicit non-contributions:** pairwise logistic、DINO、softmax、cross-attention、普通scorer和loss替换本身。

## Proposed Method

### Complexity Budget

- Frozen/reused: DINO特征、候选panel、history、\(U_{ctx}\)、`FullContextSetScorer`、optimizer/schedule。
- New trainable components: 0（G1）；后续G3最多1个Differential Query模块。
- Intentionally excluded: 新backbone、额外risk head、RL、几何证书、动态候选生成、loss组合搜索。

### System Overview

```text
合法候选与U_ctx账本
        |
        +--> R005E3C: opportunity + score-gradient + parameter-gradient decomposition
        |                 (no weight update)
        |
        +--> frozen-feature convex probe
        |       |-- succeeds: function class contains signal; current neural optimization fails
        |       `-- fails: representation/input realizability HOLD
        |
        `--> one-factor surrogate qualification
                |-- source-fit outer-train absolute gain
                |-- OOF group-level confirmation
                `-- correct/mismatch/removed visual binding
```

### Core Mechanism 1: Read-only Gradient Decomposition

对challenger按 \(u_i-u_0>0\) 划分正收益集合 \(P\) 与非正集合 \(N\)，计算：

\[
g_P^\theta=\sum_{i\in P}J_i^\top g_i^s,
\qquad
g_N^\theta=\sum_{i\in N}J_i^\top g_i^s.
\]

报告机会比例、score梯度质量、\(\|g_N\|/\|g_P\|\)与cosine。它只定位局部动力学，不产生方法PASS。

### Core Mechanism 2: Utility-Gap Pairwise Surrogate

若E3C支持softmax尾部或参数梯度压制，则在所有严格非tie pair上定义

\[
L_{PW}
=\sum_{i<j:u_i\ne u_j}
|u_i-u_j|\,
\operatorname{softplus}\!\left(
-\frac{\operatorname{sign}(u_i-u_j)(s_i-s_j)}T
\right).
\]

它有两个与当前失败直接相关的性质：

1. 若pair错排，\(y_{ij}(s_i-s_j)\le0\)，则

   \[
   \left|\frac{\partial \ell_{ij}}{\partial(s_i-s_j)}\right|
   \ge \frac{|u_i-u_j|}{2T}.
   \]

2. 令 \(\hat i\in\arg\max s_i\)、\(i^*\in\arg\max u_i\)，则

   \[
   u^*-u_{\hat i}\le \frac{L_{PW}}{\log 2}.
   \]

第一个性质删除了当前 \(p_i\to0\) 的纠错瓶颈；第二个性质把训练surrogate直接连接到hard selection
regret。严格tie pair权重为0，不强制任意唯一标签。

### Statistical Boundary

二候选条件风险可严格得到

\[
z^*(x)=T\log\frac{\mathbb E[\Delta_+\mid x]}
{\mathbb E[(-\Delta)_+\mid x]},
\]

因此其符号与 \(\mathbb E[\Delta\mid x]\) 一致。多候选共享score下不提前宣称完整Fisher consistency；
这里只使用逐样本hard-regret上界与错排梯度下界，并把multi-candidate表现交给冻结实验裁决。

### Frozen-Feature Realizability Probe

在不训练视觉backbone的条件下，对已冻结的候选/历史/视觉表示分别训练：

1. ridge utility-margin regression（凸、条件均值基线）；
2. 线性utility-gap pairwise logistic（凸排序probe）；
3. removed-visual同构probe。

以group-blocked split评估。其职责是区分“表示中没有可恢复信号”和“神经优化没有找到信号”，不是论文
主结果。若correct与removed均失败，停止loss堆叠；若correct成功而深网失败，才进入最小优化修正。

### Training Plan

1. R005E3C primary + same-GPU replay + audit；
2. 依据E3C结果冻结下一分支，不事后改门槛；
3. surrogate先通过合成错排/严格tie/极端logit单元测试与小panel overfit sanity；
4. fold0×1 seed单因素outer-train资格；
5. 通过后才运行6 folds×3 seeds OOF确认；
6. OOF通过后才恢复Correct/Mismatch/Removed视觉因果对照。

### Inference Path

训练目标不改变部署计算图：同一generic scorer输出native-relative scores，argmax选择候选；后续可靠性层仍
要求lower confidence bound严格为正才允许替换native。本阶段不增加推理时延。

## Failure Modes and Diagnostics

- 正机会极稀疏：把opportunity-aware sampling作为后续独立因素，不与loss同时修改。
- \(g_P\)与\(g_N\)强负cosine：先比较pair平衡/anchor平衡，不扩大网络。
- convex probe失败：HOLD视觉可恢复性与DQ，审查输入充分性和条件Bayes gain。
- outer-train通过、OOF失败：函数类可记忆但不泛化；进入正则化/表示审查，不宣称G1通过。
- 预测改善但 \(G_{model}\le0\)：说明surrogate仍未对齐hard decision，停止该候选。

## Novelty and Elegance Argument

RankNet/LambdaRank、Fenchel–Young、SPO+和方向梯度已经覆盖一般pairwise、代价敏感结构化损失与DFL。
因此AORTA不以新loss数量取胜。G1的价值是为视觉核心假设建立无伪阳性的训练资格；最终新颖性必须来自
差异支持如何读取appearance-robust、decision-sufficient视觉证据，并在matched scorer对照中胜出。

## Claim-Driven Validation Sketch

### Claim 1: 当前collapse可被明确的梯度机制解释

- Minimal experiment: R005E3C只读机会/梯度分解。
- Metric: opportunity fraction、正负score/parameter gradient质量、cosine、tail衰减。
- Evidence: 只允许支持具体机制，不产生方法PASS。

### Claim 2: 最小surrogate可恢复真实正收益而非仅降低loss

- Minimal experiment: 相同scorer/初始化/预算下，soft-regret历史对照 vs pairwise单因素处理。
- Metrics: \(G_{model}\)、absolute capture、beneficial/harmful nonnative、group CI、hard regret。
- Success: outer-train非退化且OOF组级正收益；否则停止该surrogate。

## Experiment Handoff Inputs

- Must-prove: 优秀错排pair具有非消失纠错梯度；generic scorer可以获得正的native-relative实际收益。
- Must-run ablations: soft-regret、margin MSE历史父模型、pairwise处理；correct/removed在G1后再做。
- Highest-risk assumptions: 正机会是否可从输入辨识；多候选共享函数类是否能同时实现有用排序。

## Compute & Timeline Estimate

- R005E3C：单GPU，primary+replay，预计分钟级至一小时内；
- convex probe：CPU/单GPU小时级；
- fold0 surrogate资格：单GPU约一次E3B预算；
- 完整OOF确认只有在前门通过后才估算和授权。
