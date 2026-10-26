# AORTA G1：决策学习与优化动力学文献审查

日期：2026-10-26  
用途：解释 R005E3A/B 的 native-only collapse，并限定下一轮最小算法修改的创新边界。

## 1. 文献地图

| 工作 | 发表状态 | 与当前问题直接相关的结论 | 对 AORTA 的约束 |
|---|---|---|---|
| Burges et al., *Learning to Rank using Gradient Descent* (ICML 2005) | 同行评审 | RankNet已经用成对概率代价和神经网络学习排序 | pairwise logistic 本身不是创新，只能作为训练工具或基线 |
| Burges, *From RankNet to LambdaRank to LambdaMART* (MSR TR 2010) | 技术报告 | LambdaRank用任务排序指标变化调制pairwise梯度 | 用utility gap调制梯度与其思想相近，不得宣称“首次代价敏感排序” |
| Blondel, Martins, Niculae, *Learning with Fenchel-Young Losses* (JMLR 2020) | 同行评审 | 给出统一的结构化、代价敏感凸损失；梯度是regularized prediction与target之差 | 支持把“预测分布×期望utility”换成target-residual型梯度，但不自动保证AORTA有限网络成功 |
| Elmachtoub, Grigas, *Smart Predict, then Optimize* (Management Science 2022；arXiv 2017) | 同行评审 | SPO+以决策误差为对象，并在相应线性优化条件下给出统计一致性 | 证明“预测准”与“决策好”需分开；AORTA是已枚举ragged候选排序，不能直接照搬SPO+保证 |
| Gupta, Huang, *Decision-Focused Learning with Directional Gradients* (NeurIPS 2024) | 同行评审 | 用方向性扰动梯度构造决策感知surrogate，并分析模型失配 | 支持单独审查梯度方向与misspecification；其优化器结构与AORTA不同，只作理论参照 |
| Pezeshki et al., *Gradient Starvation* (NeurIPS 2021) | 同行评审 | 梯度下降可能只学习一部分预测特征，使其他有用特征得不到训练 | 提醒检查参数梯度冲突；但AORTA已观测到的是候选概率尾部衰减，不能直接借用其结论命名根因 |
| Mandi et al., *Decision-Focused Learning: Foundations...* (JAIR 2024) | 同行评审综述 | DFL中surrogate、梯度估计、结构假设与决策质量必须联合审查 | 支持当前“目标正确不等于可优化”的三层审查 |
| Ge, Juba, Vorobeychik, *Learning Linear Utility Functions From Pairwise Comparison Queries* (arXiv 2024) | 预印本 | 区分预测pairwise选择与恢复真实utility参数，二者可学习性不同 | 支持把“能选对”与“恢复完整效用”分开；不能作为已同行评审定论 |
| Sakaue et al., *Online Structured Prediction with Fenchel-Young Losses* (arXiv 2024) | 预印本 | 研究结构化输出的surrogate regret与decoding | 提供surrogate-regret桥的相关背景，但设置为online full-information，不直接覆盖当前离线驾驶任务 |

## 2. 对当前失败的最稳健解释

R005E3B 已排除“只有`tanh` Jacobian导致塌缩”。现有证据支持更窄的机制：

\[
\frac{\partial L_{SR}}{\partial s_k}
=\frac{p_k}{T}(\bar u_p-u_k),
\]

当优秀 challenger 已被压到低概率尾部时，纠错信号随 \(p_k\) 指数衰减。这个机制与 RankNet/Fenchel–Young
类target-residual梯度不同；后者对被严重错排的目标仍可保持有限score级梯度。

但“score级梯度不消失”不是充分条件。共享神经网络中仍可能存在：

1. 正收益机会稀疏，平均经验风险偏向native；
2. 正/负机会参数梯度冲突；
3. 当前输入下的条件期望utility不可辨识；
4. 函数类可表达，但优化器找不到；
5. 函数类本身不足。

R005E3C随后只读确认了这一局部动力学：初始化处正收益候选的score方向约94.54%正确，但训练终点
遗漏正收益候选的score-gradient绝对均值从0.1429降到约1.70e-9；同时正/非正样本初始化参数梯度
近乎反向。该结果是描述性证据，不足以对单一根因作因果归因，也不负责结果后选择下一处理。

## 3. 下一轮surrogate的创新边界

utility-gap weighted pairwise logistic可从RankNet/LambdaRank、代价敏感结构化损失与DFL自然推出。
它能提供逐样本错排梯度和hard-regret上界，但一般多候选随机效用下并不保证恢复条件期望效用的排序；
因此不能用这些局部性质建立总体Bayes一致性桥。它只保留为经验候选，不是当前下一步。

当前冻结的最小处理是identity输出下的native-relative proper mean regression：在总体平方风险、可积性、
正且只依赖输入的权重和无限函数类条件下，它恢复条件期望utility gap；这一结论不保证有限网络、有限数据
或AdamW收敛。AORTA最终可能的独立
贡献仍是：appearance-interventional条件下，candidate differential query是否比matched generic scorer
更高效地恢复decision-sufficient visual evidence。G1 surrogate只是使该核心假设获得公平检验资格。

## 4. 可复查来源

- RankNet: https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/
- LambdaRank/LambdaMART overview: https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
- Fenchel–Young losses: https://www.jmlr.org/papers/v21/19-021.html
- SPO+: https://arxiv.org/abs/1710.08005
- Directional gradients: https://papers.nips.cc/paper_files/paper/2024/hash/907a9fb75a408f6c3a2ae1bf84c39e44-Abstract-Conference.html
- Gradient Starvation: https://arxiv.org/abs/2011.09468
- DFL survey: https://arxiv.org/abs/2307.13565
- Linear utility from pairwise queries: https://arxiv.org/abs/2405.02612
- Online structured Fenchel–Young: https://arxiv.org/abs/2402.08180

所有网页原文、可读文本、URL与SHA256元数据保存在本目录的`sources/`下。
