# R005 视觉主门 STOP 后的第一性理论决策树

日期：2026-10-26  
状态：`PRE_RESULT_DECISION_TREE__NO_NEW_METHOD_AUTHORIZED`  
适用父结果：R005E2B `UNSUPPORTED / STOP_CURRENT_VISUAL_MAIN_CLAIM`

## 1. 为什么先写决策树

本文件在 R005E2S 科学结果可见前冻结，用于阻止“看到结果后再发明理论”。R005E2S 只改变 donor
visual identity，复用已经训练的 V2 checkpoints。它不能改变父实验 STOP，也不能直接授权新网络。

## 2. 需要纠正的理论表述

令合法候选的条件期望效用为

\[
m_i(x)=\mathbb E[U_i\mid X=x],
\qquad
\widehat i(x)=\arg\max_i \widehat m_i(x).
\]

向量平方损失在总体、无限函数类和精确优化条件下恢复条件均值，因此用
\(\arg\max_i m_i\) 最大化条件期望效用。故 R005E2B 的结果**不能**简单表述为“平方损失在理论上
不具备决策一致性”。更准确的诊断是：有限样本、有限表示、有限函数类、正则化与优化下，平均关系
误差的改善没有集中到改变 argmax 的误差方向。

若同一 panel 上

\[
\|\widehat m-m\|_\infty\leq\varepsilon,
\]

则选择 regret 满足

\[
m_{i^\star}-m_{\widehat i}\leq2\varepsilon.
\]

该界说明 pointwise 一致关系估计足以支持选择，但组平均 MSE 只给出松的总体控制；它不保证有限模型
的误差恰好落在 top-choice 边界，也不保证两个 MSE 很接近的模型具有相近 regret。exact utility ties
还会造成“选择改变但 regret 为零”，所以 top-choice accuracy 不能替代 utility regret。

## 3. R005E2S 的四种结果及唯一允许动作

### A. S2 与 S3 都在 decision regret 上稳定劣于 correct

解释：正确场景绑定存在可重复的决策信息，但当前训练三臂的有限估计/优化没有稳定利用它。

允许动作：登记一个最小、容量不变的 objective qualification，只比较同一 generic scorer 的
平方关系回归与一个 decision-aligned surrogate。仍不允许 Differential Query 或 NavDream。

### B. S2 与 S3 只在 relation MSE 上稳定劣于 correct

解释：scene-bound relation signal 可复现，但尚未证明它位于决策关键方向。

允许动作：同样只做最小 objective qualification，并把 decision regret 设为主端点。若简单目标不能
把信号转化为 regret，才进入表示/结构可逼近性诊断。

### C. 只有一个 donor 规则支持 relation 或 decision binding

解释：scene-binding 结论依赖 donor 构造，可能混入未条件化的道路拓扑、交通参与者、信号或输入分布
距离。不能把更强 distribution shift 的退化解释为更强视觉理解。

允许动作：停止 loss 与架构实验，先构造 outcome-free 的 donor-overlap/conditional-balance 诊断；若
无法形成可信负对照，则收缩“视觉必要性”主张。

### D. S2 与 S3 都不支持一致 scene binding

解释：现有正确-vs-S1结果不具 donor-robust 性。冻结 DINO + generic scorer 的场景绑定证据不足。

允许动作：按“观测信息 → 表示保留 → 函数可逼近”顺序做最小 probe；禁止直接添加 ranking loss、
adapter 或 Differential Query。

## 4. 若 objective qualification 获授权，最简单基线是什么

任何新 loss 必须与原 MSE 使用相同 split、输入、模型、参数、optimizer steps、seed 与 checkpoint
selection 预算。首个候选不是复杂多项 loss，而是两种可独立否定的简单对象：

### 4.1 Regret-weighted pairwise logistic

对合法候选对定义真实差值 \(d_{ij}=U_i-U_j\)，使用

\[
\mathcal L_{pair}
=\frac{1}{|P|}
\sum_{i<j,\,d_{ij}\neq0}
|d_{ij}|\,
\log\!\left(1+\exp\left[-
\operatorname{sign}(d_{ij})
(\widehat U_i-\widehat U_j)/T
\right]\right).
\]

权重 \(|d_{ij}|\) 直接对应错序可能造成的 utility 损失，exact ties 权重为零。温度 \(T\) 必须在
outer-train/inner-validation 内冻结，不能用 OOF 或封存角色选择。

### 4.2 Soft selection regret

令

\[
p_i=\frac{\exp(\widehat U_i/T)}{\sum_{j\in V_B}\exp(\widehat U_j/T)},
\]

则

\[
\mathcal L_{soft-regret}
=\max_{i\in V_B}U_i-
\sum_{i\in V_B}p_iU_i.
\]

它直接惩罚把概率质量放在低效用候选上，并对多个真最优候选自然保持零 utility 区分。该目标并非
proper distributional loss，不能单独承担最终不确定性或校准主张；首轮只检验它能否降低 OOF regret。

## 5. 新目标的通过条件必须比“loss 更低”更严格

最小确认实验至少同时要求：

1. correct visual 相对 removed 与两个独立 donor 规则的组级 regret 改善；
2. 相同 scorer 下 decision-aligned objective 相对原 MSE 的组级 regret CI 严格有利；
3. clean relation MSE、finite、non-collapse 与 seed 稳定性没有灾难性退化；
4. 选择改变中 beneficial 大于 harmful，neutral ties 单独报告；
5. 不通过时保留负结果，不增加第二个 loss 试图事后救援。

## 6. Differential Query 的授权条件

只有满足以下全部条件，才允许训练结构创新：

- donor-robust 的 correct-scene decision signal 已存在；
- 简单 decision-aligned objective 已证明目标可学习，或以明确方式证明 generic scorer 的表示/读取
  瓶颈；
- differential query 与 generic full-context、independent scorer、D-JEPA/World4Scorer-style closest
  baselines 在参数、FLOP、token、数据与训练预算上匹配；
- 预先预测优势应集中于 late-fork、near-boundary 或样本效率，而非只增加平均容量；
- 删除 signed differential support 后该优势消失。

否则 Differential Query 仍是有动机但未获资格的候选，不是 AORTA 的既成创新。

## 7. Best-Paper 主张边界

Best-Paper 级理论不是公式数量，而是从一个失败事实推出唯一可证伪的下一步。当前最强、也最诚实的
理论状态是：

\[
\text{relation signal observed}
\not\Rightarrow
\text{decision information established}
\not\Rightarrow
\text{architecture necessity established}.
\]

R005E2S 只裁决第一条箭头中的 donor 稳健性；后两条必须由新的冻结实验分别建立。
