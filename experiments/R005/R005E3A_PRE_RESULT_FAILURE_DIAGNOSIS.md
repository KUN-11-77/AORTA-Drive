# R005E3A 预结果失败诊断与理论修正规则

日期：2026-10-26  
状态：`PRE_RESULT__DOES_NOT_CHANGE_PROTOCOL__NO_FOLLOWUP_RUN_AUTHORIZED`

## 1. 目的

本文件在 R005E3A 正式结果可见前冻结。它不新增 arm、阈值、统计量或运行权限，只规定结果出来后
如何从第一性原理定位失败层，防止把任意负结果都解释成“模型还不够复杂”。

研究对象保持为：给定合法候选集合、因果历史和正确视觉证据，学习当前登记的 contextual utility
关系，并以 hard decision regret 而非训练 loss 评价候选选择。

## 2. 五层可证伪链

### L0：对象与实现有效

必须先通过 provenance、fold/role 隔离、合法 mask、history/坐标语义、finite、同卡 exact replay、
父模型初始化与逐 job 训练预算一致性。L0 失败时结果为 `INVALID`，只能修复实现，不能讨论理论。

### L1：目标与决策对象一致

条件风险满足

\[
\mathcal R_{SR}(s\mid Z=z)
=C(z)-p(s;T)^\top\mathbb E[U\mid Z=z],
\]

因此无界理想函数类下会按条件期望 utility 排序；逐 panel 又有

\[
R_{hard}\leq n\mathcal L_{SR},\qquad n\leq7.
\]

L1 说明该 surrogate 不是任意 heuristic，但当前 `tanh` score 与 native gauge 只保证总体最优的弱
效用排序，可能产生上界饱和 top tie；它也不保证有限样本、当前 DINO 表示和当前 scorer 能达到理想
解。若实验失败，不能跳过 L2–L4 后直接宣布需要新架构。

### L2：优化与可实现性

只读诊断允许核验：

- 训练 soft-regret 是否相对初始化下降，是否出现 NaN、梯度消失或 score 饱和；
- 训练 hard regret、softmax entropy、score range 与 top-score saturation tie 是否随训练朝一致方向变化；
- 每个 panel 在 `tanh` score 范围、native gauge 和冻结温度下的可达 soft-loss floor；
- 最终训练 loss 与该 floor 的 excess，以及 fold/seed 间异质性。

这些量只能区分“训练器未学会”与“学会 surrogate 但未泛化”，不能改写正式 OOF 裁决。若 L2 不合格，
下一步只能做不读取新科学角色的优化/实现 sanity；不得增加视觉模块。

### L3：组级泛化

若 L2 合格但 OOF regret 不改善，比较训练—OOF gap、fold/group 异质性、seed 稳定性和 beneficial /
harmful / neutral choice changes。该层回答有限 42 组下的估计误差是否主导，而不是用 anchor 数制造
伪样本量。任何后续确认仍以 recording group 为独立单位。

### L4：表示与场景绑定

只有当正确视觉相对 S1、S2、S3、removed 的 regret 干预均可靠变差，才支持正确场景视觉包含当前
模型可用的决策信息。若 treatment 优于父 MSE、但输入干预门失败，则改进更可能来自 geometry/history
或优化路径，不得称为视觉机制。下一步只允许表示保留与条件绑定诊断。

### L5：结构必要性

只有 L0–L4 通过，或 L2/L3 的严格诊断证明 generic scorer 存在明确的 candidate-difference 读取瓶颈，
才可预注册 Differential Query。它必须与 generic full-context、independent scorer、D-JEPA-style 与
World4Scorer-style 基线匹配 backbone、token、参数、FLOPs、数据、优化步数和时延预算。

## 3. 冻结结果分支

### GO

含义仅为：在当前 source-fit/OOF 范围内，简单 decision-aligned objective 能把已观察到的关系信号转为
可靠 hard-regret 改善，并依赖正确场景视觉。

唯一允许的下一步是先预注册完整 (2\times3) 因子确认：

- loss：MSE / soft selection-regret；
- training visual arm：correct / mismatch / removed；
- 主对象：loss 主效应、visual 主效应，以及“soft-regret 是否放大 correct-vs-control 决策收益”的交互；
- 同一 6 folds、3 seeds、参数、steps、选择预算和组级统计。

确认前不训练 Differential Query，不读取 NavDream，不作行为或安全主张。

### MIXED

含义：目标改善成立，但正确视觉的决策绑定未被全部独立干预支持。停止扩大 loss；检查 donor 条件平衡、
视觉—候选时空对齐和 decision-critical evidence 是否被冻结表示保留。不得把平均 relation binding 代替
decision binding。

### STOP 且 L2 不合格

含义：当前实现/优化尚不能有效检验 surrogate。只允许 outcome-free 或 outer-train-only 的可实现性、
梯度条件和小样本过拟合 sanity；修复后必须登记为新实验，原 STOP 保留。

### STOP 且 L2 合格

含义：loss-only objective-mismatch 解释被当前预算下否定。停止 soft-regret 路线，依次检查：

1. frozen DINO 是否在候选决策边界保留条件信息；
2. generic scorer 是否能从已保留的信息实现所需关系；
3. 候选 panel、(U_{ctx}) 与视觉观测之间是否本就不可辨识。

只有第 1 项支持而第 2 项失败时，结构创新才获得实验动机。若第 1 项也失败，应收缩视觉主张或改变
可观测信息，而不是给现有表示叠加更复杂的 head。

## 4. Best-Paper 约束

Best-Paper 级推进不等于保证正结果，而是要求每一步都缩小解释空间：

\[
\text{正确对象}
\Rightarrow\text{可优化}
\Rightarrow\text{可泛化}
\Rightarrow\text{视觉绑定}
\Rightarrow\text{结构增量}
\Rightarrow\text{外观鲁棒行为价值}.
\]

任何箭头未通过，就在该层修正理论或实现；不得越级用复杂模块掩盖失败，也不得用后验子集、改阈值、
增加 seed 或访问封存角色改写原裁决。
