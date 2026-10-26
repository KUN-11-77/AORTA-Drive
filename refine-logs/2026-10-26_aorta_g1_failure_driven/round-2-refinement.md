# Round 2 Refinement: 冻结为 Identity + Proper Mean Regression 单因素资格实验

## 1. 修正 capacity probe 的包含关系

此前“capacity ladder”并不自动意味着各 probe 函数类相互嵌套。现在只保留一个可严格证明的包含关系：

- 当前 `FullContextSetScorer.score_head` 为 `LayerNorm → Linear → GELU → Linear`；
- exact-head probe 必须缓存最后一个 `GELU` 之后的候选表示 (q_i)，只训练末层线性权重；
- native-relative score 为

  \[
  s_i=w^\top(q_i-q_0),
  \]

  共享 bias 在差分中抵消；
- 因而该 exact-head 类是完整 scorer 函数类的严格子类。

raw-linear 与 small-interaction probe 只作为不同容量的诊断对照，不再称为嵌套 ladder，也不与 exact-head
做排他的优化归因。任何 probe 失败都只否定该 probe 的特征、函数类、正则化和样本预算。

## 2. E3C 不再选择下一处理

E3C 的冻结实现仅包含：

1. 初始化处的 positive/nonpositive **参数梯度**分解；
2. 初始化与 E3B 终点的 **score 梯度**质量比较。

它不计算 E3B 终点参数 Jacobian，不能解释历史 MSE 失败，也不存在结果后定义的“弱/强”分支阈值。
E3C 只描述 soft-regret 的局部优化几何；下一处理在读取 E3C 科学数值前已经由统计原则预先确定。

## 3. 下一处理的第一性理由

部署目标是在给定决策时信息 (X) 后最大化条件期望效用：

\[
i^*(X)\in\arg\max_i\mathbb E[U_i\mid X].
\]

以 native 为 gauge，令 (Delta_i=U_i-U_0)，训练

\[
L_{mean}
=\frac1{|\mathcal A|}\sum_{a\in\mathcal A}
\frac1{|C_a|}\sum_{i\in C_a}(s_{ai}-\Delta_{ai})^2,
\qquad s_{a0}=0.
\]

这里每个合法 anchor 等权、每个 anchor 内 challenger 等权；zero-challenger anchor 按既有实现贡献 0，
但仍保留在 anchor denominator。权重只依赖合法 panel 结构而不依赖 realized utility，故对每个 challenger
的条件风险最优解仍为 (mathbb E[\Delta_i\mid X])。预测 argmax 与 Bayes expected-utility argmax 一致。

该 loss 的 score 梯度是 residual，不含 soft action probability：

\[
\frac{\partial L_{mean}}{\partial s_i}
\propto s_i-\Delta_i.
\]

所以它是针对 E3B softmax-tail 失败的最小、有统计一致性依据的对照，而不是新 loss 贡献。

## 4. R005E3D 唯一处理差异

父实验：R005E3B identity scorer + soft selection-regret。  
处理：同一 identity scorer + 既有 `equal_anchor_challenger_margin_mse`。

冻结不变：

- fold 0 outer-train 的 560 anchors、正确视觉、候选、history、target 与 mask；
- seed `20261025`、初始化 tensor hash、参数量、data order、batch size；
- AdamW、学习率、weight decay、schedule、selected epoch 与 optimizer step budget；
- BF16、deterministic CUDA、GPU roster、primary/replay；
- argmax 采用数组首索引，score tie 时 native 优先；
- 不做 opportunity oversampling、不改 reduction、不加辅助 loss、不调温度、不改网络。

温度不参与 MSE 训练，只保留为 E3B score-gradient可比账本，不作为处理超参数。

## 5. 冻结端点和裁决

R005E3D 仍是 source-fit fold0 outer-train 资格实验，不是 OOF、视觉或鲁棒性结果。

主端点：

- (G_{model}=\mathbb E[U_{selected}-U_{native}]>0)；
- absolute oracle capture (ge0.10)；
- beneficial nonnative count (ge1)；
- nonnative choice count (ge1)。

同时完整报告 harmful/neutral count、hard regret、relation MSE、score distribution 与 epoch path。不得用 loss
下降代替 gain。所有门沿用 E3B，不在结果后修改。

裁决：

- `PASS_PROPER_MEAN_OUTER_TRAIN_QUALIFICATION`：只解锁 job-level group-held-out 预注册；
- `FAIL_FROZEN_QUALIFICATION_GATE`：只说明至少一个冻结资格端点未通过；必须逐项区分
  `NO_POSITIVE_GAIN`、`POSITIVE_BUT_BELOW_CAPTURE_GATE`、无有益选择或无非 native 选择，不据此宣称
  “没有正收益”或“函数类不可实现”，也不自动转入某个排他诊断；
- `INVALID`：机械合同任一失败，不读科学结论。

## 6. 对 R005E2B 的关系

R005E2B 是 `tanh + MSE`，R005E3B 是 `identity + soft-regret`，二者不能形成干净的 objective 消融。
R005E3D 补齐 `identity + MSE` 单元格，直接回答无界输出与 proper mean supervision 的组合能否避免
native-only collapse。R005E2B 仍只作历史参考，不作为唯一 matched parent。

## 7. 贡献边界

即使 R005E3D 通过，也只说明 generic scorer 在训练域具有非退化决策学习资格；它不证明正确视觉有增量，
不证明 Differential Query，不证明外观鲁棒、NavDream、行为或安全收益。G1 必须在后续 job-level group-held-out
与 correct/mismatch/removed 信息门均通过后才闭合。
