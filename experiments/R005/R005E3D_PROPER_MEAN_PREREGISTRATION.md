# R005E3D：Identity + Proper Mean 单因素资格实验

状态：`IMPLEMENTATION_QUALIFIED_PENDING_COMMIT_AND_SERVER_PREFLIGHT`  
日期：2026-10-26

## 1. 研究问题

R005E3B 证明删除 `tanh` 后，soft selection-regret 仍将全部 challenger 推入低概率尾部并退化为
native-only。R005E3D 只回答：

> 在完全相同的 identity scorer、初始化、数据顺序、优化器和训练预算下，将 non-proper soft action
> objective 替换为 proper native-relative conditional-mean regression，能否在 fold-0 outer-train 恢复
> 严格正的实际候选选择收益？

本轮不是新 loss 贡献，不是视觉因果检验，不是 OOF、Differential Query、NavDream 或驾驶收益实验。

## 2. 第一性统计基础

令决策时信息为 (X=(I,B,h))，候选效用为 (U_i)，native 索引为 0，

\[
\Delta_i=U_i-U_0.
\]

R005E3D 使用既有 anchor-equal、challenger-equal平方损失：

\[
L_{mean}
=\frac1{|\mathcal A|}\sum_{a\in\mathcal A}
\frac1{|C_a|}\sum_{i\in C_a}(s_{ai}-\Delta_{ai})^2,
\qquad s_{a0}=0.
\]

在可积、无正则化的 population risk 和无限函数类下，合法候选权重只依赖 (X) 且严格为正，因此

\[
s_i^*(X)=\mathbb E[\Delta_i\mid X]
=\mathbb E[U_i\mid X]-\mathbb E[U_0\mid X].
\]

其 argmax 与最大化候选条件期望效用一致。该结论不保证有限网络、AdamW、有限样本或14 epochs达到
population optimum；输入权重也会改变有限函数类投影与训练动力学。本轮只资格化冻结实现。

完整定理、证明、预测误差到条件decision regret的界及八类失效条件见
`experiments/R005/R005E3D_PROPER_MEAN_THEORY_NOTE.md`。

与 soft-regret 不同，其 score 梯度为 residual，不含 action-softmax probability：

\[
\partial L_{mean}/\partial s_i\propto s_i-\Delta_i.
\]

所以它是针对 E3B 尾部纠错消失的最小统计正确对照。

## 3. 唯一处理差异

| 项目 | R005E3B control | R005E3D treatment |
|---|---|---|
| output | identity native-relative score | 完全相同 |
| objective | soft selection-regret | proper native-relative MSE |
| 其他 | 冻结 | 完全相同 |

冻结项包括：560 个 fold-0 outer-train anchors、correct visual arm、候选、history、target、mask、seed、
初始化 hash、4,749,313 参数、batch=1、AdamW、LR、weight decay、cosine schedule、14 epochs、7840
steps、数据顺序和 BF16。禁止 oversampling、utility-dependent weighting、辅助 loss、温度搜索或网络修改。

R005E2B 的 `tanh + MSE` 仅是历史参考；它与 E3B 同时改变 activation 和 objective，不能作为本轮干净对照。

## 4. Reduction 与 tie 语义

- 每个 anchor 等权；
- 每个 anchor 内合法 challenger 等权；
- zero-challenger anchor 的 loss 为 0，但保留在 anchor mean denominator；
- native prediction/target 恒为精确 0；
- invalid candidate 不进入 loss；
- hard choice 使用数组首个最大值，native 索引为0，因此 score tie 时 native 优先；
- utility tie tolerance 固定为 `1e-12`，只用于 beneficial/harmful/neutral 记账，不改变训练目标。

## 5. 正式端点与裁决

PASS 必须同时满足：

1. `mean_gain_over_native > 0`；
2. `absolute_oracle_capture >= 0.10`；
3. `beneficial_nonnative_count >= 1`；
4. `nonnative_choice_count >= 1`。

完整报告 relation MSE、hard regret、harmful/neutral count、score distribution 与14个 epoch 的决策路径。

- `PASS_PROPER_MEAN_OUTER_TRAIN_QUALIFICATION`：只解锁 job-level group-held-out 预注册；
- `FAIL_FROZEN_QUALIFICATION_GATE`：逐项报告 `NO_POSITIVE_GAIN`、
  `POSITIVE_BUT_BELOW_CAPTURE_GATE` 等失败原因，不笼统声称“没有正收益”，不作排他因果归因；
- `INVALID`：机械合同失败，不读取科学结论。

## 6. 完整性要求

- primary + same-GPU replay；
- predictions bit-exact，model state exact；
- 独立从 bundle 复核 roster、target、mask、group 数、全部指标和裁决；
- 核验 E3A budget source 与 E3B matched control 的 SHA；
- 结果有限、角色封存、代码/config/runner/provenance exact；
- 任何实现错误先判 INVALID，不用失败结果调阈值或训练预算。

## 7. 理论与论文边界

通过只说明当前 generic scorer 在训练域具有 proper-mean decision-learning 资格。它不支持正确视觉增量、
场景绑定、组外泛化、Differential Query、外观鲁棒、NavDream、实际行为或安全收益。G1 仍需后续
group-held-out 与 correct/mismatch/removed 信息门。
