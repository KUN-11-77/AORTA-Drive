# AORTA G1 Final Proposal：先恢复可学习的条件期望决策，再检验视觉结构创新

## 核心问题

在冻结合法候选panel与历史上下文后，模型应选择最大化条件期望`U_ctx`的候选。native-relative目标为

\[
\Delta_i=U_i-U_0,\qquad i^*(X)=\arg\max_i\mathbb E[\Delta_i\mid X].
\]

当前失败不是“网络不够复杂”，而是identity+soft-regret把候选压入低概率尾部，使优秀候选的纠错梯度
近乎消失。E3C只读结果同时显示正/非正机会的聚合参数梯度强冲突，但不证明唯一因果根源。

## 最小方法资格修复

采用native-relative proper mean regression：

\[
L=\frac1{|\mathcal A|}\sum_a\frac1{|C_a|}\sum_{i\in C_a}(s_{ai}-\Delta_{ai})^2,
\qquad s_{a0}=0.
\]

总体平方风险的最优函数为`E[Δ_i|X]`，因而argmax与条件期望效用决策一致；score梯度为residual，不含
当前soft action probability。该性质只提供统计方向，不保证有限网络、有限样本与AdamW成功。

同时有逐输入桥：若`m_i=E[U_i-U_0|X]`，则预测选择的条件regret满足

\[
m_{i^*}-m_{\hat i}\le 2\|s-m\|_\infty\le2\|s-m\|_2.
\]

因此proper mean误差与决策质量存在严格联系，但平均MSE仍不能替代实际gain、near-tie与组外验证。

R005E3D严格保持架构、初始化、数据、优化器、步数、精度与GPU合同不变，只替换目标。通过门要求实际
`G_model>0`、absolute oracle capture≥10%、至少一次有益非native选择以及至少一次非native选择。

## 后续闭环

1. R005E3D训练域资格；
2. job-level group-heldout确认优化与估计泛化；
3. correct/mismatch/removed检验视觉条件增量；
4. Differential Query与参数/FLOP/数据匹配的generic scorer比较；
5. 同场景跨外观、NavDream、跨planner、clean no-harm、时延和选择性授权。

论文核心创新仍是“候选差异条件化的decision-sufficient视觉证据”，不是MSE、DINO或普通scorer。
