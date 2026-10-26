# R005E3B：无界相对score的优化/可实现性资格预注册

日期：2026-10-26  
状态：`IMPLEMENTATION_QUALIFIED__SOURCE_ONLY_RUN_AUTHORIZED`  
父结果：R005E3A `FROZEN_MIXED__ALL_NATIVE_COLLAPSE`

## 1. 唯一问题

R005E3A在outer-train和OOF均把所有challenger推至`tanh`下界`-1`。本轮只检验：移除最终有界
激活、保留native-relative gauge后，完全相同的generic scorer与soft selection-regret能否在训练域恢复
非零的绝对候选收益。

本轮不检验泛化、视觉必要性、Differential Query、NavDream、行为或安全。

## 2. 因果处理

唯一处理：

\[
s_i^{old}=\tanh(r_i-r_0)
\quad\longrightarrow\quad
s_i^{new}=r_i-r_0.
\]

保持不变：模型参数及其初始化、DINO cache、candidate/history tensor、outer-train roster、soft-regret
公式、outer-train gap中位数温度、AdamW、schedule、batch=1、数据顺序、父selected epoch与optimizer-step
预算。native仍精确为0，非法候选仍mask为0，参数量和主要FLOPs不增加。

这是单因素参数化诊断。不得同时更换loss、重加权、采样、网络、学习率、温度或epoch。

## 3. 理论预测

当前有界实现对head preactivation的梯度为

\[
\frac{\partial L}{\partial z_k}
=\frac{p_k}{T}(\mathbb E_p[u]-u_k)(1-\tanh^2 z_k).
\]

identity处理删除最后的`1-tanh^2(z_k)`饱和因子，但保留softmax小概率因子。因此：

- 若新处理恢复非零绝对收益，说明`tanh`饱和是当前塌缩的必要贡献因素；
- 若仍native-only，不能归因于`tanh`，下一层应审查soft-regret的概率加权梯度、数据机会稀疏与函数
  可实现性；
- 即使通过，也只证明训练域优化资格，不证明视觉scene binding或OOF泛化。

## 4. 分阶段设计

### E3B-A：实现与反例门

1. 默认`tanh`路径必须与既有checkpoint/state dict兼容；
2. identity路径必须保持native exact zero、challenger permutation equivariance、mask和有限值合同；
3. 合成正机会panel上，identity head的遗漏优候选梯度不得被activation Jacobian抹除；
4. 参数量、输入、forward主体和optimizer合同与父模型一致；
5. 本地与服务器测试、strict mypy、Black及单job primary/replay exact通过。

任一失败：`INVALID`，不得读取训练结论。

### E3B-B：单fold×单seed outer-train资格

- 冻结fold 0、seed `20261025`；
- 只训练identity+soft-regret，最终固定epoch读取，不选best epoch；
- 对同一outer-train anchor用correct视觉计算主指标；既有`refit_train` S1与removed只作非门控机制
  诊断。S2/S3只为outer-test冻结，不事后制造outer-train映射；
- 复用R005E3A对应tanh checkpoint作为已审计负控制，不重新训练。

## 5. 预注册指标

对每个anchor定义

\[
G_{oracle}=U^*-U_0,\quad
G_{model}=U_{\hat i}-U_0,\quad
R_{model}=G_{oracle}-G_{model}.
\]

报告：

- mean `G_model`、`G_oracle`、`R_model`；
- absolute oracle capture `G_model/G_oracle`；
- native选择率、nonnative选择数；
- beneficial/harmful/neutral nonnative选择数；
- challenger score min/max/std及exact boundary计数；
- 初始/最终及逐epochsoftmax概率、head梯度范数与score分布；
- correct相对S1/removed的prediction变化和选择变化（诊断，不作本轮正门）。

## 6. 冻结裁决

`PASS_LINEAR_REALIZABILITY`需同时满足：

1. 全部机械门通过；
2. final mean `G_model > 0`；
3. final absolute oracle capture `>=0.10`；
4. 至少一个beneficial nonnative选择，且总获取收益为正；
5. 非native选择数大于0，且不是非有限值或tie-break伪影；
6. primary/replay逐数组及final state exact。

`FAIL_LINEAR_REALIZABILITY`：机械有效但任一科学资格门失败。此时停止soft-regret+generic scorer路线，
只允许预注册具有非消失遗漏优候选梯度的最简单surrogate诊断。

`INVALID`：任一实现、replay、provenance、finite、role或预算门失败。

PASS只解锁6-fold×3-seed OOF确认实验的设计，不自动运行，不允许回填R005E3A、C27、DQ或NavDream。

本轮参数化的精确梯度分解、identity仍可能softmax尾部衰减的两候选反例，以及仅在本轮失败后才允许
预注册的非消失错排梯度surrogate，冻结于
`experiments/R005/R005E3B_OPTIMIZATION_GEOMETRY_THEORY_NOTE.md`。该说明不改变本轮门槛。

## 7. 数据角色与停止规则

- 仅使用source-fit outer-train；
- development/calibration/test/NavDream/target保持封存；
- 不根据逐epoch结果延长训练、调温度或改变门槛；
- 失败保留，不自动第三次相同运行；
- 所有结果写入registry并先机械审计再解释。

## 8. 实现资格记录

- 2026-10-26：完整本地测试集 `495/495` 通过；
- strict mypy、Black、`git diff --check` 通过；
- 默认`tanh`与新增identity共享完全相同的state dict和参数量；
- 已冻结bundle、cross-fit、mismatch、视觉cache manifest及父job结果SHA；
- 正式运行前仍须在服务器用既有checkpoint完成默认`tanh`预测逐数组兼容复算。

