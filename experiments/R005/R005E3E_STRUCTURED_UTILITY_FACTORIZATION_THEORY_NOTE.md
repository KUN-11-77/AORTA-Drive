# R005E3E 理论注记：因果已知项与视觉未知项的结构化效用分解

状态：失败驱动的方法假设；尚未获得训练授权。

## 1. 问题锚点

R005E3D证明：proper mean MSE能逃离100% native塌缩并读取部分correct视觉relation，但22次干预的13次
微小收益被4次约`-1/8`的时序门损失完全压倒。4个灾难样本均满足：

\[
\Delta F_{filtered}>0,\qquad \Delta(HC_2)=-1,
\]

所以问题不是“是否再换一个通用loss”，而是单头网络被要求同时学习两个性质不同的对象：连续、场景相关
的当前帧效用，以及由候选与上一周期动作决定的离散时序兼容门。

## 2. 第一性分解

对候选`i`定义：

- `H_i∈{0,1}`：当前场景hard-metric乘积；
- `B_i∈[0,1]`：不含two-frame comfort的加权base utility；
- `F_i=H_iB_i`：filter后one-frame score；
- `C_i=c(τ_i,h)∈{0,1}`：由当前候选、上一周期实际轨迹与确定性车辆执行算子给出的two-frame comfort。

登记的contextual row utility严格为

\[
U_i=\frac78F_i+\frac18H_iC_i.
\]

令推理信息`X=(V,τ_{0:K},h)`包含视觉、合法候选和因果历史。如果`C_i`可由`(τ_i,h)`在推理时
确定性计算，则它是`X`-可测量的，于是

\[
\boxed{
\mathbb E[U_i\mid X]
=\frac78\mathbb E[F_i\mid X]
+\frac18 C_i\,\mathbb P(H_i=1\mid X)
}.
\]

这不是近似公式，而是条件期望的线性性与`C_i`可测性直接给出的恒等式。

## 3. 网络与loss的最小结构

共享视觉—候选—历史backbone只新增两个容量受控读出：

\[
\hat\mu_i=f_\theta(X,i),\qquad
\hat p_i=\sigma(g_\theta(X,i)),
\]

其中`μ̂_i`估计`E[F_i|X]`，`p̂_i`估计`P(H_i=1|X)`。最终绝对score由固定代数组合：

\[
\hat m_i=\frac78\hat\mu_i+\frac18 C_i\hat p_i,
\qquad
s_i=\hat m_i-\hat m_0.
\]

训练使用component-proper目标：

\[
\mathcal L_{SUF}
=\mathbb E\left[(\hat\mu_i-F_i)^2
+\operatorname{BCE}(\hat p_i,H_i)\right].
\]

在总体风险、无限函数类和优化到全局极值的条件下，平方损失与log loss分别恢复条件均值和条件概率，
因此重构score恢复`E[U_i|X]`，argmax对条件期望效用是Bayes最优。有限网络、有限数据、优化与分布
移位仍需实验，定理不覆盖这些条件。

## 4. 为什么不是目标泄漏

`C_i`只有在以下资格门全部通过时才允许成为推理输入：

1. 只使用当前ego state、当前合法候选与系统上一周期实际选择；
2. 通过确定性车辆执行/运动学算子计算，不读取future GT、评分结果、hard metric或`U_ctx`；
3. 与冻结official two-frame comfort逐anchor、逐候选exact复算；
4. actual-selected-history与sequential部署时可以同样构造；
5. batch-1 P50/P95和额外显存满足预注册时延门。

任一条件失败，解析分支停止；禁止直接把E1B的comfort标签喂给测试模型。

## 5. 跨领域理论如何各司其职

| 理论领域 | 在本方法中的职责 | 不承担的主张 |
|---|---|---|
| 决策论 | 条件期望与argmax定义真正学习对象 | 不保证有限网络成功 |
| 概率预测 | MSE/BCE分别资格化连续均值与离散门概率 | 不把低loss等同驾驶收益 |
| 控制与车辆动力学 | 因果、确定性计算candidate-history兼容项 | 不从不可观测未来获得信息 |
| 结构化学习 | 固定效用代数连接两个heads并保持native-relative关系 | 不把公式重写本身称创新 |
| 风险敏感决策 | 后续单独处理低概率高代价false intervention | 不用同训练集阈值冒充保证 |
| 视觉表示学习 | 后续Differential Query只负责未知scene outcome证据 | 不让视觉网络重学已知动力学 |

## 6. 最小证伪顺序

### E3E0：解析时序接口资格

不训练。验证`C_i`因果可用、exact、确定、跨候选完整，并测量时延。失败即STOP该结构。

### E3E1：训练域结构资格

同fold-0、同seed、同backbone、同数据顺序和训练预算，对比：

1. E3D monolithic contextual mean；
2. factorized `F/H + analytic C`。

参数与FLOP差异必须通过matched dummy/readout预算控制。主要门仍是raw `G_model>0`、absolute oracle
capture、有益/有害干预与完整replay；component loss只作机制诊断。

### E3E2：组外确认

只有E3E1通过才进行6-fold×seed OOF，并加入correct/mismatch/removed。只有correct在relation、regret与
actual gain上同时优于控制，才重新打开视觉结构研究。

## 7. 与ICCV主贡献的关系

Structured Utility Factorization首先是由失败证据推出的**学习地基**，不是自动成立的主创新。AORTA
最终唯一主贡献候选仍是appearance-interventional differential decision evidence：在解析动力学项被正确
处理后，Differential Query是否比matched generic scorer更有效读取未知、外观敏感的scene outcome。

如果factorized generic scorer已经解决全部问题，则应诚实收缩论文，不强行保留Differential Query；如果
factorization只解决temporal tail，而DQ进一步改善跨外观scene outcome，二者才形成“已知结构解析处理、
未知视觉差异学习”的连贯方法故事。
