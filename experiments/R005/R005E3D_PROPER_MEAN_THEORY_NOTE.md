# R005E3D Proper-Mean 理论注记

## 1. 决策对象

令 `X` 表示决策时可用信息，包括图像、合法候选panel、候选几何与历史。给定 `X=x` 时合法候选集合
为 `A(x)`，native索引为0。候选随机效用为 `U_i`，定义native-relative效用

\[
Y_i=U_i-U_0,\qquad Y_0=0.
\]

部署目标不是恢复一次实现中的hindsight winner，而是最大化条件期望效用：

\[
i^*(x)\in\arg\max_{i\in A(x)}\mu_i(x),
\qquad \mu_i(x)=\mathbb E[U_i\mid X=x].
\]

由于对固定`x`，`E[U_0|X=x]`对所有候选相同，因此也等价于最大化
`m_i(x)=E[Y_i|X=x]`。

## 2. 条件均值恢复定理

**定理1（输入可测正权重下的proper mean）。** 假设每个合法challenger的`Y_i`平方可积；权重
`w_i(X)>0`只依赖`X`和合法panel，不依赖`Y`；函数类允许对每个`(X,i)`取任意平方可积值。则总体风险

\[
\mathcal R(s)=
\mathbb E\!\left[
\sum_{i\in A(X)\setminus\{0\}}w_i(X)(s_i(X)-Y_i)^2
\right]
\]

的几乎处处唯一最优解为

\[
s_i^*(X)=\mathbb E[Y_i\mid X].
\]

因此在固定tie规则下，`argmax_i s_i^*(X)`是条件期望效用的Bayes决策。

**证明。** 对任意固定`X=x`和合法`i`，条件风险可分解为

\[
\begin{aligned}
\mathbb E[(s_i-Y_i)^2\mid X=x]
&=(s_i-\mathbb E[Y_i\mid X=x])^2\\
&\quad+\operatorname{Var}(Y_i\mid X=x).
\end{aligned}
\]

乘以严格为正且`X`可测的`w_i(x)`不改变唯一极小点。对全部合法坐标求和仍逐坐标分离，故结论成立。
又因为

\[
\mathbb E[U_i\mid X]
=\mathbb E[U_0\mid X]+\mathbb E[Y_i\mid X],
\]

所有候选共享第一项，所以两种argmax相同。证毕。

R005E3D中每个anchor等权、anchor内challenger等权，权重由合法candidate mask决定，满足“只依赖X”
的设计条件。zero-challenger anchor没有可学习坐标，风险贡献为0；它不提供该anchor上的选择证据。

## 3. 预测误差到条件决策regret的桥

令`\hat i=argmax_i s_i(X)`、`i^*=argmax_i m_i(X)`，并固定相同tie规则。逐输入有

\[
\begin{aligned}
m_{i^*}-m_{\hat i}
&=(m_{i^*}-s_{i^*})+(s_{i^*}-s_{\hat i})
 +(s_{\hat i}-m_{\hat i})\\
&\le 2\max_{i\in A(X)}|s_i-m_i|\\
&\le 2\left(\sum_{i\in A(X)}(s_i-m_i)^2\right)^{1/2}.
\end{aligned}
\]

第二步使用`\hat i`最大化预测分数，故`s_{i^*}-s_{\hat i}\le0`。这给出一个清楚但不宽松保证：
只有条件均值估计误差在关键候选上足够小，选择regret才会小。平均MSE下降本身不保证有限样本hard choice
改善，尤其在near-tie、异方差或错误集中于top candidates时。

若所有合法坐标权重存在统一下界`w_min>0`，则平方误差风险可以进一步控制上述未加权L2误差；当前
ragged panel的anchor内归一化会让权重随候选数变化，因此正式报告仍必须直接包含hard regret与实际gain，
不能只引用该界。

## 4. 为什么它针对E3B失败

soft selection-regret的score梯度为

\[
\frac{\partial L_{SR}}{\partial s_i}
=\frac{p_i}{T}(\mathbb E_p[U]-U_i).
\]

当错误压低的优候选满足`p_i\to0`时，即使utility gap非零，纠错信号也指数衰减。平方目标的score梯度为

\[
\frac{\partial L_{mean}}{\partial s_i}=2w_i(X)(s_i-Y_i),
\]

不乘当前action probability。若优候选被压到极低分而target为正，其score residual不会因选择概率低而
自动消失。这只修复score层的尾部因子；参数Jacobian退化、共享参数梯度冲突、有限容量和优化器失败仍
可能使网络不能恢复条件均值。

## 5. 保证失效条件

以下任一情况都阻止把定理直接外推为实验成功：

1. **Utility-dependent weighting**：若权重依赖已实现`Y`，最优解变为加权条件均值，候选排序可能改变；
2. **有限函数类**：神经网络得到的是受限投影，不保证逐坐标条件均值；
3. **有限样本与组依赖**：outer-train成功不证明未见recording groups；
4. **优化不达全局风险最小点**：AdamW、BF16和有限步数可能失败；
5. **信息不足**：若`X`不能区分关键潜在状态，条件均值本身可能偏向native或接近tie；
6. **候选支持变化**：训练与部署候选机制不同会改变`A(X)`和条件分布；
7. **代理目标限制**：`U_ctx`不是完整pseudo-closed-loop policy value；
8. **外观移位**：source-fit条件均值不自动在NavDream成立。

## 6. 与实验门的对应

| 理论对象 | R005E3D观测 | 能否证明 |
|---|---|---|
| score层尾部纠错是否恢复 | epoch path、score分布、relation MSE | 只能检验冻结训练动力学 |
| 是否产生真实选择收益 | `mean_gain_over_native` | 训练域资格 |
| 是否获取足够机会 | absolute oracle capture | 训练域资格 |
| 是否摆脱native-only | nonnative与beneficial counts | 训练域资格 |
| 未见组是否成立 | 本轮不测 | 后续OOF实验 |
| 视觉是否不可替代 | 本轮不测 | correct/mismatch/removed |
| Differential Query是否必要 | 本轮不测 | matched architecture study |
| 外观/行为/安全 | 本轮不测 | NavDream与策略级评估 |

因此R005E3D即使PASS，也只闭合“proper目标使当前generic scorer在冻结训练域获得非退化决策学习资格”
这一命题；它不是AORTA最终方法贡献。
