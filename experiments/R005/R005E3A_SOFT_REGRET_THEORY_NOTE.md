# R005E3A Soft Selection-Regret 理论说明

日期：2026-10-26  
状态：`PRE_RESULT_THEORY_NOTE__DOES_NOT_CHANGE_THE_FROZEN_PROTOCOL`

## 1. 对象

对一个anchor的合法候选集合(V)，令(n=|V|\geq1)，真实contextual utility为
(u=(u_i)_{i\in V})，网络分数为(s=(s_i)_{i\in V})，且(T>0)。定义

\[
p_i(s;T)=\frac{e^{s_i/T}}{\sum_{j\in V}e^{s_j/T}},
\qquad
u^*=\max_{i\in V}u_i,
\]

\[
\mathcal L_{SR}(s,u;T)
=u^*-\sum_{i\in V}p_i(s;T)u_i
=\sum_{i\in V}p_i(s;T)(u^*-u_i).
\]

这里(s_0=u_0=0)表示native-relative gauge；非法候选不进入(V)。该loss是固定candidate panel上的
listwise、utility-sensitive decision surrogate，不是条件margin分布的proper scoring rule。

## 2. 命题一：非负性、tie兼容性与梯度方向

对任意有限(V)、有限(s,u)与(T>0)：

\[
\mathcal L_{SR}\geq0,
\qquad
\frac{\partial\mathcal L_{SR}}{\partial s_k}
=\frac{p_k}{T}\left(\sum_i p_i u_i-u_k\right).
\]

证明：第一式由(u^*-u_i\geq0)及(p_i\geq0)直接得到。第二式由softmax Jacobian
(partial p_i/partial s_k=p_i(\mathbf1[i=k]-p_k)/T)代入即可。

因此，高于当前soft选择期望utility的候选得到负梯度并被提高分数，低于期望的候选被降低。若多个候选
具有相同最优utility，loss只关心落在整个最优集合上的概率质量，不需要人为指定唯一class。若只有native
合法，则(n=1)、loss与梯度均精确为零。

## 3. 命题二：soft surrogate与hard argmax regret的有限候选桥

令(hat i\in\arg\max_i s_i)为部署时hard选择，定义

\[
R_{hard}(s,u)=u^*-u_{\hat i}.
\]

由于(hat i)也是softmax概率最大者，(p_{\hat i}\geq1/n)。于是

\[
\mathcal L_{SR}
=\sum_i p_i(u^*-u_i)
\geq p_{\hat i}(u^*-u_{\hat i})
\geq \frac1nR_{hard},
\]

从而

\[
\boxed{R_{hard}(s,u)\leq n\,\mathcal L_{SR}(s,u;T)}.
\]

AORTA冻结panel最多7个合法候选，因此逐anchor有(R_{hard}\leq7\mathcal L_{SR})。若top-score与
second-score之间的分数间隔为(delta\geq0)，则

\[
p_{\hat i}\geq\frac1{1+(n-1)e^{-\delta/T}},
\]

可进一步得到

\[
R_{hard}\leq\left[1+(n-1)e^{-\delta/T}\right]\mathcal L_{SR}.
\]

这个界说明soft loss确实控制hard regret，但最坏常数随候选数增大；它不保证有限样本训练一定降低
hard regret，也不允许用训练loss替代OOF hard-regret实验。

## 4. 命题三：总体最优方向与有限实现边界

若scores不受界且存在严格次优utility gap，则把概率质量集中到(arg\max u)可使
(inf_s\mathcal L_{SR}=0)。反过来，若最优集合外存在最小正gap
(gamma=\min_{i:u_i<u^*}(u^*-u_i)>0)，则

\[
\mathcal L_{SR}\geq
\gamma\sum_{i:u_i<u^*}p_i.
\]

所以loss趋于零会迫使次优集合的概率质量趋于零。这是固定panel、已知utility下的selection-consistency
方向，不是未知分布上的泛化定理。

当前`FullContextSetScorer`使用`tanh`，故(s_i\in[-1,1])。在有限(T)下，softmax不能把次优概率
精确压到零，存在由score范围和candidate count决定的不可约soft-loss floor。因此：

1. 不以训练loss是否接近零作为成功条件；
2. 只比较compute-matched OOF hard decision regret；
3. 若失败，不能直接推断decision-aligned learning不可能，只能否定当前有界scorer、冻结温度与预算组合；
4. 若成功，也不产生概率校准、完整策略价值或安全保证。

## 5. 温度规则的理由与边界

R005E3A令(T)等于outer-train严格正regret gaps的中位数。这样做的目的只有三个：

- 使用utility自身尺度，避免任意单位常数；
- 让典型正gap满足gap/(T\)约为1，减少全均匀或全饱和两个极端；
- 每fold只有一个确定值，不形成超参数搜索自由度。

该规则不是最优温度定理，也不保证最佳优化条件；偶数gap取两个中央次序统计量的算术平均，ties以
`1e-12`排除。任何替代温度都属于新实验，不能用于改写R005E3A。

## 6. 对本轮结果的约束

- 只有42-group OOF hard-regret门可以资格化“简单目标错配解释”；
- S1/S2/S3/removed干预必须同时伤害regret，才能说明改善仍依赖正确场景视觉；
- relation MSE、训练loss、top-choice flips只作诊断；
- 本定理不涉及appearance orbit、NavDream、sequential policy、真实闭环或安全。

因此，R005E3A是对一个最简单机制解释的严格淘汰实验，不是AORTA最终算法。
