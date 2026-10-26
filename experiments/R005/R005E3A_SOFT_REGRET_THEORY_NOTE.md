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

## 6. 命题四：总体条件风险的决策一致性

令条件变量

\[
Z=(B,V)
\]

包含合法候选panel、因果历史与当前视觉输入，并把合法候选mask视为(Z)的一部分。允许同一个
(Z=z)下的utility向量(U)仍有条件随机性，记

\[
\mu_i(z)=\mathbb E[U_i\mid Z=z].
\]

对任意只依赖(z)的score函数，soft selection-regret的条件风险为

\[
\begin{aligned}
\mathcal R_{SR}(s\mid z)
&=\mathbb E\left[\max_i U_i-\sum_i p_i(s;T)U_i\mid Z=z\right]\\
&=C(z)-\sum_i p_i(s;T)\mu_i(z),
\end{aligned}
\]

其中(C(z)=\mathbb E[\max_iU_i\mid Z=z])与(s)无关。因此总体最优score等价于在可实现的
softmax概率集合内最大化(p^\top\mu)。

在不受界logit下，若(arg\max_i\mu_i(z))唯一，则可令概率质量趋于该候选，且部署hard argmax
趋于条件期望效用最优决策。更一般地，对任意两个候选(i,j)，若(mu_i>\mu_j)但(s_i<s_j)，
交换二者score会严格提高(p^\top\mu)；这是重排不等式的直接二项形式。因此任何总体最优排序都不能
把较低条件期望效用的候选严格排在较高者之前。

当前实现还具有native gauge (s_0=0)与challenger `tanh`范围([-1,1])。交换论证仍排除
“较低utility候选具有严格更高score”，但**只能保证弱排序，不能无条件保证严格决策一致性**。多个
不同utility的高价值候选可能同时饱和在上界；此时部署tie-break可能选到其中的次优候选。比如在有限
score范围、很大温度且另有极低utility候选时，同时提高两个高utility候选的score可能比只提高严格最优
候选更能减少分给极低候选的概率，于是二者都达到上界。native固定为0也进一步限制了可实现概率集合。

因此，无界logit给出标准的条件期望决策一致性；当前有界实现只继承效用—分数的弱次序性质。要把它
升级为hard-choice一致性，还需要排除top-score饱和tie、规定与条件效用相容的tie-break，或给出额外的
temperature/utility-gap条件。R005E3A正是在实验上资格化“当前有界scorer + 冻结温度 + 冻结预算”这一
具体组合，而不是借总体定理预先保证成功。

该命题的重要作用是限定失败解释，而不是预告正结果：

1. R005E3A若失败，不能笼统归因为无界soft selection-regret在总体上优化了错误决策对象；但必须检查
   当前有界score是否造成top saturation tie；
2. 失败仍可能来自DINO表示未保留边界信息、有限网络不可实现所需条件排序、42组样本不足、优化未到达
   合适解、固定温度与有界score导致梯度条件不良，或训练分布与OOF组分布不同；
3. 条件期望最优不等于逐场景oracle最优，也不提供OOD、因果可迁移、概率校准或安全保证；
4. 若合法mask或历史没有包含在(Z)中，以上条件风险分解不再对应部署决策问题，因此mask、history和
   坐标绑定仍是理论假设而非工程细节。

在有界score集合上，不额外引入utility margin、函数逼近误差和优化误差条件时，不能把soft-risk
**excess**直接写成无常数的hard-regret上界。R005E3A因此仍以OOF hard decision regret为唯一主端点，
而不以训练loss或上述总体一致性命题替代实验。

## 7. 对本轮结果的约束

- 只有42-group OOF hard-regret门可以资格化“简单目标错配解释”；
- S1/S2/S3/removed干预必须同时伤害regret，才能说明改善仍依赖正确场景视觉；
- relation MSE、训练loss、top-choice flips只作诊断；
- 本定理不涉及appearance orbit、NavDream、sequential policy、真实闭环或安全。

因此，R005E3A是对一个最简单机制解释的严格淘汰实验，不是AORTA最终算法。

## 8. Post-result falsification：有界参数化的吸收式退化

正式结果之后，18/18个primary checkpoint在各自outer-train上均满足：所有合法challenger
`s_i=-1`、native `s_0=0`、hard选择恒为native。所有job的冻结温度均为`T=0.125`。对已处于
负饱和的challenger，logit梯度还需乘以`tanh`的Jacobian：

\[
\frac{\partial L}{\partial z_k}
=\frac{p_k}{T}(\mathbb E_p[u]-u_k)(1-\tanh^2 z_k).
\]

当`s_k\approx-1`时，`p_k`相对native含`\exp(-1/T)=\exp(-8)`量级，同时
`1-\tanh^2z_k\to0`。两项共同构成近吸收低梯度区。这个结果否定了“当前有界scorer、冻结温度和预算
可以有效资格化soft-regret”的实现假设，但不否定无界总体条件风险命题。

## 9. 必须区分相对修复与绝对机会捕获

定义

\[
G_{oracle}=\mathbb E[U^*-U_0],\quad
G_{model}=\mathbb E[U_{\hat i}-U_0],\quad
R_{model}=\mathbb E[U^*-U_{\hat i}].
\]

逐定义有

\[
\boxed{G_{oracle}=G_{model}+R_{model}}.
\]

本轮`G_model=0`而`R_model=G_oracle`。因此
`(R_parent-R_model)/G_oracle=0.159175`只说明模型相对有害parent回到native，不能解释为捕获了
15.9%的oracle收益。未来正向门必须同时要求`G_model`组级CI下界为正和
`G_model/G_oracle`达到预注册阈值；相对parent改善只能作为附属比较。
