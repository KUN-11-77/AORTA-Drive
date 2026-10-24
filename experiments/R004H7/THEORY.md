# R004H7 理论资格：从 graph quotient 到 decision quotient

状态：`THEORY_SPECIFICATION_FROZEN_FOR_REVIEW`。本文件给出 R004H7 的数学对象、可证明命题和禁止扩张的主张边界；它不是实验结果。

## 1. 图势表示与其真实表达能力

令 \(G=(V,E)\) 为有限连通图，\(D\in\mathbb R^{|E|\times|V|}\) 为固定方向的 incidence matrix。节点势为 \(u\in\mathbb R^{|V|}\)，边差分为 \(g=Du\)。

### 命题 1：图商空间与可积边空间同构

映射

\[
\bar D:\mathbb R^{|V|}/\operatorname{span}\{\mathbf 1\}
\rightarrow \operatorname{im}D,
\qquad [u]\mapsto Du
\]

是线性同构。

证明：共同常数不改变 \(Du\)，所以映射良定义。连通图满足
\(\ker D=\operatorname{span}\{\mathbf1\}\)，故商空间上的映射单射；
\(\operatorname{im}D\) 的定义给出满射。

因此，node potential 与经过精确 cut-space/Hodge projection 的 direct edge 在表示能力上等价。AORTA 不得把 potential parameterization 本身写成表达能力创新；它只能作为 integrability、优化、样本效率或查询兼容性的候选归纳偏置。

## 2. 精确候选决策映射

对一个冻结候选面板

\[
\mathcal C=(A_0,A_1,\ldots,A_K),\qquad
\varnothing\ne A_j\subseteq V,
\]

\(A_0\) 为 native 的离散扫掠节点集合。定义精确 set-min contrast：

\[
T_{\mathcal C}(u)_j
=\min_{q\in A_j}u_q-\min_{q\in A_0}u_q,
\qquad j=1,\ldots,K.
\]

它满足 \(T_{\mathcal C}(u+c\mathbf1)=T_{\mathcal C}(u)\)。定义

\[
u\sim_{\mathcal C}v
\Longleftrightarrow
T_{\mathcal C}(u)=T_{\mathcal C}(v).
\]

这是等价关系，因而定义 quotient set。由于 set-min 是分段线性的，
\(\mathbb R^{|V|}/\!\sim_{\mathcal C}\) 一般不是线性商空间；论文不得在全局把它错误写成固定矩阵 kernel quotient。

若一个 active cell 内每个集合的确定性最小节点 \(a_j\in A_j\) 固定，则

\[
T_{\mathcal C}(u)_j=u_{a_j}-u_{a_0}=A_{\sigma,j}u,
\]

此时局部决策等价类才由线性 kernel \(\ker A_\sigma\) 给出。active node 改变时必须切换 cell；一阶近似不能冒充全局等式。

## 3. 决策误差的确定性界

对任意非空集合 \(A\)，记 \(f_A(u)=\min_{q\in A}u_q\)，并令
\(e=u-v\)。有

\[
\min_{q\in A}e_q
\le f_A(u)-f_A(v)
\le \max_{q\in A}e_q.
\]

于是对每个 challenger：

\[
|T_{\mathcal C}(u)_j-T_{\mathcal C}(v)_j|
\le
\operatorname{osc}_{A_j\cup A_0}(e),
\]

其中 \(\operatorname{osc}_S(e)=\max_{q\in S}e_q-\min_{q\in S}e_q\)。该界对共同 offset 不敏感。

若逐边满足 \(|(De)_r|\le w_r\)，则对图加权最短路距离
\(\operatorname{dist}_w\)：

\[
\operatorname{osc}_{A_j\cup A_0}(e)
\le
\max_{p,q\in A_j\cup A_0}\operatorname{dist}_w(p,q).
\]

这解释了 graph-gradient envelope 如何控制 decision contrast，但不保证界足够紧，也不保证候选中存在更优轨迹。

## 4. 排序保持与 regret

令真实 native-relative margins 为 \(m_j=T_{\mathcal C}(u^*)_j\)，预测为
\(\widehat m_j=T_{\mathcal C}(\widehat u)_j\)，并定义

\[
\eta=\max_j|\widehat m_j-m_j|.
\]

若 \(j^*=\arg\max_{j\in\{0,\ldots,K\}}m_j\)，
\(\widehat j=\arg\max_j\widehat m_j\)，且 native 的 margin 定义为 0，则

\[
m_{j^*}-m_{\widehat j}\le 2\eta.
\]

证明由 \(\widehat m_{\widehat j}\ge\widehat m_{j^*}\) 与两次绝对误差界直接得到。若 teacher 最优与次优的 gap 严格大于 \(2\eta\)，则 argmax 保持。所有 ties 使用冻结候选顺序确定性打破，并单独报告 exact-tie rate。

该命题只约束冻结离散 nominal-road query；它不推出碰撞、动态交互、交通规则、舒适性、连续几何或闭环驾驶安全。

## 5. 待检验的网络机制

候选方法保持 candidate-free 视觉输出：

\[
\widehat u_\theta(X)
=f_\theta(\mathcal F(X),\operatorname{PE},m),
\qquad
\mathbf1^\top\widehat u_\theta=0.
\]

候选只进入训练期与推理期的冻结查询算子 \(T_{\mathcal C}\)，不进入视觉 trunk。候选训练目标为

\[
\mathcal L_{DQ}
=\operatorname{mean}_{j}
\operatorname{Huber}_{0.25}
\left(
\frac{T_{\mathcal C}(\widehat u)_j-T_{\mathcal C}(u^*)_j}{1\,\mathrm m}
\right),
\]

并配合现有 edge loss 与 source-derived appearance consistency：

\[
\mathcal L
=\mathcal L_{edge}+\mathcal L_{DQ}+0.25\mathcal L_{app}.
\]

\(\mathcal L_{app}\) 对同一 candidate panel 比较 clean 与冻结受控外观下的 decision contrasts。系数在读取结构评估面板前冻结，不通过 test/target 或驾驶分数选择。

## 6. 理论主张边界

R004H7 允许检验：

1. decision-contrast error 是否比全局 field/DINO error 更贴近排序失败；
2. DQ supervision 是否在相同 backbone、数据、候选和预算下减少 ranking flip/regret；
3. potential parameterization 是否提供超出 projected-edge 的经验归纳偏置；
4. candidate-free field 是否比 candidate-conditioned direct-pair head 更能迁移到另一冻结候选策略。

R004H7 不允许宣称：

- potential 与 projected-edge 有不同表达能力；
- development 结果等同 untouched test；
- 离散 shadow ranking 等同连续几何安全；
- source appearance consistency 等同 NavDream 或任意 OOD 保证；
- 一次实验胜出即可证明 ICCV 新颖性。
