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

## 7. 有限输出域上的可识别性条件

令网络只在有限节点域 \(\Omega\) 上输出任务场 \(u_\Omega\)，候选离散查询集合为
\(F(\tau)\)。完整候选margin写成

\[
M(\tau;u)=\min_{q\in F(\tau)}u(q).
\]

**命题（域外set-min不可识别）。** 若 \(F(\tau)\not\subseteq\Omega\)，且没有给定
\(u\) 在 \(\Omega^c\) 上的延拓类、边界条件或有效误差界，则
\(M(\tau;u)\) 不能由 \(u_\Omega\) 唯一确定。

**证明。** 取任意域内场 \(u_\Omega\) 以及一个
\(q^*\in F(\tau)\setminus\Omega\)。构造两个与 \(u_\Omega\) 完全一致的延拓
\(u^{(1)},u^{(2)}\)，只令 \(u^{(1)}(q^*)=a\)、\(u^{(2)}(q^*)=b\)，其中 \(a,b\)
可任意选择为小于域内最小值的不同数。则两者具有相同观测 \(u_\Omega\)，但完整set-min
分别由 \(a,b\) 决定，故不可识别。证毕。

直接使用裁剪集合得到

\[
M_\Omega(\tau)=\min_{q\in F(\tau)\cap\Omega}u_\Omega(q)
\]

是另一个estimand。没有域外假设时，\(|M_\Omega-M|\)无有限统一上界。因此完整footprint
越界时fail closed是定义域条件，不是可通过数值tolerance放宽的启发式。

这给出三种互斥的合法接口：

1. 固定prefix使所有注册查询满足 \(F_k(\tau)\subseteq\Omega\)，但结论只针对该近时域；
2. 扩展 \(\Omega\) 并同步重建投影、teacher、网络输出与容量匹配训练；
3. 对域外候选abstain，并显式报告由 \(\tau\) 决定的选择性覆盖。

H7BS-0进一步观察到当前越界全部发生在前向`x_max`且集中于长行程anchor。这是post-hoc
机制证据，不是上述命题的前提，也不能单独选择prefix或新ROI。

H7BS1使用不读取field、teacher或decision outcome的fit-only规则，在冻结字典中选出
`k=25`（2.5秒）为最大共同可行前缀，并在24/24 development groups上通过一次性支持
资格。因此当前唯一已知可查询的固定网格estimand是

\[
M_{2.5}(\tau;u)=\min_{q\in F_{25}(\tau)}u(q),
\]

而不是原完整4秒的 \(M_4\)。支持包含成立只消除了定义域不可识别性；它既不推出
\(M_{2.5}\) 的视觉误差足够小，也不推出其误差与ranking regret或flip相关，更不推出
2.5秒包含区分完整4秒候选所需的信息。若
\(F_{25}(\tau_i)=F_{25}(\tau_j)\)，则对任意场 \(u\) 都有
\(M_{2.5}(\tau_i;u)=M_{2.5}(\tau_j;u)\)；后段分叉不能由任何更强视觉backbone或更低
预测误差恢复。这一结构性不可区分性必须在读取decision outcome前单独审查。

H7BS2在fit上构造了这种反例。令无限、同相位的1m lattice离散集合为
\(F_k(\tau)\)。若 \(F_{25}(\tau_i)=F_{25}(\tau_j)\)，则对所有 \(u\) 均有相同
\(M_{2.5}\)。若 \(F_{40}(\tau_i)\ne F_{40}(\tau_j)\)，从集合差取一点赋负值即可构造
能由完整时域区分两者的场。实验在36/42 groups发现128个这样的eligible pairs，因此
2.5秒通用充分性已被否定，而不是“尚未证明”。

完整候选域还要求

\[
\forall \tau\in\mathcal C_{eligible}(I),\qquad F_{40}(\tau)\subseteq\Omega.
\]

“每个anchor至少一个challenger可查”不蕴含该条件。H7BS2前移64×64虽通过旧panel门，
却只保留3574/3864 eligible challengers；前向96×64才是冻结字典中首个fit上满足全部
候选包含的方案。该结论只确定空间定义域候选，视觉网络是否能在该域预测、所需感受野、
误差随距离的增长及真实推理时延仍是独立假设。

H7BS3R1在不读取视觉场、teacher或decision outcome的条件下，对前向96×64进行一次性
development资格，得到

\[
\forall \tau\in\mathcal C_{eligible}^{dev},\qquad
F_{40}(\tau)\subseteq\Omega_{96},
\]

其中384/384 native与2206/2206 eligible challengers均满足包含。故该离散候选集合上的
完整4秒set-min现在具有已验证的查询定义域；这不推出相机观测能识别\(u\)在整个
\(\Omega_{96}\)上的值。若远距投影缺失、teacher误差随距离增长或decoder延迟超门，空间
包含仍不能形成可用方法。因此视觉可观测性与计算可实现性是下一阶段独立必要条件。
