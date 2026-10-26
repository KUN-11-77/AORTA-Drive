# AORTA 第一性原理理论与系统规范 v1

状态：`CANONICAL_THEORY_CONTRACT__EMPIRICAL_CORE_PARTIALLY_UNTESTED`

本文件是 AORTA 当前 relation-first 研究路线的统一理论入口。它规定问题对象、基本假设、
数学结构、网络设计来源、训练与推理边界、证伪顺序以及论文主张资格。它不把尚未完成的实验
写成事实，也不替代各注册实验的冻结协议。历史 graph-gradient field、SDF、scalar set-min 与
几何证书保留为失败驱动的研究证据和基线，不再作为当前主方法的 canonical specification。

## 1. Problem Anchor

### 1.1 底线问题

冻结端到端驾驶 planner 在相同物理场景的不同外观观测下，可能产生不同甚至更差的候选与选择。
AORTA 不重新训练 candidate generator，而是在冻结 planner 提供的合法候选集合内，研究：

> 能否从视觉中恢复足以比较候选决策效用、对外观干预稳定的证据，并只在证据充分时覆盖 native？

### 1.2 必须解决的瓶颈

稳定视觉特征不自动推出稳定驾驶决策。全局特征一致、道路场拟合或几何裕度稳定，都可能不包含
区分候选排序所需的信息。方法必须同时回答：

1. 给定候选与因果历史后，视觉是否仍提供条件增量；
2. 哪种候选条件结构能更有效地读取该增量；
3. 外观变化下，保持的是正确任务关系而非平凡常数；
4. 证据不足时，如何避免错误覆盖 native；
5. 离线关系改善是否能转化为跨 planner、连续历史下的行为收益，并满足时延预算。

### 1.3 Non-goals

- 不学习 candidate generator；
- 不恢复完整世界状态作为必要目标；
- 不把道路可行性扩展为碰撞、交通规则或完整闭环安全保证；
- 不把 DINO、trajectory query、cross-attention、Set Transformer、conformal 或 fallback 单独称为创新；
- 不把 source calibration 外推为任意 OOD 保证；
- 不把 `U_ctx` 冒充官方完整 two-stage action value或策略级价值。

## 2. 从潜在世界到决策对象

令：

- \(S\) 为道路、动态参与者、信号与其他任务状态组成的潜在物理世界；
- \(A\) 为天气、光照、风格等外观因素；
- \(I=g(S,A,\epsilon)\) 为相机观测；
- \(B=(\mathcal T,h)\)，其中 \(\mathcal T=\{\tau_0,\ldots,\tau_K\}\) 是合法候选集合，
  \(h\) 是因果可用的上一周期历史；
- \(U(\tau_i;S,h)\) 为明确作用域内候选 \(i\) 的上下文效用。

理想决策是：

\[
\pi^\star(I,B)
=\arg\max_{i\in V_B}
\mathbb E[U(\tau_i;S,h)\mid I,B],
\]

其中 \(V_B\) 是冻结、合法、可变长度的候选域。地图、BEV、风险场与视觉 token 都只是可能的
中间表示；基本问题是有限观测下的条件决策。

## 3. 最小 estimand：native-relative contextual relation

固定 \(\tau_0\) 为 native，定义：

\[
\Delta_{i0}
=U(\tau_i;S,h)-U(\tau_0;S,h),
\qquad i\in V_B\setminus\{0\}.
\]

候选替换只依赖 \(\Delta_{i0}\) 的大小和符号。任何对全部候选相同的效用偏置都会抵消。因此当前
最小部署对象是 native-star margin 向量，而不是绝对地图、绝对安全零点或自由 complete-pair
关系。任意 challenger 间关系由

\[
\Delta_{ij}=\Delta_{i0}-\Delta_{j0}
\]

导出，避免循环偏好；这种可积性是接口正确性，不是论文创新。

当前已资格化的训练对象是 native-history、teacher-forced、row-level 的 \(U_{ctx}\)。它不同于：

\[
F_{raw}^{1f},\quad \widetilde F^{1f},\quad U_{ctx},\quad
Q_{\pi_{ref}},\quad J(\pi).
\]

任何主张都必须标出自己对应的对象。

### 3.1 潜在决策商与观测预测商

首先在物理状态上定义真正的任务对象。对固定候选上下文 \(B\)，令

\[
S\equiv_B^{\mathrm{lat}}S'
\iff
P(Y_B\mid S,B)=P(Y_B\mid S',B),
\]

并记潜在决策商变量为

\[
Q_B=q_B(S)=[S]_{\equiv_B^{\mathrm{lat}}}.
\]

它删除所有不改变候选相对效用分布的物理细节。另一方面，有限图像只能诱导对该潜在商变量的信念。
两个观测在当前数据分布下预测等价，当且仅当

\[
I\equiv_B^{\mathrm{obs}} I'
\iff
P(Y_B\mid I,B)=P(Y_B\mid I',B).
\]

观测预测商是 Bayes 决策可用的规范对象，但它不应与潜在决策商混同：不同外观可能改变可见信息，
从而使同一物理状态的两个图像诱导不同宽度的后验。最终网络应逼近
\(P(Y_B\mid I,B)\) 或等价的决策统计量，并在信息不足时表达不确定性，而不是假设每个图像都能
确定性恢复 \(Q_B\)。

### 3.2 外观轨道可恢复性的充要条件

先忽略观测噪声。给定合法外观集合 \(\mathcal A\)，存在确定性映射 \(r_B\)，使得

\[
r_B(g(S,a))=q_B(S),
\qquad \forall S,\ a\in\mathcal A,
\]

当且仅当满足碰撞一致性：

\[
g(S_1,a_1)=g(S_2,a_2)
\Longrightarrow
q_B(S_1)=q_B(S_2)
\]

对所有合法 \((S_1,a_1),(S_2,a_2)\) 成立。

证明：必要性由对同一观测应用 \(r_B\) 立即得到。充分性下，对任意可达观测 \(i\)，从其任一原像
\((S,a)\) 定义 \(r_B(i)=q_B(S)\)；碰撞一致性保证定义与原像选择无关。证毕。

这个充要条件把两件事严格分开：外观可以是**任务保持的**，即不改变 \(q_B(S)\)；但它不一定是
**信息保持的**，因为低照度、遮挡或渲染伪影可能让不同潜在商类碰撞到不可区分观测。后者失败时，
任何确定性“外观不变且决策充分”表示都不存在；正确目标是后验不确定性扩大与拒判，而不是强迫
feature 或point prediction保持不变。带噪声情形由后续的观测纤维与条件分布处理。该命题固定候选
上下文 \(B\)；当 planner 的候选供给本身随外观变化时，还必须单独分析 candidate-supply regret，不能
用选择器的可恢复性结论覆盖它。

最小反例说明“任务保持”不推出“观测预测等价”。令 \(S\in\{0,1\}\)、\(Y_B=S\)，clean 外观满足
\(g(S,a_c)=S\)，dark 外观满足 \(g(S,a_d)=\bot\)（完全丢失信息）。两种外观都不改变真实 \(Y_B\)，
但 dark 观测把两个潜在商类碰撞到同一图像：clean 后验是对应状态的点质量，dark 后验只能退化为先验。
不存在同时对两个状态正确的确定性不变映射。故未来 appearance 实验若只奖励point drift小，会偏好
过度自信的常数预测；必须同时评价任务风险、预测分布、coverage/width与选择性拒判。

## 4. 视觉价值的第一性检验

令 \(C\) 为候选几何与合法 mask，\(H\) 为因果历史，\(Z=\Phi(I)\) 为视觉表示，目标向量为
\(Y=(\Delta_{10},\ldots,\Delta_{K0})\)。比较：

\[
X_0=(C,H),\qquad X_1=(C,H,Z).
\]

在带合法 mask 的加权平方可积空间中，平方损失 Bayes 风险满足：

\[
\mathcal R^\star(X_0)-\mathcal R^\star(X_1)
=\mathbb E\!\left[
\left\|
\mathbb E[Y\mid X_1]-\mathbb E[Y\mid X_0]
\right\|^2
\right]\ge 0.
\]

因此视觉的理论价值不是“特征丰富”或“表示相似”，而是它是否改变给定候选与历史后的目标条件
均值。R005E2 用 correct、geometry-matched mismatch、removed 三臂和同 checkpoint 输入干预，
操作化检验有限函数类能否利用该增量。

边界：有限网络未改善可能来自信息缺失、表示损失、函数类限制、估计误差或优化失败，不能直接
证明条件互信息为零。

## 5. 外观鲁棒性不是全局特征不变

相同物理场景的不同外观记为 \(I^{a_1},I^{a_2}\)。要求

\[
\Phi(I^{a_1})\approx\Phi(I^{a_2})
\]

既非必要也非充分；常数表示可以完美稳定却没有任务信息。AORTA 关心的是条件决策关系，但先区分
外观是否保留商变量的可辨识信息。对信息保持的合法外观，定义：

\[
m_i^{(a)}
=\mathbb E[\Delta_{i0}\mid I^{(a)},B].
\]

目标应同时满足：

\[
m_i^{(a_1)}\approx m_i^{(a_2)},
\qquad
m_i^{(a)}\approx
\mathbb E[\Delta_{i0}\mid S,B].
\]

即“任务信息保留 + 外观条件稳定”。对任务保持但信息破坏的外观，不要求point prediction机械相等；
要求其条件分布保持校准、区间适当变宽，并在无法辨识margin符号时拒绝覆盖native。任何 appearance
consistency loss 都必须与 task/regret、uncertainty/coverage、abstention 和 non-collapse control
同时报告，不能用稳定性掩盖错误预测或过度自信。

## 6. 可辨识性边界

令当前模型可用观测为 \(O=(I,B)\)，定义观测纤维：

\[
\mathcal F(O)
=\{S:\ g(S,A,\epsilon)=I,\ B(S)=B\}.
\]

若存在 \(S_1,S_2\in\mathcal F(O)\)，使

\[
\Delta_{i0}(S_1)>0,
\qquad
\Delta_{i0}(S_2)<0,
\]

则候选 \(i\) 相对 native 的优劣在当前观测下不可符号辨识。此时任何确定性 point scorer 都不能
对所有相容世界保证正确。代数不变性不等于统计可辨识性；relative relation 仍可能因遮挡、未来
交通反应或表示压缩而不可辨识。

这一边界给出两个直接后果：

1. 网络最终应估计条件分布或同时区间，而不是把单点置信度冒充确定事实；
2. 证据不足时的正确动作是拒绝覆盖 native，而不是强制选择 challenger。

## 7. 为什么需要完整时空 signed differential support

将候选 \(i\) 的完整时域车辆占据表示为测度 \(\mu_i\)。相对 native 定义 signed difference：

\[
\nu_{i0}=\mu_i-\mu_0=\nu_{i0}^{+}-\nu_{i0}^{-}.
\]

其中 \(\nu^+\) 表示 challenger 额外访问的时空区域，\(\nu^-\) 表示 native 独占区域。对加性
任务泛函，差值满足：

\[
J(\tau_i;S)-J(\tau_0;S)
=\int c_S(x,t)\,d\nu_{i0}(x,t).
\]

这解释了正负差异支持为何是自然 query geometry，但该恒等式不是 AORTA 的原创，也不证明完整
NAVSIM utility 是加性的。交通交互、信号、舒适性与 future-policy effect 可能产生非加性项：

\[
\Delta_{i0}
=\int \psi(S)\,d\nu_{i0}
+\Gamma(S,\tau_i,\tau_0,h).
\]

因此最终机制必须遵循：

> signed differential support 负责提出“在哪里、何时、以什么符号比较”；全部全局视觉与历史
> token 负责回答“哪些上下文决定完整任务关系”。

差异支持是 soft geometric bias，不是硬裁剪或可见性证书。

## 8. 由理论推出的网络架构

### 8.1 数据流

\[
Z=\Phi_\omega(I_{t-H:t})
\]

\[
q_{i0}
=G_\eta(\nu^+_{i0},\nu^-_{i0},\tau_i,\tau_0,h)
\]

\[
r_{i0}
=\operatorname{CrossAttn}(q_{i0},Z,Z; b_{geo})
\]

\[
\{\widetilde r_{i0}\}
=\operatorname{SetModule}(\{r_{i0}\},\text{legal mask})
\]

\[
p_\theta(\Delta_{i0}\mid I,B)
=H_\theta(\widetilde r_{i0}).
\]

### 8.2 每层职责

| 层 | 必须解决的问题 | 删除实验 |
|---|---|---|
| 共享时空视觉编码 | 世界中有什么任务证据 | correct / mismatch / removed |
| signed differential query | 两个行为在哪里、何时不同 | differential vs ordinary query |
| 全局 visual key/value | 非加性与远距上下文 | local-only vs global |
| permutation-equivariant set interaction | 合法候选间竞争 | independent vs set scorer |
| 最小因果 history | `U_ctx` 的上下文定义 | current-only vs minimal history |
| 条件分布/同时区间 | 表达不可辨识与联合误差 | point vs distributional head |

视觉 backbone 每个 anchor 只执行一次；所有候选 query 必须向量化，候选数不得线性增加 backbone
调用次数。

### 8.3 当前 R005E2 与最终网络的区别

R005E2 使用通用 full-context set scorer，只检验正确视觉的条件增量。它不含 signed differential
support、appearance orbit loss、分布校准或 selective arbitration。只有 R005E2 通过，才允许冻结
参数、FLOP、token、数据和 seed 匹配的 generic-vs-differential 架构实验。

## 9. Loss 的分层资格

### 9.1 当前信息探针

R005E2 使用 group/anchor/challenger 等权平方 margin loss：

\[
\mathcal L_{probe}
=\mathbb E\|Y-f_\theta(C,H,Z)\|_2^2.
\]

它与条件均值 Bayes 恒等式同型，适合检验视觉增量；但不直接优化 regret，也不是最终 loss。

### 9.2 最终候选目标

只有对应机制门通过后，最终目标才可由以下有职责的项组成：

\[
\mathcal L
=\mathcal L_{proper}
+\lambda_r\mathcal L_{rank}
+\lambda_a\mathcal L_{appearance}
+\lambda_c\mathcal L_{calibration}.
\]

- \(\mathcal L_{proper}\)：学习条件 margin 分布；
- \(\mathcal L_{rank}\)：若误差集中在决策边界，控制候选符号和顺序；
- \(\mathcal L_{appearance}\)：只约束同场景外观轨道上的任务关系；
- \(\mathcal L_{calibration}\)：只在独立 calibration 上构造同时下界。

不得为了“loss 丰富”而同时启用所有项。每一项必须有独立失败证据、删除消融和预注册权重选择
规则；否则删除。

## 10. 保守仲裁的条件定理

若对全部合法 challenger 同时成立

\[
|\widehat\Delta_{i0}-\Delta_{i0}|\le\epsilon_i,
\]

则下界

\[
L_{i0}=\widehat\Delta_{i0}-\epsilon_i
\]

满足：只要 \(L_{i0}>\gamma\)，即可在该同时覆盖事件和既定 estimand 作用域内推出

\[
\Delta_{i0}>\gamma.
\]

相应选择规则为：

\[
\pi_\theta(I,B)=
\begin{cases}
\tau_{i^\star}, &
i^\star=\arg\max_i L_{i0},\ L_{i^\star0}>\gamma,\\
\tau_0, & \text{otherwise}.
\end{cases}
\]

该结论是有假设、有限作用域的相对改善合同，不是现实道路安全证明。source exchangeability、有限
calibration group 数、candidate mechanism 变化和 appearance OOD 都必须显式报告。

## 11. 实时约束是方法定义的一部分

在进入部署约束前，必须先区分候选供给与候选选择。令全集最优效用为 \(U^\star\)，冻结 planner
在观测 \(I\) 下给出的候选集合为 \(B(I)\)，集合内最优为 \(U_B^\star\)，最终选择为
\(\widehat\tau\in B(I)\)。则

\[
U^\star-U(\widehat\tau)
=
\left(U^\star-U_B^\star\right)
+
\left(U_B^\star-U(\widehat\tau)\right).
\]

第一项是 candidate-supply regret，第二项是 selection regret。AORTA 只直接优化第二项；固定 panel
实验隔离 selection 机制，最终跨 appearance/planner 行为实验必须同时报告两项，不能把候选供给变化
冒充 reranking 收益。

最终优化问题是：

\[
\min_\theta \mathcal R_{decision}(\theta)
\quad\text{s.t.}\quad
T_{P95}^{batch=1}(\theta)\le T_{max},\quad
M_{GPU}(\theta)\le M_{max}.
\]

必须报告端到端 P50/P95、峰值显存、候选数 \(K\) 的扩展曲线和 candidate generator 之外的增量
时延。训练 GPU 利用率不是部署时延证据。

## 12. 证据门与不可跳过的顺序

| Gate | 科学问题 | 当前状态 |
|---|---|---|
| G0 Estimand | 合法 `U_ctx` 是否有候选 oracle headroom | source-fit 已支持 |
| G1 Information | 正确视觉是否提供可学习条件增量 | generic MSE probe未支持；soft-regret出现native-only优化塌缩，尚未有效裁决信息存在性 |
| G2 Inductive bias | Differential Query 是否胜过 matched generic scorer | 未验证 |
| G3 Robustness | 外观变化下是否稳定且保留任务信息 | 未验证 |
| G4 Reliability | 同时区间是否 non-vacuous 且降低错误干预 | 未验证 |
| G5 Behavior | actual-history、sequential、跨 planner 是否有净收益 | 未验证 |
| G6 Deployment | clean no-harm 与 batch-1 P95 是否通过 | 未验证 |

下游门不得回填上游主张；oracle ceiling、视觉 MAE 或离线 relation accuracy 都不是驾驶收益。

## 13. STOP 后的第一性诊断

失败必须依次定位：

\[
\text{信息存在性}
\rightarrow
\text{表示保留性}
\rightarrow
\text{函数可逼近性}
\rightarrow
\text{损失决策一致性}
\rightarrow
\text{部署可用性}.
\]

| 观察 | 允许的解释与动作 |
|---|---|
| correct ≈ removed 且 correct ≈ mismatch | 当前表示无可恢复增量；做 observability/representation probe，禁止直接加复杂主架构 |
| correct 胜 removed、不胜 mismatch | 场景绑定或 donor 混杂；执行冻结敏感性分析，不换网络 |
| proper risk 改善、regret 不改善 | 决策边界目标错配；分析 near-tie 后才允许 ranking/distributional objective |
| raw/intermediate RGB 有效、冻结 DINO 无效 | 表示瓶颈；允许预注册 decision-conditioned temporal adapter |
| 特权信息也无法预测 | estimand 不可观测或噪声过大；收缩目标或停止路线 |
| 离线有效、sequential 失败 | history distribution shift；不得声称行为收益 |

STOP 不是自动开启新架构搜索的许可。

### 13.1 决策收益的守恒分解

对固定合法panel，以native为基准定义oracle机会、模型获取收益和选择regret：

\[
G_{oracle}=\mathbb E[U^*-U_0],\qquad
G_{model}=\mathbb E[U_{\hat i}-U_0],\qquad
R_{model}=\mathbb E[U^*-U_{\hat i}].
\]

则严格有`G_oracle=G_model+R_model`。这一恒等式把三种现象分开：真正获取正机会、退回native避免
有害干预、以及仍未恢复的oracle regret。任何视觉决策主张必须以`G_model>0`及其组级不确定性为核心，
不能把相对有害parent的regret下降冒充视觉收益。

R005E3A提供了必要反例：100%选择native可显著优于有害parent，却有`G_model=0`且对所有视觉干预
完全不敏感。因此native fallback既是保守部署机制，也是必须显式排除的学习退化解。

### 13.2 条件期望决策、surrogate 与有限网络必须分层

当前G1的规范对象是

\[
m_i(X)=\mathbb E[U_i-U_0\mid X],
\qquad i^*(X)\in\arg\max_i m_i(X).
\]

因此必须严格区分：

\[
G_{hindsight}>0
\not\Rightarrow G_{Bayes}(X)>0
\not\Rightarrow \exists f\in\mathcal F:G_f>0
\not\Rightarrow \operatorname{Opt}(f,\theta_0)>0
\not\Rightarrow G_{heldout}>0.
\]

R005E3C只读诊断显示，当前soft-regret在初始化处对94.54%的正收益候选给出raise方向，但正/非正机会
的聚合参数梯度近乎反向，且训练终点正收益score-gradient均值衰减约`8.42e7`倍。这支持softmax低概率
尾部纠错衰减与梯度冲突的局部描述，不构成单一根因或函数类不可实现性证明。

utility-gap pairwise logistic虽然有逐样本错排梯度与hard-regret上界，但一般多候选随机效用下不保证
恢复条件期望utility排序；故不得作为Bayes一致性桥。当前最小资格处理是identity输出下的
native-relative proper mean regression：在平方可积、正且只依赖输入的权重、无限函数类和总体风险
最优化条件下，其解为`E[U_i-U_0|X]`。对任意预测score，还有逐输入界

\[
m_{i^*}-m_{\hat i}\le2\|s-m\|_\infty\le2\|s-m\|_2.
\]

但有限网络、AdamW、有限样本、候选支持变化与外观移位均不在该定理保证内。R005E3D的机械审计通过，
但训练域`G_model=-0.00085196`、absolute capture为负，正式FAIL。proper目标恢复了22次non-native选择和
部分correct视觉relation，却被4次精确接近`-1/8`的two-frame temporal-gate错误压倒；这说明总体proper
不等于有限单头网络会利用已知效用结构。

### 13.3 已知动力学与未知场景结果必须结构化分工

登记目标可写为

\[
U_i=\frac78F_i+\frac18H_iC_i,
\]

其中`F_i=H_iB_i`是filter后one-frame score，`H_i`为scene hard gate，`C_i=c(τ_i,h)`为
candidate-history two-frame comfort。如果`C_i`能从推理时因果可用的候选和实际历史确定性计算，则

\[
\mathbb E[U_i\mid X]
=\frac78\mathbb E[F_i\mid X]
+\frac18C_i\mathbb P(H_i=1\mid X).
\]

因此最小结构假设不是更深网络，而是两个proper heads分别预测连续`F`与二元`H`，再由固定代数与解析
`C`重构score。它只有在解析`C`不读取future/evaluator label、能exact复算且时延合格时才合法；否则
停止。该factorization是G1学习地基而非论文主创新，必须与monolithic matched scorer做单因素对照。

## 14. 模块准入五问

任何新公式、模块、loss 或训练阶段必须回答：

1. 它解决哪个已观察、可复现的失败？
2. 它由什么数学假设或决策要求推出？
3. 为什么更简单、容量匹配的方法不足？
4. 哪个独立消融能够否定它？
5. 删除它会造成什么预先可预测的变化？

无法回答则不进入最终方法。

## 15. 贡献边界

### 15.1 唯一主贡献候选

`appearance-interventional differential decision evidence learning`：在完整时空 native–challenger
signed difference 提出的 query 上读取全局视觉与历史证据，学习外观干预下稳定的 contextual
relation。

它必须在相同数据、候选、backbone、参数、FLOP、token、optimizer 与 seed 预算下，稳定优于
D-JEPA-style native-relative set relation、World4Scorer-style candidate state、DA-WAM/GTRS-style
scorer、普通 full-context set scorer 与 independent scorer，才具备网络创新资格。

### 15.2 从属机制候选

`identifiability-aware conservative arbitration`：当同时下界不足时保留 native。该机制只有在
coverage/sharpness non-vacuous、错误干预下降且不是全拒判时，才形成有效从属贡献。

### 15.3 明确非创新项

DINO、cross-attention、trajectory query、signed integral、Set Transformer、relative scoring、
uncertainty、conformal、native fallback、Adapter/LoRA 本身均不是 AORTA 的原创贡献。

## 16. 当前可以和不可以说什么

当前可以说：

- source-fit 冻结合法候选域上的 `U_ctx` 存在可复算 oracle opportunity；
- R005E2 的目标、输入、对照、统计单位和失败解释具有闭合预注册；
- Differential Query 有由 common-min、late-fork、ROI/support 失败驱动的结构动机。
- 当前soft-regret塌缩伴随可复算的概率尾部score-gradient衰减；proper mean是具有总体条件均值依据的
  最小资格对照，而不是论文创新。
- proper mean单头已被R005E3D训练域负收益否定；精确component分解把主要失败定位为离散temporal gate
  漏读，并提出尚未验证的structured utility factorization。

当前不可以说：

- 正确视觉已经能恢复 `U_ctx`；
- Differential Query 已优于通用 scorer；
- AORTA 已经具备 NavDream 外观鲁棒性；
- 离线 margin 改善带来了 planner 行为或安全收益；
- calibration 给出任意 OOD 保证；
- 当前系统已经达到 ICCV 方法成功标准。

## 17. 文档优先级

发生冲突时使用以下顺序：

1. 已冻结的具体实验 preregistration 与 provenance；
2. 本文件中的当前理论对象、方法边界与门顺序；
3. `AORTA_RESEARCH_CONSTITUTION_V1_CN.md` 与 `CLAIM_EVIDENCE_MATRIX.md`；
4. `RESEARCH_BRIEF.md` 的时间序列证据；
5. 历史 `METHOD_SPEC v0.4`、旧 field/geometry 文档与 refine logs。

本文件可以因新的已审计结果更新，但不得事后改写已经冻结实验的目标、控制、阈值或裁决。
