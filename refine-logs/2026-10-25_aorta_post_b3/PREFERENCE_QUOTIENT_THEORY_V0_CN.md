# AORTA 候选偏好商空间理论 v0

状态：`FORMAL_CANDIDATE_THEORY__B4_INVALID__B4R1_PREREGISTERED_NOT_RUN`

本文只建立可核查的数学地基，并说明它如何约束网络与 loss。它不证明跨外观泛化、OOD coverage、
真实驾驶安全或相对已有工作的原创性。B4 未通过时，不授权基于本文训练网络。

查新边界见 `references/2026-10-25_preference_quotient_novelty_boundary.md`：Hodge decomposition、
graph least squares 与谱隙依赖均有成熟先例，不能单独作为 AORTA 的原创性来源。

## 1. 合法估计对象

固定物理场景 \(S\)、planner \(\pi\)、native \(\tau_0\) 和六个注册 challenger。令预冻结
task-intent membership 为 \(e_i(S,\pi)\in\{0,1\}\)，合法候选域为

\[
V_{S,\pi}=\{0\}\cup\{i:e_i(S,\pi)=1\},\qquad K_{S,\pi}=|V_{S,\pi}|.
\]

当前 evaluator 对每个合法候选独立使用
\([\tau_0,\tau_i]\) 面板评分，因此定义

\[
y_i=U(S;\tau_0,\tau_i),\qquad i\in V_{S,\pi}.
\]

网络输出 native-conditioned 节点分数

\[
s_i=s_\theta(X,\tau_0,\tau_i),\qquad i\in V_{S,\pi},
\]

而学习对象是 pair preference

\[
\Delta_{ij}=s_j-s_i,qquad \Delta_{ij}^{*}=y_j-y_i.
\]

这里的节点是候选轨迹，不是道路网格节点。由于 evaluator 存在面板归一化，\(y_i\) 不是脱离
native context 的绝对效用；换 planner 或 native 时，估计对象也随之改变。

**候选域一致性公理。** 标签、score vector、incidence matrix、Laplacian、loss、oracle 与 regret
必须共享同一个 \(V_{S,\pi}\)。masked candidate 不得以任意常数、负无穷近似或伪标签进入图。
B4 违反这一公理，故其正向 oracle diagnostics 无效；B4R1 只允许使用 B4 前已冻结的 H7BS1
membership 修复。

## 2. 候选图、gauge 与可识别性

对每个 context 令候选比较图为无向加权图 \(G_{S,\pi}=(V_{S,\pi},E,w)\)，任取边方向并定义
incidence matrix \(D\in\mathbb R^{|E|\times K_{S,\pi}}\)：对边 \(e=(i\to j)\)，该行在 \(i\)
处为 \(-1\)，在 \(j\)
处为 \(+1\)。于是全部边偏好为

\[
p=Ds,qquad p^{*}=Dy.
\]

令 \(W=\operatorname{diag}(w_e)\)，候选图 Laplacian 为

\[
L_w=D^\top W D.
\]

### 命题 1：商空间可识别性

若 \(K_{S,\pi}\ge2\) 且正权边子图连通，则

\[
\ker D=\ker L_w=\operatorname{span}\{\mathbf 1\}.
\]

因此映射 \([s]\mapsto Ds\) 在商空间
\(\mathbb R^K/\operatorname{span}\{\mathbf 1\}\) 上是一一的；候选偏好唯一确定节点分数的共同
offset 等价类，但不能确定绝对零点。

当 \(K_{S,\pi}=1\) 时，商空间维数为 0，没有 pair supervision、cycle 或 \(\lambda_2\)。该
context 仍必须保留在 candidate-supply 与 oracle-gain 统计中并贡献零增益，但不应伪造图边。

**证明。** \(D\mathbf 1=0\)。若 \(Dv=0\)，每条正权边两端的值相等；图连通推出所有节点值相等，
故 \(v=c\mathbf 1\)。又
\(v^\top L_wv=\|W^{1/2}Dv\|_2^2\)，所以 \(L_wv=0\) 与 \(Dv=0\) 具有相同零空间。证毕。

**架构含义。** 网络无需、也不应被迫学习跨场景共享的绝对 score 零点。训练与评估应使用
\(Ds\) 或等价的 mean-zero representative，而不是 absolute-score MAE。

## 3. 一个由商空间推出的核心 loss

令节点误差 \(e=s-y\)。定义 weighted quotient loss

\[
\boxed{
\mathcal L_Q(s,y;G)
=\|W^{1/2}D(s-y)\|_2^2
=(s-y)^\top L_w(s-y).
}
\]

它同时具有三项精确性质：

1. **共同 gauge 不变：**对任意 \(c,d\in\mathbb R\)，
   \(\mathcal L_Q(s+c\mathbf1,y+d\mathbf1;G)=\mathcal L_Q(s,y;G)\)；
2. **零损失可识别：**若正权图连通，\(\mathcal L_Q=0\) 当且仅当
   \(s-y=c\mathbf1\)；
3. **pairwise 等价：**它就是所有注册边上的加权 preference squared error，不是附加的
   graph regularizer。

完全图、统一边权时还有恒等式

\[
\sum_{i<j}\big[(s_j-s_i)-(y_j-y_i)\big]^2
=K\left\|P_\perp(s-y)\right\|_2^2,
\quad
P_\perp=I-\frac1K\mathbf1\mathbf1^\top.
\]

因此可以用 \(O(K)\) 的 centered-node 实现获得与 \(O(K^2)\) 全 pair squared loss 完全相同的目标。
若使用非统一 task-critical 权重，则必须保持正权比较图连通，否则会产生额外不可识别分量。

## 4. loss 到 decision regret 的确定性桥梁

令 \(\lambda_2(L_w)>0\) 为候选图的谱隙，\(i^*=\arg\max_i y_i\)，
\(\hat i=\arg\max_i s_i\)。

### 命题 2：单场景谱 regret 界

若某一固定 context 上 \(K_{S,\pi}\ge2\) 且

\[
\mathcal L_Q(s,y;G)\le \eta^2,
\]

则

\[
\boxed{
y_{i^*}-y_{\hat i}
\le \frac{2\eta}{\sqrt{\lambda_2(L_w)}}.
}
\]

**证明。** 令 \(\bar e=K^{-1}\mathbf1^\top e\)，\(e_\perp=e-\bar e\mathbf1\)。由 Laplacian
谱界，

\[
\lambda_2(L_w)\|e_\perp\|_2^2
\le e^\top L_we=\mathcal L_Q\le\eta^2.
\]

于是 \(\|e_\perp\|_\infty\le\eta/\sqrt{\lambda_2(L_w)}\)。共同项 \(\bar e\) 不影响排序，故
节点分数在某个共同 gauge 下具有一致误差上界
\(\varepsilon=\eta/\sqrt{\lambda_2}\)。再由
\(s_{\hat i}\ge s_{i^*}\)，

\[
y_{i^*}-y_{\hat i}
\le |e_{\perp,i^*}|+|e_{\perp,\hat i}|
\le2\varepsilon.
\]

证毕。

**边界。** 这是单 context、确定性的 algebraic bound。训练集平均 loss 小不能自动推出每个未见
场景的 \(\eta\) 小；经验风险到 population/OOD 的推广仍需独立统计假设和实验。谱隙也说明，任意
删除 pair 或极端重加权会削弱理论控制，不能只按当前模型误差挑“容易的边”。

### 推论：top-margin 稳定性

若真实最优候选与次优候选的效用间隔

\[
m^*=y_{i^*}-\max_{j\ne i^*}y_j
>\frac{2\eta}{\sqrt{\lambda_2(L_w)}},
\]

则 \(\hat i=i^*\)。等价地，小平均 MAE 不是排序保证；必须联合报告 quotient error、谱结构与
真实 top margin。

## 5. exact preference architecture 的结构含义

任意自由反对称 edge prediction \(q\in\mathbb R^{|E|}\) 可正交分解为

\[
q=q_{grad}+q_{cycle},
\]

其中

\[
q_{grad}=D L^\dagger D^\top q,qquad
D^\top q_{cycle}=0
\]

（未加权形式；加权内积下使用对应的加权投影）。共享节点 scorer 直接输出 \(q=Ds\)，因而
\(q_{cycle}=0\)。这给出：

- 反对称性；
- 任意闭合环路 preference 和为零；
- 从 \(O(K^2)\) 自由 pair 输出降为 \(O(K)\) 节点 score；
- 与标量 evaluator 标签的结构一致。

但这仍不是经验优势。如果真实任务偏好存在实质非传递性，或 direct-pair head 的 cyclic residual
不会伤害 regret/泛化，那么 exact-gradient 反而可能是不合适的归纳偏置。matched-structure 实验必须
直接检验这一点。

### 命题 2.5：native-star 空泛性

若部署与监督只包含 native edges \(E_\star=\{(0,i):i>0\}\)，则对任意自由 edge vector
\(q\in\mathbb R^{K-1}\)，取 \(s_0=0,s_i=q_i\) 即有 \(D_\star s=q\)。所以

\[
\operatorname{Im}(D_\star)=\mathbb R^{K-1},
\]

且 star 没有 cycle。此时 node-score difference 与直接输出 \(K-1\) 个 native-relative deltas
只是重参数化，exactness 不限制假设空间。

**方法边界。** Hodge/coboundary 只有在 complete candidate graph 的额外边具有独立统计价值，且
matched experiments 显示其改善 held-out regret/泛化时才可能成为结构贡献。当前
challenger–challenger labels 由 native-conditioned scalar 代数派生，故 cycle consistency 首先是
标签恒等式，不能在结果前被写成创新。完整反驳见
`STRUCTURAL_NOVELTY_PREMORTEM_V1_CN.md`。

## 6. 外观商目标与坍缩反例

对保持 \((S,\tau_0,\mathcal T,y)\) 不变的合法外观干预 \(T_a\)，定义

\[
\mathcal L_{app,Q}
=\left\|W^{1/2}D
\left[s_\theta(T_aX)-\operatorname{sg}(s_\theta(X))\right]
\right\|_2^2.
\]

这只要求 preference 在外观轨道上稳定，允许共同 scene score 偏移。

### 命题 3：不变性不足

常数 scorer \(s_\theta(X,\tau_0,\tau_i)=c(X)\) 对所有候选给出相同值，因而对任意外观变换都有
\(Ds=0\) 和 \(\mathcal L_{app,Q}=0\)，却不能表示任何非零真实偏好。

因此必须同时具有合法的 \(\mathcal L_Q(s,y;G)\) 或等价 decision supervision；跨外观漂移小不能
单独证明任务信息存在。correct-image、mismatch、removed-image 与 prior controls 仍是不可省略的
视觉必要性反证。

## 7. 选择性 native 仲裁

对每个 challenger 定义

\[
\Delta_{i0}^*=y_i-y_0,qquad
\widehat\Delta_{i0}=s_i-s_0.
\]

假设存在同时误差事件

\[
\mathcal E=\left\{
|\widehat\Delta_{i0}-\Delta_{i0}^*|\le r_i,
\ \forall i=1,\ldots,K-1
\right\}.
\]

令 \(L_i=\widehat\Delta_{i0}-r_i\)，选择 \(\hat i=\arg\max_iL_i\)，且只在
\(L_{\hat i}>0\) 时干预。

### 命题 4：条件性 native improvement

在事件 \(\mathcal E\) 上，任何被该规则接受的 challenger 都满足
\(y_{\hat i}>y_0\)。若独立 calibration 分布上
\(\Pr(\mathcal E)\ge1-\alpha\)，则同分布条件下错误宣称改善的概率至多为 \(\alpha\)。

**证明。** 在 \(\mathcal E\) 上，
\(\Delta_{i0}^*\ge\widehat\Delta_{i0}-r_i=L_i\)。接受意味着
\(L_{\hat i}>0\)，故 \(\Delta_{\hat i0}^*>0\)。概率结论来自错误改善事件包含于
\(\mathcal E^c\)。证毕。

**边界。** coverage 必须对全部候选同时成立；逐候选 marginal intervals 不够。source exchangeability
不能无条件外推到 NavDream、另一 planner 或新 candidate policy。网络输出 uncertainty 也不等于
上述 coverage，必须单独验证 sharpness、false improvement 与 shift 后失覆盖。

## 8. 网络实现合同

最小候选网络为

\[
Z=E_{vision}(X),\quad
a_i=E_{traj}(\tau_i),\quad
h_i=H_\theta(Z,a_0,a_i),\quad
s_i=W_sh_i.
\]

为了使理论和实现同构，必须满足：

1. \(E_{vision}\) 对面板只执行一次；
2. \(E_{traj}\) 编码完整时域 pose、yaw、time 与相对 native 几何；
3. \(H_\theta\) 对所有 challenger 参数共享；
4. 非 native 候选置乱时，score 同步置乱；
5. pair preference 只能由 score difference 产生，主模型不另设自由 edge head；
6. 不确定性/半径与 point score 分开审计，不能用同一训练误差声称 coverage；
7. 推理复杂度目标为
   \(O(C_{vision}+K(C_{traj}+C_{query}))\)，并实测 batch-1 P50/P95/P99。
8. 可变大小面板必须以 ragged/mask-aware batching 实现；mask 在 score、softmax/argmax、edge
   construction、loss、uncertainty 与 arbitration 全路径一致，并通过 permutation/mask metamorphic
   tests。

## 9. 与实验一一对应的反证门

| 理论/设计 | 最小验证 | 失败后的裁决 |
|---|---|---|
| 合法 \(y_i\) 与非平凡 preference | B4R1 合法域双 materialization + oracle ceiling | 停止当前 panel/estimand |
| quotient loss 有任务信息 | correct image vs mismatch/removed/prior | 停止视觉必要性主张 |
| exact-gradient 归纳偏置有必要 | native-vector vs star-score vs complete-score vs free complete-edge matched study | 若star-score≈native-vector且complete structure无held-out收益，删除Hodge/结构贡献 |
| 完整时域有必要 | full 4 s vs prefix/fixed field | 若无差异，简化架构 |
| 谱误差关联 decision | per-group \(\mathcal L_Q\)、top margin、regret | 不把 loss 写成 decision bridge |
| selective rule 非真空且可靠 | simultaneous coverage + sharpness + harm | 只保留条件定理 |
| 跨外观/跨 planner | NavDream + 多 planner 冻结验证 | 收窄适用范围 |
| 实时性 | 同硬件 batch-1 profiler | 不宣称在线部署 |

## 10. 当前允许和禁止的主张

允许：

- 上述四个命题作为带明确假设的数学事实；
- weighted quotient loss 是与候选比较图一致的 gauge-invariant 目标；
- 图谱隙给出 per-context loss 到 regret 的确定性桥梁。

禁止：

- 声称该 loss 或 architecture 已有经验优势或文献原创性；
- 从训练平均 loss 推出 NavDream regret 保证；
- 把 conditional native improvement 写成真实驾驶安全保证；
- 在 B4R1 通过前训练或按 B4/B4R1结果调整图权重；
- 用公式复杂度掩盖 matched baseline 可能等效的事实。

