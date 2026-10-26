# AORTA Visual-First Theory V2：从稳定视觉到可恢复的规划敏感修复

状态：候选理论体系；尚未获得方法有效性或投稿资格。
日期：2026-10-26

## 0. 研究锚点

Constant Eye 已经给出一个重要事实：冻结 DINOv3-H+ 并为不同 planner 训练适配器与规划头，可以显著改善外观 OOD。AORTA 不重复把 DINO、adapter 或 NavDream 当作创新，而研究它未回答的问题：

> 对一个已经训练好的冻结驾驶策略，外观变化究竟破坏了哪些会影响决策的内部信息；这些信息在何种条件下能由外部稳定视觉证据增量恢复，并在受限低秩接口中修复？

非目标：恢复完整 RGB、强制所有 feature 不变、重新训练 planner、把接近 clean trajectory 等同于安全、用更多模块替代机制证据。

## 1. 第一性对象

设物理状态为 \(S\)，外观为 \(A\)，历史为 \(H\)，图像为

\[
X=g(S,A),
\]

冻结 planner 在层 \(\ell\) 分解为

\[
z^\ell=\phi_{\leq\ell}(X,H),\qquad
o=f_{>\ell}(z^\ell,H),
\]

其中 \(o\) 是硬选择前的轨迹、候选分数或分布参数。任务评价记为 \(U(S,o,H)\)。

### 1.1 外观效应的行为介导充分性，而非一般充分统计量

一般“存在充分统计量”是空泛的，因为总可以取 \(T(z)=f(z,H)\)。本研究只要求更强、可否定的对象：存在**低维、跨场景共享、测试时可恢复**的表示，能够解释给定外观族诱发的大部分行为差异。

若存在低维 \(T_\ell(z^\ell)\) 与 \(q_\ell\) 使

\[
f_{>\ell}(z^\ell,H)=q_\ell(T_\ell(z^\ell),H)
\]

在研究域内近似成立，并且 \(T_\ell\) 能在 group-held-out 场景中由测试时证据恢复，才称为候选介导表示。统计版本

\[
I(O;X\mid T_\ell,H)=0.
\]

只提供背景约束，不能独立构成贡献。稳定但常数的表示满足外观不变，却不满足任务需要；因此 feature drift 不是主要优化对象。

### 1.2 决策等价类

对固定历史定义

\[
z\sim_{\ell,H}z'
\iff
f_{>\ell}(z,H)=f_{>\ell}(z',H).
\]

AORTA 需要恢复的是等价类中影响行为的坐标，而不是 clean feature 的全部自由度。

## 2. planner-sensitive 几何

当 tail network 在 \(z_c\) 邻域可微，令

\[
J_\ell(z_c,H)=\frac{\partial f_{>\ell}}{\partial z}\bigg|_{z_c,H}.
\]

用任务输出度量 \(W\succeq0\) 定义 pullback metric

\[
G_\ell(z_c,H)=J_\ell^\top WJ_\ell.
\]

外观漂移 \(\delta z=z_a-z_c\) 的一阶任务影响为

\[
D_{task}^2(\delta z)=\delta z^\top G_\ell\delta z.
\]

若 Hessian 算子范数在邻域内不超过 \(M\)，则

\[
\|f(z_c+\delta z)-f(z_c)-J_\ell\delta z\|
\leq \frac{M}{2}\|\delta z\|^2.
\]

因此：

- \(\ker J_\ell\) 中的大幅变化在一阶上可与行为无关；
- row-space 中的小幅变化可能显著影响行为；
- Jacobian 只给局部敏感性，不给正确性或因果充分性。

### 2.1 由 task-weighted appearance drift 直接定义低秩基

对第 \(i\) 个 source clean/shift pair，定义

\[
\delta_i=z_{c,i}-z_{a,i},\qquad P_B=BB^\top,\qquad B^\top B=I_r.
\]

固定秩 \(r\) 的共享基直接由

\[
\boxed{
B_r^*=\arg\min_{B^\top B=I_r}
\mathbb E_i\left[\|J_i(I-P_B)\delta_i\|_W^2\right]
}
\]

定义。它最小化“投影后仍未覆盖的 planner-sensitive 外观效应”，同时排除只敏感但未受外观扰动、或受扰动但位于任务 null space 的方向。\(W\) 是结果可见前冻结的输出度量：回归 planner 主分析使用物理单位归一化的轨迹位置度量，yaw/heading 为独立次指标；不得用 official outcome 反向选择 \(W\)。实现使用 JVP/VJP 或随机低秩算子，不显式构造完整 Jacobian。

对空间 feature \(Z\in\mathbb R^{P\times C}\)，主架构不学习混合任意空间位置的巨大基，而使用保持 token/BEV 拓扑的通道子空间

\[
\mathcal P_B=I_P\otimes BB^\top,\qquad B\in\mathbb R^{C\times r}.
\]

reader 预测空间系数场 \(A\in\mathbb R^{P\times r}\)，残差为 \(AB^\top\)。这样 rank 描述通道机制复杂度，而空间位置仍由原 planner 接口和注册投影保持。

### 2.2 冻结求解器

\(B_r^*\) 不是普通 PCA/SVD。实现采用确定性的 Stiefel 流形梯度下降：

1. discovery split 的 channel-drift PCA 仅作初始化；另用两个固定随机正交初始化，共三次启动；
2. 欧氏梯度由 matrix-free JVP/VJP 计算，Riemannian 梯度为
   \[
   \nabla_R=\nabla_E-B\operatorname{sym}(B^\top\nabla_E);
   \]
3. 以 QR retraction 保持 \(B^\top B=I\)；
4. 使用固定 Armijo backtracking：初始步长1、收缩0.5、充分下降常数 \(10^{-4}\)；
5. 相对目标下降小于 \(10^{-6}\) 持续20步或达到500步停止；
6. 只按 discovery objective 选择三次启动中的最低值，禁止看 selection/outer 指标。

在正式数据前，必须在小型显式 Jacobian 上比较 dense objective、autograd gradient、有限差分和 matrix-free 解；objective/gradient 超出预注册容差即停止。

每个样本的 held-out basis coverage 为

\[
\kappa_i(B)=1-
\frac{\|J_i(I-P_B)\delta_i\|_W^2}
{\|J_i\delta_i\|_W^2+\epsilon}.
\]

必须报告 rank–coverage 曲线、组分布、最差组以及转弯/路口/高曲率分层。若共享低秩基只覆盖常见场景而遗漏关键组，则停止全局 CPTR；不得事后增大 rank。

## 3. 因果介导资格

对同一物理场景的 clean/shift pair，在层 \(\ell\) 和子空间 \(P\) 进行 interchange intervention：

\[
z_{patch}^{\ell}=(I-P)z_a^{\ell}+Pz_c^{\ell}.
\]

定义行为恢复量

\[
\Delta D_{\ell,P}
=D(o_a,o_c)-D(o_{patch},o_c).
\]

主证据不是单个 restoration ratio，而是同一 anchor 上 \(\Delta D\) 的配对组级分布、任务效用变化及对照差。

必要对照：

1. same-scene clean patch；
2. wrong-scene clean patch；
3. shifted-self identity patch；
4. 等维、等能量 random subspace；
5. energy-matched noise；
6. reverse clean-to-shift patch。

full clean patch 只证明 hook 与 tail 的机械上界，在确定性分解下本身接近构造性结果。科学证据必须来自低秩 patch 在严格 group-held-out 数据上的恢复、双向干预、剂量—响应，以及相对这些对照的差异。它仍只支持模型内部因果介导，不等价于真实世界语义或安全因果。

## 4. 增量可恢复性边界

令外部稳定证据为 \(D=\psi_{DINO}(X)\)。目标不是无条件恢复 \(z_c\)，而是识别任务统计量 \(T_\ell(z_c)\)。

### 命题 1：观测碰撞下的不可能性

若存在两个合法场景 \(s_1,s_2\)，使

\[
(z_a(s_1),D_a(s_1),H(s_1))=(z_a(s_2),D_a(s_2),H(s_2)),
\]

但

\[
T_\ell(z_c(s_1))\neq T_\ell(z_c(s_2)),
\]

则不存在确定性修复器 \(R(z_a,D_a,H)\) 能在两个场景上同时正确恢复目标统计量。

这是 AORTA 的 fail-closed 边界：DINO 稳定不意味着它保留了 planner 所需的全部信息。若 correct/mismatch/removed 或场景碰撞诊断不支持目标的增量可恢复性，应增加合法时序/几何证据或拒判，而不是强制回归。

### 命题 2：条件行为保持

假设：

- A1：\(f_{>\ell}(z,H)=q(T_\ell(z),H)+e_F\)，且 \(\|e_F\|\leq\varepsilon_F\)；
- A2：\(q\) 对第一输入是 \(L\)-Lipschitz；
- A3：外部证据给出 \(\widehat T_a\)，满足 \(\|\widehat T_a-T_\ell(z_c)\|\leq\varepsilon_D\)；
- A4：修复表示 \(\widehat z_a\) 满足 \(\|T_\ell(\widehat z_a)-\widehat T_a\|\leq\varepsilon_R\)。

则

\[
\|f_{>\ell}(\widehat z_a,H)-f_{>\ell}(z_c,H)\|
\leq L(\varepsilon_D+\varepsilon_R)+2\varepsilon_F.
\]

该界只控制模型行为相对 clean 的偏差；clean 本身错误时不保证效用改善或安全。

### 命题 3：方法特有的误差分解

对空间接口令

\[
\widetilde B=I_P\otimes B_r^*,\qquad
\widetilde P_B=\widetilde B\widetilde B^\top,
\]

且 \(\Delta Z_i=Z_{c,i}-Z_{a,i}\)。真实的逐位置系数场及其向量化形式为

\[
A_i^*=\Delta Z_iB_r^*,\qquad a_i^*=\operatorname{vec}(A_i^*).
\]

修复器预测 \(\widehat A_i\)，令 \(\widehat a_i=\operatorname{vec}(\widehat A_i)\) 以及

\[
\widehat z_i=z_{a,i}+\widetilde B\widehat a_i.
\]

则相对 clean 的向量化表示误差为

\[
e_i=\widehat z_i-z_{c,i}
=\widetilde B(\widehat a_i-a_i^*)-(I-\widetilde P_B)\delta_i.
\]

定义 \(\|v\|_W=\sqrt{v^\top Wv}\)；当 \(W\succeq0\) 奇异时它是半范数。若局部二阶余项在该 \(W\)-输出半范数下满足 \(\|r_i\|_W\le M_i\|e_i\|^2/2\)，则

\[
\boxed{
\|f(\widehat z_i)-f(z_{c,i})\|_W
\leq
\|J_i\widetilde B(\widehat a_i-a_i^*)\|_W
+\|J_i(I-\widetilde P_B)\delta_i\|_W
+\frac{M_i}{2}\|e_i\|^2
}
\]

三个可测项分别对应 reader 误差、低秩基未覆盖误差和非线性余项。这才是 CPTR 的直接理论来源。

对候选打分 planner，若 clean top-1 margin 为 \(\gamma\)，且修复后

\[
\|\widehat s-s_c\|_\infty<\gamma/2,
\]

则 clean winner 保持。该 margin 结论仍不适用于 NAVSIM 离散门控的全局效用；output restoration、selection preservation、official utility 与 safety 必须分开报告。

## 5. 由机制导出的候选算法：Causal Planner-Tangent Repair（CPTR）

CPTR 是 **rank-constrained repair**，不声称求得满足行为约束的全局最小范数解。它只有在 R006V0 因果介导门通过后才获得实现资格。

### 5.1 系统图

```text
shifted RGB -----------------> frozen native encoder -> z_a^ell ----+----> frozen planner tail -> output
       \                                                           |
        +-> frozen DINOv3-H+ -> registered spatial evidence -------+-> per-location low-rank coefficients
                                                                    |
              source-derived, intervention-qualified basis B_l <---+
                                                                    |
                                     Z_hat_a = Z_a + A_theta B_r^T
```

### 5.2 冻结与可训练边界

冻结：原 planner 全部参数、DINOv3-H+、候选生成器、评估器、通过 nested group-held-out R006V0 资格化的接口层、rank 和基 \(B_r^*\)。
新增：一个共享 evidence-to-spatial-coefficient reader \(A_\theta\)。DINO 到 native interface 的空间绑定必须使用冻结相机/BEV投影并通过错配与坐标哨兵；若某层没有合法空间绑定，该层不得进入 R006V1。最多再允许一个 uncertainty/abstention head，且仅在增量可恢复性失败证据出现后加入。

### 5.3 为什么不是普通 adapter

普通 adapter 可任意改写表示，可能获得参数量收益但无法说明改了什么。CPTR 的输出被限制在 \(B_r^*\)，其定义直接最小化未覆盖的 planner-sensitive appearance effect；训练目标经过冻结 planner tail 检验，不把全特征 L2 当成最终目标。

### 5.4 训练目标

source-only clean/controlled-appearance pair 上：

\[
\widehat Z_a=Z_a+A_\theta(D_a,Z_a,H)B_r^{*\top}.
\]

主损失为冻结 planner 的 functional restoration：

\[
\mathcal L_{func}=D_o(f_{>\ell}(\widehat z_a,H),f_{>\ell}(z_c,H)).
\]

从属约束为

\[
\mathcal L=\mathcal L_{func}
+\lambda_T\|\widehat A-(Z_c-Z_a)B_r^*\|_F^2
+\lambda_E\|\widehat A\|_F^2
+\lambda_{id}\|A_\theta(D_c,Z_c,H)\|_F^2.
\]

- \(\mathcal L_{func}\)：修复 planner 使用的函数，不追求任意 feature 相似；
- 投影项：只对资格化 task subspace 提供 source supervision；
- residual-energy 项：在资格化子空间内限制不必要的干预，但不声称得到全局最小范数解；
- clean identity：直接约束 no-harm。

权重必须在 source selection split 预冻结；不可根据 NavDream 调整。

## 6. 三道资格门

### P1：低秩行为介导

R006V0 只做 activation intervention，不训练 CPTR。full patch 仅为机械上界；真正门槛是 nested group-held-out 的低秩 \(B_r^*\) patch 能否覆盖足够 planner-sensitive appearance effect，并在双向、剂量—响应和 wrong-scene/random/noise 对照下保持 scene-specific。失败则停止全局共享 CPTR。

### P2：DINO 增量可恢复性

在冻结 P1 目标上做 source group-held-out probe。令

\[
\mathcal R_D(\mathcal F)=\inf_{r\in\mathcal F}\mathbb E\|a^*-r(z_a,D_a,H)\|^2,
\quad
\mathcal R_0(\mathcal F)=\inf_{r\in\mathcal F}\mathbb E\|a^*-r(z_a,0,H)\|^2.
\]

其中两臂使用参数量、输入维度、训练预算、优化器与选择协议匹配的函数类 \(\mathcal F\)，removed arm 以零证据保持同一网络。有限实验只能支持“DINO 在指定函数类与训练合同下提供组外增量可预测信息”，不能证明 Bayes risk 顺序或严格可识别性。correct DINO 必须优于条件匹配 mismatch、removed、native-only；报告目标系数误差与 functional restoration。失败不允许直接扩大网络。

### P3：rank-constrained 修复的行为闭环

只有 P2 通过才训练 CPTR，并比较 frozen base、Constant Eye、容量匹配普通 adapter、全特征回归、random-subspace repair。最终检验 clean no-harm、NavDream、三类 planner、batch-1 P50/P95 和 official utility。

## 7. 论文主张边界

可争取的唯一主贡献：

> 一套对冻结 planner 的外观失效进行因果定位，并只在具有组外增量可恢复证据的 planner-sensitive tangent subspace 中实施 rank-constrained repair 的方法。

从属贡献：用可证伪的介导性—可恢复性—行为保持链，把“稳定视觉”与“稳定决策”之间的缺口形式化。

明确不宣称原创：DINOv3、Jacobian、SVD、activation patching、adapter、Lipschitz 界、信息瓶颈、source augmentation 或 NavDream。

## 8. Best-Paper 资格

只有同时满足以下条件，才允许称为 Best-Paper 候选：

1. 发现性：多个 planner 中存在可重复的低秩、规划敏感外观漂移机制；
2. 必要性：因果 patch 优于 wrong-scene/random/energy controls；
3. 增量可恢复性：DINO 对该机制目标有不可替代的 held-out 信息；
4. 方法优势：CPTR 在相同数据、backbone、参数/FLOPs 下优于 Constant Eye 与普通 adapter；
5. 行为闭环：NavDream、clean、跨 planner、时延和 official utility 同时闭合；
6. 诚实边界：不把恢复 clean 行为写成安全保证，不把 source 干预写成任意 OOD 因果定理。

任一核心门失败，降低主张或停止该分支；不得以新增模块、阈值搜索或删样替代裁决。
