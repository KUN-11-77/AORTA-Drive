# AORTA Visual-First Theory V2：从稳定视觉到可恢复的规划敏感修复

> **Superseded on 2026-10-27.** R006V0/R006V0F 后 fixed-global-rank/CPTR 主线已停止；当前 canonical
> 视觉理论为 `docs/AORTA_DINO_INTERFACE_FIRST_PRINCIPLES_V1_CN.md`。本文保留为历史假设与数学账本，
> 不再授权Fantope、DINO reader、CPTR或VJP训练。

状态：`FIXED_GLOBAL_RANK_FALSIFIED__FAILURE_DECOMPOSITION_RUNNING__CPTR_HOLD`；候选理论体系，尚未获得方法有效性或投稿资格。
日期：2026-10-26

## 0. 研究锚点

Constant Eye 已经给出一个重要事实：冻结 DINOv3-H+ 并为不同 planner 训练适配器与规划头，可以显著改善外观 OOD。AORTA 不重复把 DINO、adapter 或 NavDream 当作创新，而研究它未回答的问题：

> 对一个已经训练好的冻结驾驶策略，外观变化究竟破坏了哪些会影响决策的内部信息；这些信息在何种条件下能由外部稳定视觉证据增量恢复，并通过由机制证据资格化的受约束接口修复？

非目标：恢复完整 RGB、强制所有 feature 不变、重新训练 planner、把接近 clean trajectory 等同于安全、用更多模块替代机制证据。

### 0.1 一分钟论文故事

现有视觉鲁棒驾驶通常试图让整幅表示在天气变化下保持稳定；Constant Eye 进一步表明，冻结 DINO
并为各 planner 重训接口可以改善外观 OOD。但 planner 并不需要全部视觉信息，而且“整体稳定”不保证
“决策稳定”。我们的核心待检验假设是：外观扰动可能主要通过 planner Jacobian 所定义的任务敏感方向
影响行为，而大量 feature drift 可能位于决策等价类内。第一轮 nested 实验已否定“固定阈值下存在跨折
一致的全局最小rank”这一强接口，故当前不再预设共享固定维数，也未获得 CPTR 训练资格。AORTA 先用
结果前冻结的 repeated-initialization 诊断区分估计器不稳定、折间条件复杂度和方向不共享；只有得到稳定
且可干预的 planner-tangent 对象，再证明冻结 DINO 能组外恢复其坐标，才允许训练受约束修复器。若任何
一门失败，方法主张停止或收缩，而不是事后采用rank 32。

一句话：**不是让视觉特征处处不变，而是先验证外观扰动是否存在可稳定估计、可组外干预、可由DINO
恢复的 planner-sensitive 介导结构；固定全局低秩只是已被否定的一种候选接口，而不是预设答案。**

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

其中 \(o\) 是硬选择前的轨迹、候选分数或分布参数。任务评价记为 \(U(S,o,H)\)。受控外观配对只在
\((S,H)\) 与除 RGB 外的 planner 输入完全一致时定义；否则 \(z_c-z_a\) 不能解释为外观效应。

### 1.1 外观效应的行为介导充分性，而非一般充分统计量

一般“存在充分统计量”是空泛的，因为总可以取 \(T(z)=f(z,H)\)。本研究不把“重建任意输入下的完整
planner 输出”作为必要条件，而要求一个更窄、可否定的对象：存在**低通道秩、跨场景共享、测试时可恢复**
的表示，能够在典型组中解释给定外观族诱发的过半可 patch 行为差异。

下面的完整因子分解是一个足够但非必要的强条件。若存在低维 \(T_\ell(z^\ell)\) 与 \(q_\ell\) 使

\[
f_{>\ell}(z^\ell,H)=q_\ell(T_\ell(z^\ell),H)
\]

在研究域内近似成立，并且 \(T_\ell\) 能在 group-held-out 场景中由测试时证据恢复，则它当然是候选介导
表示；但 AORTA 的主实验只检验更弱的**配对外观效应介导**：低秩干预能否恢复 full same-scene patch
所定义的有限行为差异。统计版本

\[
I(O;X\mid T_\ell,H)=0.
\]

只提供背景约束，不能独立构成贡献，也不是 R006V0 的经验 estimand。稳定但常数的表示满足外观不变，
却不满足任务需要；因此 feature drift 不是主要优化对象。

### 1.2 决策等价类

对固定历史定义

\[
z\sim_{\ell,H}z'
\iff
f_{>\ell}(z,H)=f_{>\ell}(z',H).
\]

AORTA 需要恢复的是等价类中影响行为的坐标，而不是 clean feature 的全部自由度。

## 2. planner-sensitive 几何

当 tail network 在激活 \(z\) 邻域可微，令

\[
J_\ell(z,H)=\frac{\partial f_{>\ell}}{\partial z}\bigg|_{z,H}.
\]

用任务输出度量 \(W\succeq0\) 定义位置相关的 pullback metric

\[
G_\ell(z,H)=J_\ell(z,H)^\top WJ_\ell(z,H).
\]

运行时待修复漂移 \(\delta=z_c-z_a\) 在 shifted 起点的一阶任务影响为

\[
D_{task,a}^2(\delta)=\delta^\top G_\ell(z_a,H)\delta.
\]

记 \(\|y\|_W=\sqrt{y^\top Wy}\)。当 \(W\) 奇异时这是半范数。若在所研究邻域内

\[
\|J_\ell(z_1,H)-J_\ell(z_2,H)\|_{2\to W}
\leq L_J\|z_1-z_2\|_2,
\]

其中 \(\|A\|_{2\to W}=\sup_{\|v\|_2=1}\|Av\|_W\)，则对局部增量 \(v\)

\[
\|f(z+v)-f(z)-J_\ell(z,H)v\|_W
\leq \frac{L_J}{2}\|v\|_2^2.
\]

因此：

- \(\ker J_\ell\) 中的大幅变化在一阶上可与行为无关；
- row-space 中的小幅变化可能显著影响行为；
- Jacobian 只给局部敏感性，不给正确性或因果充分性。

### 2.1 由 task-weighted appearance drift 直接定义低秩基

对第 \(i\) 个 source clean/shift pair，令

\[
\Delta Z_i=Z_{c,i}-Z_{a,i}\in\mathbb R^{P\times C}.
\]

固定行优先向量化 \(\operatorname{rvec}\)，使 \(\delta_i=\operatorname{rvec}(\Delta Z_i)\)。令
\(S_\ell\in\{0,1\}^{P\times P_{repair}}\) 为保序 site selector，满足
\(S_\ell^\top S_\ell=I\) 与 \(M_\ell=S_\ell S_\ell^\top\)。对共享通道基
\(B\in\mathbb R^{C\times r}\)、\(B^\top B=I_r\)，定义

\[
\mathcal B_{S,B}=S_\ell\otimes B,
\qquad
\Pi_{M,B}=\mathcal B_{S,B}\mathcal B_{S,B}^\top
=M_\ell\otimes BB^\top.
\]

于是对任意 site coefficient field \(A\in\mathbb R^{P_{repair}\times r}\)，

\[
\operatorname{rvec}(S_\ell AB^\top)=\mathcal B_{S,B}\operatorname{rvec}(A),
\]

且 \(\Pi_{M,B}\delta_i=\operatorname{rvec}(M_\ell\Delta Z_iBB^\top)\)。固定通道秩 \(r\) 的共享基由

\[
\boxed{
B_r^*=\arg\min_{B^\top B=I_r}
\mathbb E_i\left[
\|J_{a,i}(I-\Pi_{M,B})\delta_i\|_W^2
\right],
\qquad
J_{a,i}=J_\ell(z_{a,i},H_i)
}
\]

定义。它最小化“投影后仍未覆盖的 planner-sensitive 外观效应”，同时排除只敏感但未受外观扰动、
或受扰动但位于任务 null space 的方向。\(W\) 是结果可见前冻结的输出度量：回归 planner 主分析使用
物理单位归一化的轨迹位置度量，yaw/heading 为独立次指标；不得用 official outcome 反向选择 \(W\)。
实现使用 JVP/VJP，不显式构造完整 Jacobian。

这里必须在 shifted 点求导，因为运行时修复从 \(z_a\) 出发。clean 点 Jacobian 可以作为稳定性诊断，但不得
替代正式目标；代码、缓存、求解器和论文公式统一使用 \(J_a\)。

主架构不学习混合任意 sites 的巨大基，而使用保持接口拓扑的共享通道子空间

\[
\mathcal P_B^\ell[\Delta Z]=M_\ell\Delta ZBB^\top,
\qquad B\in\mathbb R^{C\times r}.
\]

reader 预测 site 系数 \(A\in\mathbb R^{P_{repair}\times r}\)，嵌入接口后的残差为
\(S_\ell AB^\top\)。因此 \(r\) 是**共享通道秩**，不是整个干预只有 \(r\) 个自由度；系数场仍有
\(P_{repair}r\) 个标量自由度，而修复矩阵的通道秩至多为 \(r\)。site 语义由真实 planner 接口保持。
对LTF C3，64个8×8 BEV memory sites可修复，第65个
non-image status site严格冻结；对C4，\(P_{repair}=1\)且其语义是trajectory query，不得称为空间位置。
具体地，\(M_{C3}=\operatorname{diag}(\mathbf1_{64},0)\)，\(M_{C4}=[1]\)。因此

\[
\mathcal P_B^{C3}[\Delta Z]
=\begin{bmatrix}\Delta Z_{1:64}BB^\top\\0_{1\times C}\end{bmatrix},
\qquad
\mathcal P_B^{C4}[\Delta Z]=\Delta ZBB^\top.
\]

矩阵形式的残差目标与上面的 \((I-\Pi_{M,B})\delta_i\) 完全相同，从而 tensor 顺序、site mask 与
status 不可修复约束完全显式。

该目标只依赖投影子空间：对任意 \(Q\in O(r)\)，\(B\) 与 \(BQ\) 给出同一
\(\Pi_{M,B}\)。因此可识别对象是 \(\operatorname{span}(B)\) 或投影矩阵，而不是单根 basis vector 的符号、
顺序或语义。跨重启/跨折稳定性必须用 projector distance 或 principal angles 比较；未来 reader 的系数监督
只能在部署时冻结的确定性 basis gauge 内解释，不能把某一列系数赋予跨 gauge 的自然语义。

最后，tail 输出维度有限只说明**每个样本**的 Jacobian 行空间秩有上界；它不推出不同场景、不同 sites
存在同一个低秩通道子空间。后者是 R006V0 明确要否证或支持的经验假设，而不是线性代数恒等式。

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
\frac{\|J_{a,i}(I-\Pi_{M,B})\delta_i\|_W^2}
{\|J_{a,i}\delta_i\|_W^2+\epsilon}.
\]

必须报告 rank–coverage 曲线、组分布、最差组以及转弯/路口/高曲率分层。若共享低秩基只覆盖常见场景而遗漏关键组，则停止全局 CPTR；不得事后增大 rank。
有限 source groups 上通过该门只支持登记采样总体中的经验共享性；group bootstrap 的外推单位是 recording
group，不构成对任意道路、天气或 planner 的分布无关泛化定理。

#### 2.2.1 R006V0 后的 rank 理论修正

R006V0 在机械审计通过后得到每折最小合格rank `[8,32,4,32,4,32]`，未达到冻结的4/6共识门，故
`STOP_GLOBAL_SHARED_INTERFACE`不可被rank 32的事后6/6通过改写。更重要的是，task coverage并不具有
普通PCA的自动单调性。对嵌套projector $P_{r+1}=P_r+Q$，令

\[
a=J(I-P_r)\delta,\qquad b=JQ\delta,
\]

则

\[
\|a-b\|_W^2-\|a\|_W^2
=\|b\|_W^2-2\langle a,b\rangle_W.
\]

因此只有当 $2\langle a,b\rangle_W\ge\|b\|_W^2$ 时，新增rank才改善该样本的task residual；planner
输出空间中的抵消项会使更大rank反而恶化coverage。该恒等式和反例已由
`tests/test_r006_rank_geometry.py`固化。由此：

1. 最小合格rank只是给定fold、estimator和threshold下的操作性量，不是内在维数；
2. ordered flag只能消除独立rank子空间不可比性，不能自动保证coverage单调；
3. R006V0F已在4/6 folds观察到同fold/rank的seed资格翻转，按冻结规则裁决
   `RANK_ESTIMATION_UNSTABLE`；它只授权一次凸estimator/group-refit诊断，不能解锁adaptive-rank、
   DINO reader或CPTR；
4. 若跨折方向差异未显著超过seed内分散度，global-basis路线应停止，而不是继续扩大rank。

### 2.3 从局部 Jacobian 到有限外观干预

若从 shifted 到 clean 的线段上 tail 可微，则有精确路径积分

\[
f(z_c)-f(z_a)=\int_0^1J_\ell(z_a+t\delta,H)\,\delta\,dt,
\qquad \delta=z_c-z_a.
\]

因此低秩 Jacobian coverage 只描述路径起点处的一阶可覆盖性，不等于有限位移已经被修复。令
\(a^*=\mathcal B_{S,B}^\top\delta\)、\(q=\mathcal B_{S,B}\widehat a\) 为目标投影系数与实际修复量。
若包含 \(z_a\)、\(z_a+q\) 与 \(z_c\) 的星形路径邻域内，
Jacobian 满足 2 节定义的 \(L_J\)-Lipschitz 条件，则

\[
\boxed{
\begin{aligned}
\|f(z_a+q)-f(z_c)\|_W
\leq{}&\|J_a\mathcal B_{S,B}(\widehat a-a^*)\|_W\\
&+\|J_a(I-\Pi_{M,B})\delta\|_W\\
&+\frac{L_J}{2}\bigl(\|q\|_2^2+\|\delta\|_2^2\bigr).
\end{aligned}}
\]

证明：令
\(r(v)=f(z_a+v)-f(z_a)-J_av\)。由 Jacobian 的诱导范数 Lipschitz 条件与路径积分，
\(\|r(v)\|_W\le L_J\|v\|_2^2/2\)。又

\[
f(z_a+q)-f(z_c)=J_a(q-\delta)+r(q)-r(\delta),
\]

并且
\(q-\delta=\mathcal B_{S,B}(\widehat a-a^*)-(I-\Pi_{M,B})\delta\)。对三项使用三角不等式即得。

前两项分别对应 reader 误差与低秩基未覆盖误差，第三项是从局部线性算子走到有限干预的非线性账本。
所以 R006V0 必须同时报告 held-out Jacobian coverage 与真实冻结 tail 上的 patch restoration；任一单独成立
都不足以通过机制门。实际 Stiefel 求解只能得到登记初始化和优化合同下的数值近似解，不声称全局最优。

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

为排除“极小但统计显著”，对每组定义full-patch归一化有限恢复比例

\[
\rho_g=
\frac{\sum_{i\in g}\left[D(o_{a,i},o_{c,i})-D(o_{patch,i},o_{c,i})\right]}
{\sum_{i\in g}\left[D(o_{a,i},o_{c,i})-D(o_{full,i},o_{c,i})\right]+10^{-12}}.
\]

只有42组 \(\rho_g\) 中位数的group-bootstrap 95% CI下界超过0.5，才允许主张“典型组中解释大部分
可patch行为漂移”并解锁CPTR；否则即使存在稳定正效应，也只记录为弱介导并停止主方法分支。

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

证明：分别对 \(\widehat z_a\) 与 \(z_c\) 使用 A1，并以三角不等式分离两个 residual；再由 A2 得到
\(L\|T_\ell(\widehat z_a)-T_\ell(z_c)\|\)，最后插入 \(\widehat T_a\) 并使用 A3、A4。

该界只控制模型行为相对 clean 的偏差；clean 本身错误时不保证效用改善或安全。

### 命题 3：方法特有的误差分解

沿用 2.1 节的 row-vectorization 与 site selector，令

\[
\mathcal B^*=S_\ell\otimes B_r^*,\qquad
\Pi^*=M_\ell\otimes B_r^*B_r^{*\top}.
\]

且 \(\Delta Z_i=Z_{c,i}-Z_{a,i}\)。真实的逐位置系数场及其向量化形式为

\[
A_i^*=S_\ell^\top\Delta Z_iB_r^*,\qquad
a_i^*=\operatorname{rvec}(A_i^*).
\]

修复器预测 \(\widehat A_i\)，令 \(\widehat a_i=\operatorname{rvec}(\widehat A_i)\)、
\(\delta_i=\operatorname{rvec}(\Delta Z_i)\) 以及

\[
\widehat z_i=z_{a,i}+\mathcal B^*\widehat a_i.
\]

则相对 clean 的向量化表示误差为

\[
e_i=\widehat z_i-z_{c,i}
=\mathcal B^*(\widehat a_i-a_i^*)-(I-\Pi^*)\delta_i.
\]

该等式直接来自
\(\delta_i=\Pi^*\delta_i+(I-\Pi^*)\delta_i\) 与
\(\Pi^*\delta_i=\mathcal B^*a_i^*\)，因此在固定 vectorization 与 selector 下是精确恒等式，而非近似。

行为误差使用2.3节以 shifted 点 \(J_{a,i}\) 为起点的有限干预界：reader误差、低秩基未覆盖误差，以及
\(L_{J,i}(\|q_i\|_2^2+\|\delta_i\|_2^2)/2\) 的非线性余项。这三个可测对象才是 CPTR 的直接理论来源；
不得把起点局部 coverage 写成有限扰动的全局保证。

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
        +-> frozen DINOv3-H+ -> site-conditioned evidence reader ---+-> per-site low-rank coefficients
                                                                    |
              source-derived, intervention-qualified basis B_l <---+
                                                                    |
                                     Z_hat_a = Z_a + S_l A_theta B_r^T
```

### 5.2 冻结与可训练边界

冻结：原 planner 全部参数、DINOv3-H+、候选生成器、评估器、通过 nested group-held-out R006V0 资格化的接口层、rank 和基 \(B_r^*\)。
新增：一个共享 evidence-to-site-coefficient reader \(A_\theta\)。它以native interface sites为queries、
以冻结DINO patch tokens为keys/values做cross-attention，再输出每个repairable site的rank-r系数。C3输出
64×r并为status site补严格零；C4输出1×r的query-conditioned系数。该设计不假设DINO patch与BEV或
trajectory query一一对应，而把“正确视觉证据能否读取”留给R006V1检验。最多再允许一个
uncertainty/abstention head，且仅在增量可恢复性失败证据出现后加入。

### 5.3 为什么不是普通 adapter

普通 adapter 可任意改写表示，可能获得参数量收益但无法说明改了什么。CPTR 的输出被限制在 \(B_r^*\)，其定义直接最小化未覆盖的 planner-sensitive appearance effect；训练目标经过冻结 planner tail 检验，不把全特征 L2 当成最终目标。

### 5.4 训练目标

source-only clean/controlled-appearance pair 上：

\[
\widehat Z_a=Z_a+S_\ell A_\theta(D_a,Z_a,H)B_r^{*\top}.
\]

主损失为冻结 planner 的 functional restoration：

\[
\mathcal L_{func}=D_o(f_{>\ell}(\widehat z_a,H),f_{>\ell}(z_c,H)).
\]

从属约束为

\[
\mathcal L=\mathcal L_{func}
+\lambda_T\|\widehat A-S_\ell^\top(Z_c-Z_a)B_r^*\|_F^2
+\lambda_E\|\widehat A\|_F^2
+\lambda_{id}\|A_\theta(D_c,Z_c,H)\|_F^2.
\]

- \(\mathcal L_{func}\)：修复 planner 使用的函数，不追求任意 feature 相似；
- 投影项：只对资格化 task subspace 提供 source supervision；
- residual-energy 项：在资格化子空间内限制不必要的干预，但不声称得到全局最小范数解；
- clean identity：直接约束 no-harm。

系数损失只在冻结的 deterministic basis gauge 内定义；方法与行为主张依赖的是
\(S_\ell\widehat A B_r^{*\top}\) 和 frozen-tail 输出，而不是某一列 basis coefficient 的独立语义。

权重必须在 source selection split 预冻结；不可根据 NavDream 调整。

## 6. 三道资格门

### P1：低秩行为介导

R006V0 只做 activation intervention，不训练 CPTR。full patch 仅为机械上界；真正门槛是 nested group-held-out 的低秩 \(B_r^*\) patch 能否覆盖足够 planner-sensitive appearance effect，并在双向、剂量—响应和 wrong-scene/random/noise 对照下保持 scene-specific。失败则停止全局共享 CPTR。

### P2：DINO 增量可恢复性

在冻结 P1 目标上做 source group-held-out probe。C3/C4都使用上述同一site-conditioned reader family，
仅site数量与status mask不同。令

\[
\mathcal R_D(\mathcal F)=\inf_{r\in\mathcal F}\mathbb E\|a^*-r(z_a,D_a,H)\|^2,
\quad
\mathcal R_0(\mathcal F)=\inf_{r\in\mathcal F}\mathbb E\|a^*-r(z_a,0,H)\|^2.
\]

其中两臂使用参数量、输入维度、训练预算、优化器与选择协议匹配的函数类 \(\mathcal F\)，removed arm 以零证据保持同一网络。有限实验只能支持“DINO 在指定函数类与训练合同下提供组外增量可预测信息”，不能证明 Bayes risk 顺序或严格可识别性。correct DINO 必须优于条件匹配 mismatch、removed、native-only；报告目标系数误差与 functional restoration。失败不允许直接扩大网络。

### P2.1 科学计算图与部署计算图

机制发现需要math-SDPA与关闭MHA fastpath以获得资格化JVP；最终推理优先使用官方快速图。二者只有在
完整确认集上轨迹最大绝对差不超过`1e-5`且离散选择完全一致时才可互换。否则CPTR必须在科学图上部署，
并把DINO、native encoder、reader和tail的batch-1 P50/P95全部计入时延。

### P3：rank-constrained 修复的行为闭环

只有 P2 通过才训练 CPTR，并比较 frozen base、Constant Eye、容量匹配普通 adapter、全特征回归、random-subspace repair。最终检验 clean no-harm、NavDream、三类 planner、batch-1 P50/P95 和 official utility。

## 7. 论文主张边界

可争取的唯一主贡献：

> 一套对冻结 planner 的外观失效进行因果定位，并只在具有组外增量可恢复证据的 planner-sensitive tangent subspace 中实施 rank-constrained repair 的方法。

从属贡献：用可证伪的介导性—可恢复性—行为保持链，把“稳定视觉”与“稳定决策”之间的缺口形式化。

明确不宣称原创：DINOv3、Jacobian、SVD、activation patching、adapter、Lipschitz 界、信息瓶颈、source augmentation 或 NavDream。

## 8. Best-Paper 资格

只有同时满足以下条件，才允许称为 Best-Paper 候选：

1. 发现性：多个 planner 中存在可重复的任务敏感外观失效规律，或其异质性可由因果可用变量前瞻预测；
2. 必要性：因果 patch 优于 wrong-scene/random/energy controls；
3. 增量可恢复性：DINO 对该机制目标有不可替代的 held-out 信息；
4. 方法优势：由机制导出的结构在相同数据、backbone、参数/FLOPs 下优于 Constant Eye 与普通 adapter；
5. 行为闭环：NavDream、clean、跨 planner、时延和 official utility 同时闭合；
6. 诚实边界：不把恢复 clean 行为写成安全保证，不把 source 干预写成任意 OOD 因果定理。

任一核心门失败，降低主张或停止该分支；不得以新增模块、阈值搜索或删样替代裁决。
低秩、固定 layer、固定 rank 与 CPTR 均是可淘汰的结构假设，不是第一性目标。详细假设预算与停止规则见
`docs/AORTA_HYPOTHESIS_TASTE_GATE.md`。

截至R006V0F，Best-Paper故事仍未获得方法级正证据：当前最扎实的贡献是排除了“固定全局最小rank”与
“单次best-of-restarts足以资格化basis”两个接口。只有凸诊断能证明稳定task metric、有限干预显示
scene-specific介导、DINO提供组外增量信息，并且matched架构与行为闭环同时通过，才可恢复方法主张。
