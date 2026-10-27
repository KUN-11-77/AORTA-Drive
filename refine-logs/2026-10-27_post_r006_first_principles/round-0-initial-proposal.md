# AORTA post-R006：Best-Paper 级第一性理论重构（Round 0）

状态：`INITIAL_REFINEMENT_PROPOSAL__NOT_METHOD_EVIDENCE__NO_EXPERIMENT_AUTHORIZATION`

## 1. Problem Anchor

### 一句话研究问题

在物理场景与驾驶历史不变、只有视觉外观改变时，如何为**完全冻结的驾驶规划器**恢复真正影响其行为的视觉信息，同时避免修改与当前决策无关的内部自由度？

### 必须解决的瓶颈

Constant Eye 及学长的既有实验已经支持冻结 DINOv3-H+ 是有效的外观稳定视觉底座。本项目不再重复验证 DINO 是否有用，而直接回答：在统一冻结 DINOv3-H+ 的条件下，稳定视觉证据应通过什么任务接口进入冻结 planner，才能以最小干预恢复真实行为。R006V0 已否定跨折一致的固定最小 rank；R006V0F 又表明当前 Stiefel basis 在 4/6 folds 发生 seed 资格翻转。因此“共享固定低秩子空间”停止，不再占用正式实验预算。

### 不可破坏的约束

- planner、候选生成器与 DINOv3-H+ 冻结；所有方法臂共享同一 DINOv3-H+ 权重、预处理、token 与缓存合同；
- source-only 设计，NavDream/test/target 继续封存；
- 推理变量必须因果可用；
- 最多两个新增可训练组件；
- matched backbone、数据、参数/FLOPs、优化步数与 batch-1 时延；
- 任何修复都必须单独报告 clean no-harm、行为恢复、official utility 与安全边界；
- 低秩、固定 layer、固定 rank、Jacobian、cross-attention 都不是预设贡献。

### 成功标准

1. 机制：正确视觉证据能在未见 recording groups 上预测/恢复 planner-sensitive appearance effect；
2. 必要性：由该机制导出的接口在相同预算下稳定优于 Constant Eye、普通 adapter、直接输出修复和无任务约束 reader；
3. 行为：NavDream、多 planner、clean no-harm 与 batch-1 P95 同时闭合；
4. 简洁：一分钟故事只需要“发现—原则—一个核心算子—行为结果”。

## 2. 失败证据迫使出的理论修正

R006 的负结果不否定“planner 只对部分表示变化敏感”，但否定了把这些敏感方向压成一个跨场景固定 projector 的强接口。第一性对象应从

\[
\text{global shared subspace } P
\]

改为输入条件化的局部任务算子

\[
J_i=\frac{\partial f_{>\ell}(z,H_i)}{\partial z}\bigg|_{z=z_{a,i}}.
\]

不同样本的 row space 可以旋转、维数可以变化；但对给定样本，任何一阶有效的最小能量修复都不需要落在 \(\ker J_i\) 中。这是无需共享低秩假设的结构事实。

## 3. 先诊断 DINO-to-planner 接口，再选择方法

冻结 DINOv3-H+ 已有正向证据，但这不等于当前 planner 正确使用了其信息。下一步首先执行一个短、可裁决的接口因果审计，而不是预设低秩或 VJP 必然正确。对同一物理场景的 clean/shift pair，在统一冻结 DINO 与 planner 下比较：

1. clean DINO tokens；
2. shifted DINO tokens；
3. geometry-matched wrong-scene DINO tokens；
4. removed/constant DINO tokens；
5. clean/shift 的局部 token interchange 与 dose-response；
6. DINO feature drift、adapter 输出 drift、planner 输出 drift 与 task-gradient-weighted drift。

它必须区分三个竞争解释：

- **H-A 信息不足：**clean DINO token 也不能恢复任务输出；此时任何 DINO-only interface 都不应继续堆网络，应增加合法时序/多视角信息或收缩主张；
- **H-B 接口放大：**DINO drift 小，但 adapter/planner 输出 drift 大，且 clean token interchange 可恢复；应研究 task-conditioned contractive interface；
- **H-C 任务信息被压缩：**DINO 含有可恢复任务信息，但通用 adapter 丢失，correct 显著优于 mismatch/removed；应研究 decision-conditioned evidence routing；
- **H-D 无场景依赖：**wrong-scene 与 correct 效果接近；当前增益主要来自先验，停止“视觉机制”主张。

下面的 sample-conditioned tangent repair 仅是 H-B 被支持后的首选候选，不是预设答案。

### Route A：Sample-Conditioned Tangent Repair（H-B 条件候选）

### Route A：Sample-Conditioned Tangent Repair（推荐）

对 source clean/appearance pair 定义 planner 输出差

\[
d_i=o_{c,i}-o_{a,i},\qquad o=f_{>\ell}(z,H).
\]

冻结输出度量 \(W\succeq0\)，定义局部 ridge repair teacher

\[
q_i^*(\lambda)
=\arg\min_q
\frac12\|J_iq-d_i\|_W^2+\frac\lambda2\|q\|_2^2.
\]

其唯一解为

\[
q_i^*=J_i^\top W^{1/2}
\left(W^{1/2}J_iJ_i^\top W^{1/2}+\lambda I\right)^{-1}
W^{1/2}d_i,
\]

所以

\[
\boxed{q_i^*\in\operatorname{range}(J_i^\top)}.
\]

这是一条局部 representer 结论：最小能量的一阶修复必定位于**当前样本**的 planner-sensitive cotangent span，而不需要跨场景共享 basis。

部署时，证据 reader 从 shifted DINO tokens、native activation 与 causal history 预测低维输出侧控制 \(\hat\alpha_i\)，通过一次 VJP 形成

\[
\hat q_i=J_i^\top W^{1/2}\hat\alpha_i,
\qquad
\hat z_i=z_{a,i}+\eta_i\hat q_i.
\]

门控 \(\eta_i\in[0,1]\) 由 source calibration 冻结；若视觉证据不足或 trust-region 预测无效，则 \(\eta_i=0\)。核心 trainable component 只有 evidence-to-control reader；可选的第二组件是共享的不确定性/拒判头。

#### 为什么它由失败自然导出

- fixed global basis 不稳定：改用 sample-conditioned row space；
- 全特征 adapter 可任意改写：限制到当前 planner 的一阶有效方向；
- 直接 imitation 可能过度修复：使用 minimum-energy / trust-region 目标；
- DINO 稳定不等于任务充分：correct/mismatch/removed 检验 \(\alpha\) 的组外可恢复性；
- 局部线性并非全局保证：用 nonlinear tail restoration 与 remainder ledger 验证。

#### 可证伪预测

在相同输入、参数量与训练预算下，如果 planner-sensitive restriction 有意义，则 Proposed 应在 unseen groups 上以更小的干预能量达到更高的 frozen-tail restoration，并比 ordinary adapter 产生更低 clean harm；若只得到训练拟合或被 direct-output residual 匹配，则该机制没有独立价值。

### Route B：Soft Fantope Task Metric（停止，不进入下一轮正式实验）

对已开封 inner artifacts 拟合

\[
P^*=\arg\min_{P\in\mathcal F_r}
\sum_i\|J_i(I-P)\delta_i\|_{W_i}^2,
\quad
\mathcal F_r=\{0\preceq P\preceq I,\operatorname{tr}P=r\}.
\]

它只能区分“Stiefel 优化失败”“稳定软 metric”“共享结构本身不稳定”，不能直接验证 DINO-to-planner 方法，也不能形成 AORTA 的主创新。考虑到后续实验必须围绕 DINOv3-H+，该路线从正式执行序列删除；Fantope 仅保留为可选离线数学附录，不阻塞、不解锁、也不延迟主实验。fixed-projector/CPTR 家族按现有负结果停止。

## 4. Route A 的理论闭环与边界

### 4.1 局部最优性

将任意 \(q=q_{\parallel}+q_{\perp}\)，其中 \(q_{\perp}\in\ker J_i\)、\(q_{\parallel}\perp\ker J_i\)。则数据项只依赖 \(q_{\parallel}\)，而 \(\|q\|^2=\|q_{\parallel}\|^2+\|q_{\perp}\|^2\)。当 \(\lambda>0\) 时最优解必有 \(q_{\perp}=0\)。因此方法不是“猜一个低秩空间”，而是删除对当前局部输出无效的自由度。

### 4.2 有限干预误差账本

若 Jacobian 在 trust region 内满足

\[
\|J(z_1)-J(z_2)\|_{2\to W}\le L_J\|z_1-z_2\|_2,
\]

则

\[
\|f(z_a+\hat q)-o_c\|_W
\le
\|J_i\hat q-d_i\|_W
+\frac{L_J}{2}\|\hat q\|_2^2.
\]

第一项由 evidence reader 与局部 inverse 控制，第二项决定 trust-region/拒判；该界只保证相对 clean planner output 的恢复，不保证 clean 正确、official utility 或真实安全。

### 4.3 可恢复性边界

令部署证据 \(E_i=(D_i,z_{a,i},H_i)\)。若存在两个合法场景 \(s_1,s_2\) 满足 \(E(s_1)=E(s_2)\)，但所需输出侧控制 \(\alpha^*(s_1)\ne\alpha^*(s_2)\)，则任何确定性 reader 都不可能同时恢复二者。正确策略是 posterior widening / abstention，而不是增加网络容量。

### 4.4 当前不主张的内容

- 不主张局部 tangent 等于全局因果子空间；
- 不主张恢复 clean output 等于提高驾驶安全；
- 不主张 DINO 必然包含 \(\alpha^*\)；
- 不主张一次 VJP 必然满足实时性；
- 不主张 Jacobian、minimum-norm inverse 或 activation repair 本身原创。

## 5. 三个最小实验块：全部围绕统一冻结 DINOv3-H+

### E0：DINO-to-planner 接口因果审计

不训练新方法。对 source group-held-out 的同场景 clean/shift pair 做 full/local token interchange、wrong-scene、removed 与 dose-response；同时复算 DINO drift、adapter drift、planner-output drift 和 task-gradient-weighted drift。按 H-A/H-B/H-C/H-D 给出唯一裁决，禁止同一结果同时为多个互斥故事背书。

Kill rule：H-A 或 H-D 成立时停止当前 DINO-only 方法主张；H-B/H-C 都没有明确支持时，不启动新网络搜索。

### E1：由 E0 机制导出的 DINO 方法与 matched 对照

从第一批训练开始就以相同冻结 DINOv3-H+ 为视觉输入。在 source group-held-out 上比较：

1. Proposed：DINO evidence-to-control reader + sample-conditioned VJP repair；
2. DINO + 容量/FLOPs 匹配普通 residual adapter；
3. DINO + direct-output residual；
4. Constant Eye 式 DINO adapter/head；
5. DINO + fixed global basis（已失败路线，仅作历史对照）；
6. correct / geometry-matched mismatch / removed DINO；
7. full-clean patch、wrong-scene、random/energy-matched repair 作为机制上下界与负对照。

主指标：单位干预能量的 group-held-out nonlinear-tail restoration、组级恢复比例、correct-minus-mismatch、clean identity、trust-region violation、batch-1 端到端 P50/P95（包括 DINO 与 VJP）。

若 E0 支持 H-B，Proposed 为 DINO evidence-to-control + sample-conditioned VJP repair；若支持 H-C，Proposed 改为 decision-conditioned evidence routing。二者不得同时堆叠。Kill rule：Proposed 不能同时优于 mismatch/removed、matched DINO adapter 与 direct-output residual，就停止对应方法；不通过增加模块挽救。

### E2：核心算法必要性与论文行为闭环

E1 通过后，在完整 source split 上冻结架构和超参数，比较 Proposed、Constant Eye、matched DINO adapter、DINO direct-output residual，以及不使用任务接口的同容量 reader。所有方法共享完全相同的 DINOv3-H+ tokens、数据、增强与训练预算；随后一次性执行 NavDream zero-shot × 至少三类 planner。

主指标：group-held-out functional restoration、official source utility、NavDream hard-failure net reduction、clean harm、最差组、组级置信区间、参数/FLOPs/P50/P95。

Kill rule：普通 DINO adapter/direct residual 匹配 Proposed，则删除架构创新；无跨外观行为净收益、clean harm 超门或时延超预算，停止 Best-Paper/方法主张。

## 6. 一分钟故事候选

> Constant Eye 说明稳定视觉底座能帮助外观鲁棒驾驶，但没有说明稳定信息应怎样进入一个完全冻结的 planner。我们发现，把 planner 的外观失效压进跨场景固定低秩空间并不稳定。回到第一性原理，给定一个样本，任何最小能量的一阶有效修复都位于该样本 planner Jacobian 的 cotangent span。我们因此让冻结 DINO 只预测输出侧控制，再通过 planner 的 VJP 把它变成输入条件化、任务有效的修复；证据不足或非线性超出 trust region 时保持原模型。该方法必须在 matched adapter、Constant Eye、错配视觉和多 planner NavDream 上同时证明必要性与行为收益，否则我们删除这一主张。

## 7. 当前选择

DINOv3-H+ 是冻结共同底座，不再作为研究变量。正式顺序是：先做一次 DINO-to-planner 接口因果审计；若支持 H-B，再推进 Route A；若支持 H-C，则改为 decision-conditioned evidence routing；若支持 H-A/H-D，则停止当前 DINO-only 方法主张。Route B 与 fixed-projector/CPTR 正式停止，不再用 Fantope、升 rank、换 seed 或调阈值拖延主实验。
