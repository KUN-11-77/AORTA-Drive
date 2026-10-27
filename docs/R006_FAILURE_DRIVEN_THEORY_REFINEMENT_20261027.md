# R006 失败证据驱动的理论修正

状态：`FIXED_MINIMUM_RANK_HYPOTHESIS_FALSIFIED__NEXT_HYPOTHESES_NOT_YET_CONFIRMED`。

## 1. 原命题为何过强

原 R006V0 将两个不同命题绑定在一个 gate 中：

1. 存在跨场景稳定的任务敏感方向或子空间；
2. 用固定 coverage 阈值定义的**最小充分维数**在不同训练折中也相同。

记第 (f) 折、rank (r) 的估计投影为 (P_{f,r}=B_{f,r}B_{f,r}^{\top})，inner-selection
coverage 为 (C_f(r))，则登记选择为

\[
r_f^*=\min\{r\in\{4,8,16,32\}:C_f(r)\geq c_0,\ C_f^{\min}(r)\geq c_{\min}\}.
\]

(r_f^*) 同时受真实谱尾、有限组采样、basis 估计误差、非凸求解和固定阈值影响。即使存在稳定的
低维核心方向，也不推出 (r_f^*) 必须在至少4/6折相同。因此本轮 STOP 精确否定的是这个**复合且
更强**的接口假设，而不是所有低维结构。

## 2. 实验迫使理论增加的层次

新的理论必须把“方向、复杂度和可干预性”分开：

### H1：共享层

同一因果切口在不同折稳定出现。本轮 C4 为6/6、C3为0/6，对 H1 提供 inner evidence，但仍不是
outer intervention 证据。

### H2：共享方向核心

用旋转不变距离

\[
d_{\mathrm{proj}}^2(P_f,P_g)=r-\|B_f^{\top}B_g\|_F^2
\]

描述子空间，而不是比较不可识别的 basis 坐标。Post-hoc 结果显示 rank4/8 的归一化 projector overlap
均值分别为0.9785/0.9827，使 H2 值得继续检验；但 discovery 集重叠，尚不能视为独立确认。

### H3：异质有效维数

不同场景分布可能共享核心方向，却具有不同的谱尾质量或任务敏感能量，因此达到固定 coverage 所需的
维数可以变化。该命题解释 `{4,8,32}` 的最小秩分散，但尚未排除优化与估计误差。

### H4：有限干预介导

即使 H1–H3 成立，也必须用 same-scene、wrong-scene、random-subspace、noise、reverse 与 dose
intervention 证明方向具有 scene-specific causal efficacy。basis coverage 不是介导结论。

## 3. 下一架构不能怎样产生

以下做法均被禁止：

- 看到 rank32 在6/6折合格后，直接把它追认为原协议 PASS；
- 降低4/6共识门或只报告最有利 folds；
- 直接训练 DINO reader/CPTR，让下游结果替上游假设兜底；
- 把更多模块、更复杂 loss 或更多数学术语当作修复。

## 4. 决定下一理论路线的最小诊断

下一阶段只使用已开封 inner artifacts，依次回答：

1. **估计/优化诊断**：独立 rank 求解为何出现 rank8 合格3折、rank16仅1折？复算多初始化稳定性、
   discovery objective 与 selection coverage，判断非凸局部解和 selection 方差；
2. **共享方向诊断**：在不依赖 basis 旋转的 projector/principal-angle 指标上检验 fold stability；
3. **谱尾诊断**：用单个有序、嵌套的最大 rank flag 评估 prefix coverage，避免每个 rank 独立拟合造成
   的不可比性；
4. **异质性诊断**：检查所需维数是否随 group/task-drift能量系统变化，而非随机波动。

只有这些诊断能区分：

- `A`：共享方向核心存在，固定最小秩规则错误；
- `B`：方向共享但估计器/求解器不稳定；
- `C`：全局共享子空间本身不成立。

## 5. 若 A 成立，候选方法怎样从理论自然导出

### 5.0 若首先得到 estimator instability：用凸松弛区分优化失败与结构失败

不能因多初始化不稳定就直接换更复杂网络。令 Fantope

\[
\mathcal F_r=\{P=P^\top:0\preceq P\preceq I,\ \operatorname{tr}(P)=r\}.
\]

把非凸 rank-\(r\) projector 约束放宽为 \(P\in\mathcal F_r\)：

\[
\min_{P\in\mathcal F_r}
\sum_i\left\|J_i(I-P)\delta_i\right\|_{W_i}^2.
\]

对每个样本，\(P\mapsto W_i^{1/2}J_i(I-P)\delta_i\) 是仿射映射，平方范数为凸函数；
\(\mathcal F_r\) 也是凸紧集，因此该松弛存在全局最优解，且不会把随机初始化方差与结构异质性混在一起。
它只是一项 estimator diagnosis：最优 \(P\) 可以具有分数特征值，并不自动对应 rank-\(r\) 子空间；若
取 top-\(r\) 特征向量舍入，还必须报告松弛目标与舍入目标的 gap。

该诊断给出可区分预测：

- 若凸最优与舍入解跨重复稳定，而现有 Stiefel 解不稳定，主要问题属于求解器/参数化；
- 若凸最优本身对 group resample 高度不稳定，不能再用“非凸局部极值”挽救共享结构；
- 若凸软解稳定但任何 projector 舍入都有大 gap，证据支持软 task metric，而非固定低秩接口；
- 无论哪种情况，都不能跳过有限 patch、DINO 可恢复性与 matched 方法对照。

只有正式 R006V0F 返回 RANK_ESTIMATION_UNSTABLE 时，才允许把这一凸审计预注册为下一实验；当前不
执行，也不改变本轮路线。Fantope 是已有凸谱松弛对象，不能作为 AORTA 创新；文献出处与合法借用边界见
references/2026-10-27_r006_fantope_boundary.md。

### 5.1 若共享方向稳定但有效维数异质

可以预注册“共享有序 flag + 自适应有效维数”，而不是事后固定 rank32。令

\[
B=[b_1,\ldots,b_R],\quad B^\top B=I,\quad
P_r=B\,\mathrm{diag}(\mathbf 1[k\le r])B^\top,
\]

并在 discovery 上联合优化多个预注册 rank 的 task-weighted residual：

\[
\min_{B^\top B=I}\sum_{r\in\mathcal R}\lambda_r
\mathbb E\|J(I-P_r)\delta\|_W^2.
\]

该结构保证不同 rank 是同一有序 flag 的前缀，消除了“各 rank 学到不同局部解”这一混淆。部署时若
需要输入条件化门控，其门控只能依赖因果可用视觉/状态变量，并必须具有复杂度惩罚、拒判和 matched
fixed-rank基线；不得使用 outcome 或 test label 选维数。

这仍只是候选理论。它的 ICCV/Best-Paper 价值只有在以下链条同时成立时才存在：

\[
\text{稳定共享方向}\Rightarrow
\text{有限干预介导}\Rightarrow
\text{DINO增量可恢复}\Rightarrow
\text{matched adapter不可替代}\Rightarrow
\text{NavDream/跨planner行为收益}.
\]

## 6. 当前裁决

R006V0 原 fixed-global-rank 分支停止，R006V1、CPTR 与 NavDream 不解锁。下一步只允许预注册的
inner failure decomposition；根据 A/B/C 结果决定修正 estimator、提出 adaptive-rank 方法，或放弃
全局子空间假设。保持这一可证伪性，比强行让当前实验“通过”更接近 Best Paper 标准。
