# AORTA 与 PEA 升级路线的 ICCV 方法裁决

日期：2026-10-24  
状态：`INTERNAL_METHOD_SPEC__NO_NEW_EXPERIMENT__NOT_ICCV_READY`

## 1. 当前裁决

AORTA 当前具有较完整的问题定义、条件性数学合同和可实现系统链，但尚未达到 ICCV 方法论文标准。阻塞项不是公式数量，而是：

1. decision-relevant representation 的不可替代性尚未成立；
2. QuotientFieldPredictor 尚未胜过 direct edge + projection、direct pair、SDF/BEV、READ-style 与 FlowDrive-style；
3. held-out 视觉必要性和决策收益尚未完成；
4. 当前道路 teacher 只覆盖 nominal road factor，不是完整驾驶充分统计量。

因此，当前应评价为 `REVISE`，而不是 `READY`。

## 2. PEA 失败揭示的结构性缺口

PEA 通过 source moments 与 target/test features 的几何对齐提出校正方向。抽象写成

\[
d_{\mathrm{pea}}(h)=-\nabla_h D_{\mathrm{mom}}(h;\mathcal S),
\]

其中 \(D_{\mathrm{mom}}\) 衡量均值、方差或协方差相对 source statistics 的偏离。

这个目标没有编码候选决策的符号。若候选 \(a,b\) 的 margin 是 \(m_{ab}(h)\)，则一阶变化为

\[
\Delta m_{ab}
\approx
g_{ab}(h)^\top d_{\mathrm{pea}}(h),
\qquad
g_{ab}(h)=\nabla_h m_{ab}(h).
\]

即使 \(D_{\mathrm{mom}}\) 下降，也完全可能有

\[
g_{ab}^\top d_{\mathrm{pea}}< -m_{ab},
\]

从而使原本正确的候选排序翻转。PEA 的全局统计目标与决策方向之间不存在一般性的单调关系。

本地 confirmation 与该分析一致：PEA-MG 相对 Native 为负，27 个 new-hard 主要来自 DAC/DDC/NC 硬约束；feature/trajectory recovery 与 PDMS 增益相关很弱，固定 magnitude-style Guard 无法识别有害校正。

## 3. 是否应以 PEA 为主线

### 不建议直接合并进当前 AORTA 主线

原因：

- PEA 使用目标测试流统计，AORTA 当前主张是 source-only appearance robustness，假设不同；
- PEA 校正 DiffusionDrive 内部特征，而 AORTA 当前视觉机制使用冻结 DINOv3 与独立关系头，接口不同；
- 同时宣称新表示、新校准、新 TTA 和新仲裁会形成 contribution sprawl；
- 当前没有证据表明对 DINO relation branch 做 PEA 对齐有额外 headroom。

### 可以保留为条件备选路线

如果新的 outcome-blind development 诊断证明 PEA correction 中确实存在可预测的正/负方向，才允许建立一个 task-constrained PEA；否则 PEA 只作为失败基线和机制动机。

## 4. 条件备选：Task-Constrained PEA

该备选不改变 frozen planner 的候选生成。Native planner 先产生固定候选集；PEA correction 只作用于 shadow relation branch，用于估计候选之间的道路任务 margin。

```text
RGB/history
  ├─> frozen planner ─> fixed candidate set C
  └─> frozen visual encoder
        ├─> native relation estimate
        └─> PEA proposal d_pea
              -> task-constrained projection
              -> corrected relation estimate

fixed candidates + relation estimates
  -> pairwise lower confidence margins
  -> keep native / choose challenger / abstain
```

### 4.1 决策敏感子空间

堆叠候选对比 Jacobian：

\[
G_{\mathcal C}(h)
=
\begin{bmatrix}
g_{10}(h)^\top\\
\vdots\\
g_{K0}(h)^\top
\end{bmatrix}.
\]

PEA proposal 可分为

\[
d_{\parallel}=P_Gd_{\mathrm{pea}},
\qquad
d_{\perp}=(I-P_G)d_{\mathrm{pea}},
\]

其中

\[
P_G=G^\top(GG^\top+\lambda I)^{-1}G.
\]

\(d_{\parallel}\) 是一阶影响候选 margin 的部分；\(d_{\perp}\) 在局部近似下属于 decision-null direction。该分解本身是标准线性代数，不作为原创性声明。

### 4.2 Task-constrained projection

对 PEA proposal 解一个小型凸投影：

\[
d^*=\arg\min_d
\frac12\|d-d_{\mathrm{pea}}\|_H^2+\frac{\lambda}{2}\|d\|_2^2
\]

满足

\[
\widehat m_{a0}(h+d)-\epsilon_{a0}\ge \gamma
\]

以及 trust region

\[
\|d\|_2\le r.
\]

若约束不可行、bound 不可靠或 correction 没有带来正的 lower-bound improvement，则 abstain 并保留 native。

该设计的核心不是“让 PEA 更接近 source”，而是“在保留 PEA 可恢复的 appearance correction 时，禁止其跨越当前可见的道路任务决策边界”。

### 4.3 条件性安全命题

设 margin predictor 满足

\[
|m_{a0}(h+d)-\widehat m_{a0}(h+d)|\le\epsilon_{a0}
\]

且部署规则只在

\[
\widehat m_{a0}(h+d)-\epsilon_{a0}>0
\]

时选择 challenger。则在该误差事件成立时，被选择候选在 teacher 所定义的道路任务 margin 上不劣于 native。

这个结论不覆盖动态碰撞、信控、完整 PDMS、闭环安全或任意 appearance OOD；它也依赖固定 candidate set 和有效的 margin bound。

## 5. 与 AORTA decision quotient 的关系

Task-Constrained PEA 不应成为第二个并行贡献。它只有两种合法位置：

1. **强基线位置：** 使用 AORTA 的 decision margin 对 PEA proposal 做最小投影，验证“任务约束能否挽救通用对齐”；
2. **方法实现位置：** 若 H7 证明 PEA proposal 提供独立增益，可把它作为 decision-quotient operator 的一个实例，但论文仍只保留“decision-aligned correction”这一项主贡献。

如果不加 PEA 的 relation head 已经达到同等效果，应删除 PEA 分支。

## 6. 最小资格实验

### Gate P0：Oracle selective headroom

在新的 development split 上，仅判断 PEA 与 Native 之间是否同时存在足够的正收益和负收益，以及 oracle selective PEA 是否有非平凡上界。若没有 headroom，停止 PEA 升级。

### Gate P1：Signed predictability

比较以下量对 `PEA beneficial vs harmful`、new-hard 和 pairwise regret 的 group/session-level 判别：

- PEA moment discrepancy；
- feature movement norm；
- trajectory recovery；
- native margin；
- decision-contrast lower bound。

只有最后一项在未见 sessions 上稳定提供增量解释力，才继续。

### Gate P2：Frozen method comparison

在新的 confirmation 上比较：

1. Native；
2. PEA-M；
3. 原 PEA-MG Guard；
4. task-null projection；
5. task-constrained PEA；
6. AORTA without PEA。

主指标为 clean harm、new-hard、coverage、road-factor regret、DAC/DDC 与完整 PDMS 分项。统计单位为 recording session。

## 7. ICCV-ready 判据

无论采用纯 decision quotient 还是 PEA-derived 实例，都必须满足：

1. 一个主贡献，不超过两个新训练组件；
2. 理论对象与实际算法完全一致，定理的假设能被实验审计；
3. 对 READ/FlowDrive、direct pair、direct edge + projection、SDF/BEV 和 Native/PEA/MG-style 做公平对照；
4. 未见 scene 与 appearance 上有效，并证明正确图像是必要输入；
5. 至少两个 planner/candidate providers 上获得决策增益；
6. clean non-degradation、abstention/coverage 曲线和失败案例完整报告；
7. 不能靠更多候选、更大 backbone、目标域标签或不同 scorer 获胜。

## 8. 推荐决策

主线继续保持 decision quotient；PEA 升级只做条件备选，不立即开发。R004H6 完成前不改网络。R004H6 通过后，先做 H7-B decision relevance diagnosis；只有其同时显示 PEA selective headroom 与 signed predictability，才启动 Task-Constrained PEA 的最小实验。

这能避免为了利用一个失败基线而偏离原问题，也把 PEA 的负结果转化为明确的理论缺口：分布对齐必须受任务方向与决策裕量约束。

