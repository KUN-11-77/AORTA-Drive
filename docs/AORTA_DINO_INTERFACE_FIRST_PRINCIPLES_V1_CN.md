# AORTA DINO Interface First-Principles Theory V1

状态：`MECHANISM_FIRST_CANDIDATE__LOW_RANK_STOPPED__NO_METHOD_CLAIM`。
日期：2026-10-27。

## 0. 论文锚点

Constant Eye 已证明冻结 DINOv3-H+ 可以成为跨规划范式的外观稳定视觉底座。AORTA 不把 DINO
附着到旧 SDF、低秩、Differential Query 或 CPTR 体系上，而从 DINO 重新定义研究对象：

> 当 DINOv3-H+ 与已部署 planner 均被冻结时，稳定视觉证据为什么仍可能经过接口变成不稳定决策，以及怎样用最小、任务条件化接口修复，而不重训 planner、不破坏 clean 行为？

所有后续正式实验统一使用 DINOv3-H+；backbone、权重、预处理、token 顺序与缓存 provenance 冻结。
旧方法只保留为失败证据或必要基线，不具有默认进入新架构的资格。

### 0.1 新主线的第一性问题

给定冻结 DINO 证据 (D)、任务上下文 (B) 与决策目标 (Y)，新主线只研究三个对象：

1. **条件充分性：**在已知 (B) 后，(D) 是否仍降低预测 (Y) 的 Bayes risk；
2. **appearance-orbit 功能保持：**信息保持外观变化下，最终任务函数而非全部 token 是否稳定；
3. **接口可实现性：**何种最小接口能保留前两项，同时不以场景先验、更多参数或重训 planner 制造增益。

因此“低秩是否存在”不是论文问题；它最多是某个接口的可淘汰性质。论文问题是 DINO 任务证据的
**充分性、传输与使用**。

相关命题、证明假设与决策误差边界见
`docs/AORTA_DINO_DECISION_SUFFICIENCY_THEORY_V1_CN.md`。

## 1. 数学对象：appearance orbit 上的功能商

设物理场景 \(S\)、外观 \(a\)、历史 \(H\)，冻结 DINO 特征为

\[
D^a=\Phi(g(S,a)).
\]

视觉接口 \(A_\theta\) 与冻结 planner tail \(T\) 给出

\[
O_\theta^a=T(A_\theta(D^a),H).
\]

将外观分为信息保持干预 \(\mathcal A_{pres}\) 与可能破坏可观测性的干预 \(\mathcal A_{loss}\)。前者要求物理任务和完成决策所需证据在观测中仍可恢复；后者可包含严重低照度、遮挡或传感退化。信息保持 appearance orbit 为 \(\mathcal O_{pres}(S)=\{g(S,a):a\in\mathcal A_{pres}\}\)。目标不是要求全部 DINO tokens 不变，而是组合函数只在信息保持 orbit 上维持任务功能：

\[
T(A_\theta(D^a),H)=T(A_\theta(D^{a'}),H),
\quad a,a'\in\mathcal A_{pres},
\]

同时对不同场景保留任务可分性。对 \(\mathcal A_{loss}\) 不要求强制不变，而评价风险、校准与拒判。常数表示虽满足不变性，却不满足任务充分性，因此必须同时报告 task utility 与 non-collapse。

## 2. 为什么 DINO 稳定不推出决策稳定

令 \(\delta_D=D^{a'}-D^a\)，沿 \(D_t=D^a+t\delta_D\) 有精确路径积分

\[
O_\theta^{a'}-O_\theta^a
=\int_0^1J_{T\circ A_\theta}(D_t)\delta_D\,dt.
\]

所以 feature drift \(\|\delta_D\|\) 小不是充分条件；接口的方向性增益可以放大残余 drift。该公式只是机制分解，不证明现实中放大存在。存在性必须由真实有限干预、接口交换和 matched controls 支持。

为避免坐标尺度伪像，主结论使用物理单位归一化的 planner output/trajectory metric；DINO 与 hidden-state drift 只作描述性账本。任何“放大率”都必须配合 finite output change 与接口层 intervention，不能单靠比值或相关性。

## 3. 四个可并存的竞争解释

1. **信息不足：**完整 clean DINO 也不能形成任务恢复上界；
2. **接口放大：**DINO residual drift 小，但 adapter/tail 把它放大为显著输出变化；
3. **压缩丢失：**完整 DINO 可组外恢复任务目标，通用 adapter 后表示不行；
4. **场景先验：**correct DINO 不优于 mismatch/removed。
5. **层间互补：**预冻结的中间层在最后层/last-four mean 之外提供组外任务增量，并能经真实接口影响
   planner 输出或效用。

它们通过同一冻结审计分层裁决，允许 `AMPLIFICATION`、`COMPRESSION`、`LAYER_COMPLEMENTARITY`、
`MIXED` 与 `UNRESOLVED`；不能为了故事强迫唯一归因，也不能因为某个相关性显著而跳到预设网络。
full-token probe 失败只停止当前接口与读出合同，不证明 DINO 在信息论意义上不足。单层/多层 probe
提升若不能传递为真实 planner consequence，只能写成可读性结果，不能授权多层融合方法。

## 4. 因果审计与信息审计

### 4.1 先识别真实拓扑，再选择接口交换

不能预设所有系统都有 native 与 DINO 两条可分离视觉路径。源码核验表明，Constant Eye 的
TransFuser-DINO 由 DINO tokens 经 projector、CNN 重采样、Transformer decoder 和 trajectory head
形成 DINO-only 路径；对它强行定义 native×DINO 因子交换没有对应计算对象。此类系统应做顺序切口：

\[
D^a\rightarrow P(D^a)\rightarrow C(P(D^a))
\rightarrow Q^a\rightarrow O^a,
\]

依次把 clean stage value 替换进 shifted forward，测量每个完整切口的有限恢复，并以 full raw-token
replacement 作为机械上界。只有真实系统满足

\[
D^a=\Phi(X^a),\qquad (Z_n^a,Q^a)=\Psi_{native}(X^a,H),
\]

且两条路径在源码和 checkpoint 中可独立固定时，才允许对

\[
\widehat O^a=T\!\left(F(Z_n^a,Q^a,D^a,H),H\right)
\]

执行 native×DINO 的 2×2 因子化交换。实验设计服从真实计算图，而不是要求真实模型服从统一故事。
两种拓扑都必须包含 correct / geometry-matched mismatch / removed DINO、identity replacement 与
已登记完整因果切口。

clean token interchange 只提供特权诊断上界，不是部署算法。wrong-scene/removed 可能是分布外输入，
因此必须结合任务误差、组外 probe 与 donor 账本解释。跨 planner 汇总只比较共同 estimand，不把不同
拓扑的 stage 名称强行一一对应。

### 4.2 信息保留 probe

对同一功能目标 \(Y\) 比较容量与训练预算匹配的：

- full DINO tokens；
- compressed adapter tokens；
- compressed + skip full tokens；
- last layer、冻结 last-four mean 与由 discovery-only 规则产生的有限中间层候选；
- 预冻结的单层与等维层组合；
- history/native-only；
- correct / mismatch / removed。

平方损失下的总体 Bayes risk 恒等式

\[
\mathcal R^*(Y\mid B)-\mathcal R^*(Y\mid B,D)
=\mathbb E\|\mathbb E[Y\mid B,D]-\mathbb E[Y\mid B]\|^2
\]

只解释视觉增量的总体对象；有限网络实验只能支持指定函数类和数据合同下的组外可预测增量，不能宣称信息论充分或不可能。

## 5. 方法不继承旧设计，而由 DINO 机制证据唯一授权

当前不预先命名或锁定 FQB、cross-attention、低秩 projector 或某个 planner 内部切口。R007 先估计两个
可复算的量：

\[
\Delta_{suf}(C)=\mathcal R^*(Y\mid B,C(D))-\mathcal R^*(Y\mid B,D),
\]

其中 \(C\) 是真实接口中的压缩；以及

\[
\Delta_{orb}(F)=\mathbb E_{a\in\mathcal A_{pres}}
d_Y(F(B,D^a),F(B,D^{clean})),
\]

其中 \(F\) 是 DINO 到任务输出的完整组合函数。前者刻画函数类与数据合同下的证据损失，后者刻画
信息保持外观 orbit 上的功能漂移。它们分别决定允许的最小方法：

- 若 \(\Delta_{suf}\) 无可靠证据：禁止为“压缩修复”增加 full-token reader；
- 若 \(\Delta_{orb}\) 无可靠证据：禁止增加功能一致性约束；
- 若 correct DINO 不优于 mismatch/removed：停止当前视觉主张，不能用更复杂接口救援；
- 只支持压缩丢失：比较最小 full-token skip/reader 与容量匹配普通 adapter；
- 只支持层间互补：比较固定等权层组合与同参数简单 learned scalar mixture；若简单混合等效，不增加
  多层路由器；
- 只支持接口放大：在原接口上比较最小 task-output orbit regularization；
- 两者独立成立：才允许组合成一个 task-conditioned DINO evidence interface，并逐项做删除消融。

通用训练包络只表达目标，不预设网络：

\[
\mathcal L(\theta)=\mathcal L_{task}(F_\theta(B,D^{clean}),Y)
+\lambda_{pres}\,\mathbb E_{a\in\mathcal A_{pres}}
d_Y(F_\theta(B,D^a),F_\theta(B,D^{clean}))
+\lambda_{comp}\,\Omega_\theta.
\]

\(\Omega_\theta\) 只能是被 R007 机制授权的最小结构约束；不能默认取低秩、稀疏、cross-attention
或旧 AORTA loss。稳定性项必须和 task loss、correct-vs-mismatch/removed、non-collapse 与 official clean
utility 一起评价，避免得到稳定但错误的常数映射。新增可训练组件默认不超过一个；DINO 与被比较的
planner 保持冻结，除非 faithful Constant Eye baseline 按原文协议单独重训并明确标注。

## 6. 为什么不继续低秩或直接采用 VJP

R006V0/R006V0F 已使 fixed-global-rank/CPTR 停止。Fantope 只会继续诊断旧接口，不直接回答 DINO 如何修复 planner，因此不进入主实验。

sample-conditioned VJP repair 的 ridge 公式正确，但一阶输出为 Gram-filtered direct-output residual。它只有在额外实验证明 task-shaped restriction 相对同 reader 的无约束 residual 带来不可替代的 clean no-harm 或组外收益时才有资格；否则保留为附录基线。

## 7. 三个实验块与停止规则

1. **DINO 接口因果审计：**裁决信息不足、接口放大、压缩丢失或场景先验；不通过不训练新方法。
2. **机制授权的 matched study：**统一 DINOv3-H+，只实现 R007 唯一授权的最小接口，并与
   Constant Eye、full-token ordinary reader、matched residual adapter、ordinary output consistency、
   direct-output residual 比较；若简单方法等效，删除复杂方法创新。
3. **行为闭环：**NavDream zero-shot、至少三类 planner、clean non-inferiority、official utility、hard-failure net reduction、batch-1 P50/P95；任一硬门失败停止 Best-Paper 主张。

统计单位为 recording group；seed 不扩充独立样本量。用于反复提出假设的 groups 只作 discovery，必须另留前瞻确认单位。

## 8. 一分钟故事（候选，不是已成立结论）

> Constant Eye 证明 DINOv3-H+ 能提升恶劣外观下的自动驾驶，但没有解释真正起作用的是哪些任务证据，
> 也没有说明这些证据经过不同 planner 接口后为何仍会丢失。我们把 DINO 固定为共同视觉底座，直接审计
> 从 token 到驾驶输出的真实信息流：完整证据是否任务充分、哪一段压缩丢失它、哪一段放大残余外观漂移。
> 然后只针对被组外实验确认的瓶颈设计一个最小接口，并要求它在相同 DINO、参数与算力下击败普通
> adapter/consistency，同时满足 clean 非劣和实时约束。机制不成立，方法就不出现。

## 9. 当前主张边界

当前只完成理论重构与实验设计。尚未证明接口放大/压缩机制、任何新 DINO 接口有效、NavDream 增益、
跨 planner 泛化、实时性或安全收益。Best-Paper 只是执行标准，不是当前资格。

## 10. 证据生长原则：不再围绕实现假设自证

实验链固定为：**已有正线索 → 最小竞争解释 → 能区分解释的有限干预 → 组外确认 → 唯一许可的方法**。
其中 DINOv3-H+ 的既有正增益是起点；low-rank、Fantope、VJP、cross-attention、某一层或某一 rank
都只是可淘汰实现假设。任何假设若在预冻结的区分实验中明显不是答案，立即停止，不通过增加 rank、
种子、loss 或模块延长寿命。

因此 R007 的第一科学产物不是新网络，而是一张可复算的 DINO-to-planner 信息流机制图。只有当该图
显示某个瓶颈可重复、组外成立且简单对照不能解释时，网络结构才由瓶颈反推。正式执行合同见
`experiments/R007/R007V0_DINO_INTERFACE_CAUSAL_AUDIT_PREREGISTRATION.md`。
