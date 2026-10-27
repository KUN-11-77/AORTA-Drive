# AORTA DINO Interface First-Principles Theory V1

状态：`MECHANISM_FIRST_CANDIDATE__LOW_RANK_STOPPED__NO_METHOD_CLAIM`。
日期：2026-10-27。

## 0. 论文锚点

Constant Eye 已证明冻结 DINOv3-H+ 可以成为跨规划范式的外观稳定视觉底座。AORTA 不重复这一贡献，而研究更窄、更难的问题：

> 当 DINOv3-H+ 与已部署 planner 均被冻结时，稳定视觉证据为什么仍可能经过接口变成不稳定决策，以及怎样用最小、任务条件化接口修复，而不重训 planner、不破坏 clean 行为？

所有后续正式实验统一使用 DINOv3-H+；backbone、权重、预处理、token 顺序与缓存 provenance 冻结。

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

它们通过同一冻结审计分层裁决，允许 `AMPLIFICATION`、`COMPRESSION`、`MIXED` 与 `UNRESOLVED`；不能为了故事强迫唯一归因，也不能因为某个相关性显著而跳到预设网络。full-token probe 失败只停止当前接口与读出合同，不证明 DINO 在信息论意义上不足。

## 4. 因果审计与信息审计

### 4.1 接口交换

实际系统包含两条随外观共同变化的路径：

\[
D^a=\Phi(X^a),\qquad
(Z_n^a,Q^a)=\Psi_{native}(X^a,H),
\]

\[
\widehat O^a=T\!\left(Z_n^a+R_\theta(Q^a,D^a,H),H\right).
\]

因此同一 \((S,H)\) 下必须固定随机性并做因子化交换：

- 固定 shifted native path，只交换 clean / shifted / geometry-matched mismatch / removed DINO tokens；
- 固定 DINO path，只交换 clean / shifted native activation 与 query；
- joint clean/shift swap；
- adapter/bridge 后表示与已登记完整因果切口。

clean token interchange 只提供特权诊断上界，不是部署算法。wrong-scene/removed 可能是分布外输入，因此必须结合任务误差、组外 probe 与 donor 账本解释。

### 4.2 信息保留 probe

对同一功能目标 \(Y\) 比较容量与训练预算匹配的：

- full DINO tokens；
- compressed adapter tokens；
- compressed + skip full tokens；
- history/native-only；
- correct / mismatch / removed。

平方损失下的总体 Bayes risk 恒等式

\[
\mathcal R^*(Y\mid B)-\mathcal R^*(Y\mid B,D)
=\mathbb E\|\mathbb E[Y\mid B,D]-\mathbb E[Y\mid B]\|^2
\]

只解释视觉增量的总体对象；有限网络实验只能支持指定函数类和数据合同下的组外可预测增量，不能宣称信息论充分或不可能。

## 5. 由支持机制导出的唯一候选方法

方法必须随诊断收缩：只支持压缩丢失时，先解锁 full-token bridge；只支持接口放大时，先在相同结构上比较 ordinary functional consistency；两者同时成立且各自消融不可替代时，才允许组合为 **Functional Quotient Bridge (FQB)**。它以 native planner query/interface states 为 query、完整 DINO tokens 为 key/value，输出一个注入冻结 planner 完整切口的残差：

\[
R_\theta(Q,D,H)=g_\theta(Q,D,H)\odot
\operatorname{CrossAttn}_\theta(Q,D,D),
\]

\[
\widehat Z=Z_{native}+R_\theta(Q,D,H),
\qquad \widehat O=T(\widehat Z,H).
\]

新增可训练组件只有 bridge；只有出现不确定性失败证据后才允许第二个拒判头。DINO 与原 planner 全冻结。

训练目标为

\[
\mathcal L=
\mathcal L_{task}^{clean}
+\lambda_o\mathbb E_a
\|\widehat O^a-\operatorname{sg}(O_{ref}^{clean})\|_W^2
+\lambda_{id}\|R_\theta(Q,D^{clean},H)\|_2^2
+\lambda_r\|R_\theta(Q,D^a,H)\|_2^2.
\]

这些项分别对应任务锚定、appearance-orbit 功能保持、clean residual 抑制与最小干预。clean residual penalty 不保证 clean no-harm；后者必须由真实输出和 official utility 的组级非劣检验裁决。这些项都不是单独的新理论；方法价值只在机制发现使其成为必要结构、并且 matched ordinary consistency/full-token adapter 不能替代时成立。功能目标 \(Y\)、clean reference \(O_{ref}^{clean}\)、输出度量 \(W\) 与完整切口均须结果前冻结。

## 6. 为什么不继续低秩或直接采用 VJP

R006V0/R006V0F 已使 fixed-global-rank/CPTR 停止。Fantope 只会继续诊断旧接口，不直接回答 DINO 如何修复 planner，因此不进入主实验。

sample-conditioned VJP repair 的 ridge 公式正确，但一阶输出为 Gram-filtered direct-output residual。它只有在额外实验证明 task-shaped restriction 相对同 reader 的无约束 residual 带来不可替代的 clean no-harm 或组外收益时才有资格；否则保留为附录基线。

## 7. 三个实验块与停止规则

1. **DINO 接口因果审计：**裁决信息不足、接口放大、压缩丢失或场景先验；不通过不训练新方法。
2. **FQB matched study：**统一 DINOv3-H+，比较 Constant Eye、full-token ordinary reader、matched residual adapter、ordinary output consistency、direct-output residual 与 FQB；若简单方法等效，删除 FQB 创新。
3. **行为闭环：**NavDream zero-shot、至少三类 planner、clean non-inferiority、official utility、hard-failure net reduction、batch-1 P50/P95；任一硬门失败停止 Best-Paper 主张。

统计单位为 recording group；seed 不扩充独立样本量。用于反复提出假设的 groups 只作 discovery，必须另留前瞻确认单位。

## 8. 一分钟故事（候选，不是已成立结论）

> Constant Eye 证明冻结 DINOv3-H+ 能让重新训练的规划器更抗外观变化，但部署中的原 planner 仍然需要一个正确接口。我们研究的不是让特征处处不变，而是同场景外观 orbit 上冻结 planner 的功能是否保持。我们先用 token/接口交换证明：残余 DINO drift 在通用接口中被放大或关键任务证据被压缩；随后用一个原生 planner query 驱动的 full-token Functional Quotient Bridge，只向完全冻结的 planner 注入完成当前任务所需的视觉证据，并直接约束 clean no-harm 和 orbit 功能保持。若该结构不能击败相同 DINO、相同预算的普通 adapter 与 consistency，它就不是答案。

## 9. 当前主张边界

当前只完成理论重构与实验设计。尚未证明接口放大/压缩机制、FQB 有效、NavDream 增益、跨 planner 泛化、实时性或安全收益。Best-Paper 只是执行标准，不是当前资格。

## 10. 证据生长原则：不再围绕实现假设自证

实验链固定为：**已有正线索 → 最小竞争解释 → 能区分解释的有限干预 → 组外确认 → 唯一许可的方法**。
其中 DINOv3-H+ 的既有正增益是起点；low-rank、Fantope、VJP、cross-attention、某一层或某一 rank
都只是可淘汰实现假设。任何假设若在预冻结的区分实验中明显不是答案，立即停止，不通过增加 rank、
种子、loss 或模块延长寿命。

因此 R007 的第一科学产物不是新网络，而是一张可复算的 DINO-to-planner 信息流机制图。只有当该图
显示某个瓶颈可重复、组外成立且简单对照不能解释时，网络结构才由瓶颈反推。正式执行合同见
`experiments/R007/R007V0_DINO_INTERFACE_CAUSAL_AUDIT_PREREGISTRATION.md`。
