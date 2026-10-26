# AORTA Best-Paper 候选级实现标准 v1.0

状态：`MANDATORY_PROJECT_CONTRACT`  
生效日期：2026-10-25  
适用范围：AORTA 后续方法实现、训练、评估、审计、论文主张与公开仓库  
不追溯修改：已经冻结并运行中的 R005E2B；该实验完成后按本标准审计，不得中途改协议

## 0. 标准的含义

本文件把“按 Best Paper 水准实现”解释为一组可检查、可失败的工程—科学硬门，而不是对奖项、
接收或正结果的承诺。我们能够强制保证的是：问题定义不偷换、实现忠于数学对象、对照公平、
结果可复算、失败被保留、主张不超过证据。任何实验结果均不能保证 Best Paper；只有同时出现
重要发现、不可替代的方法优势和完整行为闭环，AORTA 才具备 Best Paper 候选资格。

本项目拒绝两种伪高水平实现：

1. 用更多模块、公式或计算量制造复杂感；
2. 用严格工程审计替代真正的科学发现。

核心原则是：**最小但充分的数学对象，最强但公平的竞争解释，最短但完整的证据链。**

## 1. 唯一主对象与实现不变量

### 1.1 科学对象

在冻结 planner 给出的合法 ragged 候选集与因果历史

\[
B=(\mathcal C,h),\qquad \mathcal C=\{\tau_0,\ldots,\tau_{K-1}\},
\]

上，AORTA 预测 native-relative contextual relation

\[
Y_B=(\Delta_{10},\ldots,\Delta_{K-1,0}),\qquad
\Delta_{i0}=U_{ctx}(\tau_i;h)-U_{ctx}(\tau_0;h).
\]

最终方法候选逼近条件分布

\[
p_\theta(Y_B\mid I,B),
\]

而不是把 BEV、SDF、道路场、风险热图、feature invariance 或完整 world reconstruction 当作主目标。

### 1.2 潜在决策商、观测预测商与外观可辨识性

对固定的 \(B\)，潜在状态按任务分布等价：

\[
S\equiv_B^{\mathrm{lat}}S'
\iff
p(Y_B\mid S,B)=p(Y_B\mid S',B).
\]

有限观测上的预测等价关系为

\[
I\equiv_B^{\mathrm{obs}} I'
\iff
p(Y_B\mid I,B)=p(Y_B\mid I',B).
\]

任务保持的外观不一定信息保持。只有当任意观测碰撞

\[
g(S_1,a_1)=g(S_2,a_2)
\]

都蕴含两个状态属于同一潜在决策商类时，才存在对全部合法外观确定性、外观不变且决策充分的表示。
否则实现必须允许条件分布变宽并拒判，禁止用强制feature/point consistency制造过度自信。

实现中的 appearance consistency、distributional head、visual controls 和 arbitration 都必须服务于
这一对象。任何 loss 若不能说明自己约束了上述条件分布的哪一性质，不得进入主方法。

### 1.3 六个不可破坏的不变量

1. **Domain invariant：** 只在冻结、合法、带 provenance 的候选域上定义关系；禁止 silent drop。
2. **Causality invariant：** 历史只能包含决策时可用的信息；不得用 future、teacher 或 target 回填。
3. **Coordinate invariant：** 图像、候选、历史、occupancy 与 query 必须显式声明坐标系、单位和时间轴。
4. **Relation invariant：** 部署输出为同一 native-star；任意 challenger 关系由差分获得，禁止循环偏好。
5. **Role invariant：** source、inner validation、development、calibration、test、NavDream/target 角色隔离。
6. **Claim invariant：** relation、contextual utility、policy value、驾驶行为和安全是不同层级，禁止外推。

违反任一不变量的运行标记为 `INVALID`，不能通过补阈值、删样本或事后解释恢复为科学证据。

## 2. 模块准入门：没有理由的模块不得实现

任何新模块、loss、数据处理或推理规则必须在代码前回答以下五问，并写入相应 preregistration/ADR：

1. 它由哪个已观察失败或严格反例触发？
2. 它改变哪个数学对象或缩小哪个误差项？
3. 为什么最简单的替代方案不能解决？
4. 哪个容量、数据和预算匹配的消融可以否定它？
5. 删除它时，预注册预测是什么？

未回答五问：`REJECT_FROM_MAIN_METHOD`。只有“可能提升指标”“文献常用”“理论看起来高级”均不构成
准入理由。默认最终方法最多新增两个可训练部件：

- differential query encoder；
- global-evidence distributional relation head。

额外模块必须由新的、已审计失败门单独解锁。

## 3. 架构实现合同

### 3.1 Differential evidence 只负责提出问题

对 challenger 与 native 的完整时空 occupancy measure，构造

\[
\nu_{i0}=\mu_i-\mu_0=\nu^+_{i0}-\nu^-_{i0}.
\]

signed support 编码器生成 query；共享的全局视觉 token 作为 key/value 回答。实现不得把差异区域
假定为完整效用的解析充分统计量，因为交通参与者、信号、遮挡和全局道路语境可能是非加性的。

### 3.2 必须保持的网络性质

- backbone 对每个 anchor 只计算一次，并被所有候选共享；
- 支持 ragged \(K\) 与显式 mask，padding 不得影响 logits、loss 或统计；
- 对 challenger 顺序 permutation-equivariant；native 身份显式且唯一；
- native-relative 输出可积：\(\widehat\Delta_{ij}=\widehat\Delta_{i0}-\widehat\Delta_{j0}\)；
- 空间投影、token 顺序和 camera/time binding 由测试固定；
- removed/mismatched/correct visual arms 使用同一可训练参数、优化器与输入预算；
- 不确定性表示必须对应可检验的条件误差，不得把 softmax confidence 当作保证。

### 3.3 公共接口要求

所有公共张量必须有类型、shape、单位、坐标系、时间范围和 mask 语义。推荐的概念接口为：

```text
VisualEvidence(tokens, camera_ids, frame_times, spatial_binding)
CandidatePanel(poses, valid_mask, native_index, dt, frame)
CausalHistory(poses, valid_mask, dt, source, frame)
RelationDistribution(location, scale_or_quantiles, candidate_mask)
ArbitrationDecision(selected_index, lower_bounds, abstain_reason)
```

禁止由数组长度推断时间尺度、由文件名猜测角色、由零向量隐式表示无效候选。

## 4. 数值与代码正确性硬门

### 4.1 Fail-closed

以下情况必须停止并产生明确错误：非有限数、空合法候选集、native 缺失或重复、坐标/单位不匹配、
时间轴不一致、cache hash 不符、未知数据角色、候选或历史被 silent truncate、sealed split 被访问。

禁止 silent interpolation、silent padding、silent clipping、silent fallback 或用默认值吞掉缺失字段。

### 4.2 测试金字塔

每个新方法变更至少包含：

1. 数学单元测试：反对称/可积性、mask、置换等变、边界和反例；
2. 数据合同测试：时间、坐标、hash、角色隔离、ragged panel；
3. 合成可学习性测试：正确视觉可学习，错配/移除不能凭接口泄漏标签；
4. 端到端小样本测试：训练、保存、恢复、推理与独立复算；
5. 确定性 replay：同 job、同 GPU、同软件栈产生冻结容差内相同输出。

修复 bug 时必须先保留能够复现该 bug 的测试。测试通过只证明实现资格，不证明方法有效。

### 4.3 数值账本

每次正式运行记录 dtype、混合精度、TF32、deterministic flags、CUDA/cuDNN、GPU 型号、峰值显存、
非有限值计数和 replay 容差。任何容差必须在读取科学结果前冻结，并有误差尺度依据。

## 5. 训练合同

### 5.1 数据与选择

- 以 recording group 为最小独立划分和统计单位，禁止 anchor 级泄漏与伪重复；
- 模型选择、early stopping 与超参数只使用 source inner folds；
- development/calibration/test/NavDream/target 不参与 checkpoint、增强、阈值或架构选择；
- donor mapping 在 outcome 前冻结并审计，不由标签或结果选择；
- 每个正式 arm 使用相同 folds、seeds、训练步数上限和选择规则。

### 5.2 优化与可比性

主方法和匹配基线固定：backbone、视觉 token、候选域、历史、监督、optimizer、scheduler、seed、
batch 定义、early stopping、训练 wall/GPU budget。报告参数量、trainable 参数量、FLOPs、显存和时间。

若最接近基线天然更重，优先提供原论文规模与预算匹配规模两种结果；不得通过削弱基线、给予 AORTA
更多标签或更长调参来制造优势。

### 5.3 运行纪律

- 正式运行前冻结 commit、config、manifest、checkpoint、seeds、统计与裁决；
- 不读取 partial scientific metrics 来改变协议；
- 资源利用率优化只能改变并行调度，不能改变单 job 的数据、顺序、优化或数值语义；
- primary 与 replay 完成前不得发布科学结论；
- 失败运行保留原目录与原因，不覆盖、不删除、不自动开始结果驱动的第三次运行。

## 6. 公平基线与消融合同

### 6.1 必须覆盖的竞争解释

最终方法至少需要面对以下容量与预算匹配对照：

- trajectory/history-only；
- correct / mismatched / removed visual；
- independent candidate scorer；
- generic set/full-context scorer；
- absolute score subtraction；
- unrestricted pairwise head（用于显示接口风险，不作为默认部署）；
- field/BEV/SDF/risk-map 路线；
- D-JEPA、World4Scorer、DA-WAM、GTRS 等最接近思想的可比实现或严格边界说明。

### 6.2 核心必要性实验

必须独立回答：

1. 视觉是否在 \((\mathcal C,h)\) 之外提供条件增量？
2. differential query 是否优于 matched generic scorer，而非只优于弱 baseline？
3. 优势是否集中在 near-tie、late-fork、复杂交互等理论预测子集？
4. appearance consistency 是否保留任务信息，而非产生常数坍缩？
5. distributional uncertainty 是否改善风险—覆盖和 selective regret？
6. arbitration 是否在 clean no-harm 下转化为行为收益？

若第 2 项不成立，删除 Differential Query 的核心创新主张；若第 1 项不成立，停止当前 RGB 恢复路线，
先诊断可观测性/对齐/监督，而不是增加网络复杂度。

## 7. 评价与统计合同

### 7.1 指标层级

不得用单一平均 MSE 代表成功。正式报告至少覆盖：

- proper relation risk / calibration；
- pair sign、top choice、near-boundary error；
- decision regret 与 oracle capture；
- appearance-induced relation/selection flips；
- risk–coverage 与 abstention rate；
- worst-group、失败组数和组级分布；
- candidate supply regret 与 selection regret 分解；
- clean no-harm、actual-history、sequential 与 cross-planner；
- batch-1 end-to-end P50/P95、峰值显存和随候选数 \(K\) 的缩放。

### 7.2 统计单位与不确定性

置信区间、配对检验和 bootstrap 以 recording group 为独立单位。anchor 可用于组内聚合，但不能作为
独立样本扩大显著性。报告 effect size、置信区间、最差组与预注册阈值；禁止只报告 p-value。

### 7.3 供给—选择分解

对全局合法最优 \(\tau^*\)、panel 最优 \(\tau_B^*\) 与选中 \(\hat\tau\)：

\[
U(\tau^*)-U(\hat\tau)
=
\underbrace{U(\tau^*)-U(\tau_B^*)}_{\text{supply regret}}
+
\underbrace{U(\tau_B^*)-U(\hat\tau)}_{\text{selection regret}}.
\]

AORTA 只直接优化 selection regret。候选 generator 的失败不能被归为 scorer 失败，反之亦然；最终论文
必须并列报告两者，防止固定 panel 的离线收益掩盖真实候选供给瓶颈。

## 8. 实时性与系统实现合同

最终系统必须满足冻结在线预算，而不是只报告 head 的 CUDA kernel 时间：

\[
T_{e2e}=T_{backbone}+T_{projection}+T_{query}+T_{relation}+T_{calibration}+T_{arbitration}.
\]

要求：

- batch=1、真实 ragged panel、冷启动与稳态均测；
- 报告 P50/P95/P99、峰值显存、CPU/GPU 同步和数据搬运；
- backbone 每 anchor 一次，候选 query 向量化；
- 报告 \(K\)、token 数、历史长度和相机数的 scaling curve；
- 与最接近基线在同硬件、同精度、同计时边界测量；
- 超预算时优先删除非必要模块或复用计算，不以改变科学对象换速度。

时延不通过则方法最多是离线分析器，不能主张可部署仲裁器。

## 9. Provenance、复现与独立审计

正式运行产物必须包含：

- code/config/model/input/manifest SHA-256；
- git revision 与 dirty-state；
- 完整软件和硬件环境；
- 数据角色与 sealed-access ledger；
- 每个 fold/arm/seed 的 job identity 与 GPU mapping；
- primary/replay 输出和 exactness 报告；
- finite-value、fold isolation、donor isolation 与统计重算；
- `experiments/registry.csv` 登记及人可读 `RESULTS.md`；
- claim–evidence matrix 更新和失败边界。

科学结论发布前必须完成机械 audit 和独立 experiment-integrity audit。外部 reviewer 不可用时明确标记
`PROVISIONAL_INTERNAL_REVIEW`，不得把内部复算描述为独立科学复核。

## 10. 失败处理合同

每个失败先分类：

1. **Implementation failure：** 公式与代码、数据绑定、数值或运行合同不一致；修复后重新资格化。
2. **Optimization failure：** 训练未达到预注册 sanity/收敛条件；只允许预先登记的诊断，不自动搜参。
3. **Hypothesis failure：** 实现有效且优化合格，但冻结科学判据不通过；保留 STOP/MIXED，重审理论。

修正必须形成新 ADR、新 commit、新 run id 和新 preregistration；旧裁决永不覆盖。只有当失败机制证明
现有归纳偏置缺失某项必要信息时，才允许提出新算法。新算法仍需回到第 2 节五问。

## 11. 主张纪律

每条论文句子必须能映射到：数学定义、代码对象、实验文件和适用边界。统一使用以下状态：

- `UNTESTED`：尚无正式实验；
- `SUPPORTED_WITHIN_SCOPE`：在明确 estimand/split/setting 内支持；
- `MIXED`：部分判据或子群支持；
- `UNSUPPORTED`：有效实验不支持；
- `INVALID`：实现、数据或审计资格失败。

禁止的外推包括：

- feature 稳定 \(\Rightarrow\) 决策充分；
- relation MSE 下降 \(\Rightarrow\) 驾驶收益；
- `U_ctx` 提升 \(\Rightarrow\) 完整 pseudo-closed-loop value；
- source calibration \(\Rightarrow\) 任意 OOD 保证；
- fallback/abstention \(\Rightarrow\) 安全；
- fixed-panel selection gain \(\Rightarrow\) candidate supply 已解决。

## 12. 分级 Definition of Done

### Gate I — `IMPLEMENTATION_READY`

- 数学对象与接口逐项映射；
- 单元/合同/端到端/反例测试通过；
- provenance、finite checks、fail-closed 与 deterministic replay 通过。

### Gate II — `SCIENTIFIC_RUN_VALID`

- 预注册、角色隔离、基线公平性和统计单位通过；
- primary/replay/audit 闭合；
- 没有结果驱动修改。

### Gate III — `METHOD_SUPPORTED`

- visual increment 成立；
- Differential Query 胜过 matched closest scorers；
- 关键消融、困难子集、三 seeds 和未见 source groups 支持。

### Gate IV — `PAPER_READY`

- NavDream/目标外观零样本、跨 planner、clean no-harm、sequential/actual-history 支持；
- 完整 evaluator 行为收益成立；
- end-to-end latency、memory 与公开复现材料闭合；
- claim audit 无越界。

### Gate V — `BEST_PAPER_ELIGIBLE`

除 `PAPER_READY` 外，还必须同时满足：

1. 发现并可靠复现“全局表示稳定不等于决策充分”的重要 gap；
2. differential evidence 在最强 matched baseline 下呈现不可由容量解释的机制优势；
3. 该优势跨外观、跨 planner 转化为显著行为收益，并揭示清晰适用边界；
4. 故事能用一个问题、一个核心对象、一个主要算法和一条闭环证据链讲清；
5. 结果对领域具有可迁移 insight，而非只针对 AORTA/NAVSIM 的工程技巧。

Gate V 是投稿自审资格，不是奖项保证。若发现不支持其中任一核心命题，必须诚实收缩论文定位。

## 13. 每次方法变更的强制检查表

任何进入主分支的方法 PR/commit 必须在关联文档回答：

```text
[ ] 触发它的真实失败/反例是什么？
[ ] 数学对象、假设和适用边界是什么？
[ ] tensor shape / unit / frame / time / mask 是否显式？
[ ] 最简单替代与最强 closest baseline 是什么？
[ ] 参数、FLOPs、数据、优化和预算是否匹配？
[ ] 哪项消融可以否定该模块？
[ ] 有哪些单元、反例、端到端和 replay 测试？
[ ] 正式运行是否预注册并写入 registry？
[ ] 时延、显存与 K-scaling 是否测量？
[ ] 结果将改变哪条 claim；失败时删掉哪条 claim？
```

任一项不能回答时，该实现可以保留为探索性分支，但不得成为 canonical 方法或论文贡献。

## 14. 对当前 R005E2B 的应用

R005E2B 已在提交 `0fdee4b464865d76f2efbc62955baa7e8f7ca8f5` 和冻结合同下运行，因此：

- 不因本标准修改其代码、arms、loss、seeds、threshold、GPU roster 或裁决；
- 它只回答正确冻结视觉对 `U_ctx` relation 是否存在可学习条件增量；
- 完成后检查 exact replay、角色封存、fold/donor isolation、有限值、公平 arm 与组级统计；
- 即使 GO，也不等于 Differential Query、外观鲁棒、完整行为收益或 Best Paper 资格；
- 即使 STOP，也先区分实现、优化和假设失败，不自动增加复杂模块。

本标准从 R005E2B 的结果审计和所有后续新工作开始强制执行。
