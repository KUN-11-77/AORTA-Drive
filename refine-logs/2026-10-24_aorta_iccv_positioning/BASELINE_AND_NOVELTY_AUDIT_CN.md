# AORTA 面向 ICCV 的基线与创新性重审

日期：2026-10-24  
状态：`POSITIONING_REVISED__METHOD_NOVELTY_NOT_YET_QUALIFIED`

## 1. 结论

FlowDrive 与 READ 是 AORTA 必须正面回应的最近邻方法，但“超越”不能被定义为跨协议击败论文中的单个 headline score。AORTA 必须在统一 backbone、候选、数据角色和评价接口下，证明其在以下主张轴上具有不可替代优势：

1. 对几何保持的 appearance shift，保持候选之间的任务排序；
2. 不更新冻结 planner，不使用目标域标签或目标域适配；
3. 干预具备显式 abstention，并控制 clean harm；
4. 相比完整 BEV/risk/energy field，所学习的量更接近真正决定候选比较的最小信息。

当前 QuotientFieldPredictor 只是一种待检验的结构化候选。图上的 gauge invariance、cycle consistency 和 potential parameterization 本身不构成 ICCV 级原创性。

PEA 和 MG-Select 可以进入基线体系，但必须按复现资格分层：

- PEA：可作为更宽松目标流访问条件下的 test-time adaptation 基线。本项目已有独立确认负结果，可正式报告为“PEA adapted to DiffusionDrive”，不能泛化成“PEA 方法无效”。
- MG-Select：AutoVLA 结果只属于 test-time core adaptation；ReCogDrive 结果属于 MG-style diffusion adaptation。两者均不是包含 joint condition-dropout training 的严格原论文复现，不能直接标成 official MG-Select baseline。

## 2. 最近邻方法的正确位置

| 方法 | 已解决的问题 | 对 AORTA 的直接挑战 | 可否直接量化比较 | AORTA 必须证明的剩余差异 |
|---|---|---|---|---|
| FlowDrive | 解析 risk/lane energy fields、anchor refinement、diffusion trajectory generation | potential/energy field 与场引导规划不是新概念 | 当前公开仓库未释放完整模型与训练评测代码；只能先做论文公式重实现 | appearance-shift 下的候选比较稳定性、冻结 planner 仲裁、abstention 与 clean-harm 控制 |
| READ | 从地图、动态 agents、专家行为与运动残差学习连续时空 risk field；可联合训练，也可修正冻结外部 planner 轨迹 | learned field、trajectory query、frozen-planner post-processing 都已被覆盖 | 未发现公开实现入口；应做容量与接口匹配的 READ-style 重实现 | 不重建完整风险面，直接控制候选决策边界穿越，并证明这一目标带来更强 OOD 排序可靠性 |
| PEA | source statistics + test-time embedding alignment | “对齐特征即可恢复鲁棒性”是强竞争解释 | 可以；本地 driving adaptation 已有审计结果 | AORTA 必须表明任务关系保持优于全局统计对齐，且不依赖目标流更新 |
| MG-Select | 条件掩码参考分布驱动的 verifier-free Best-of-N 选择 | “无需外部几何，只用模型内部置信度即可选候选”是强竞争解释 | 当前只能报告 core adaptation / MG-style；完整原方法需 joint condition-dropout training | AORTA 必须证明外部任务关系证据带来超过 likelihood、uniform-KL、mask-KL 的增量 |

FlowDrive 和 READ 的论文数值只用于 contextual comparison。若传感器、backbone、训练数据、NAVSIM 版本或 scorer 不一致，不得把 headline score 直接写成 AORTA 的公平胜负。

## 3. 重新冻结问题锚点

### Bottom-line problem

在同一物理道路场景与驾驶意图下，纯外观变化会使冻结 planner 的候选排序发生错误翻转。我们希望在不修改 planner、不过度重建完整世界状态、不使用目标域标签的条件下，恢复足以可靠比较候选行为的视觉任务信息。

### Non-goals

- 不声称静态道路关系足以表示完整驾驶安全；
- 不声称 source calibration 给出任意 appearance OOD 保证；
- 不把 DINOv3、potential field、graph gradient、Hodge decomposition、conformal calibration 或 min-plus solver 单独写成原创理论；
- 不通过增加完整世界模型、动态预测器和新 planner 来扩大论文贡献面。

### Success condition

在未见 recording groups、冻结候选与冻结 planner 上，正确图像产生的关系表示必须同时满足：任务正确、跨外观稳定、优于无图像/错配图像先验，并比 field MAE、全局特征漂移、SDF、READ-style field、FlowDrive-style field 和内部置信度更好地预测并减少候选排序翻转与 decision regret。

## 4. 推荐的单一核心贡献：decision quotient，而非 graph quotient

令场或关系变量为 \(z\in\mathbb R^d\)，冻结候选集合为 \(\mathcal C\)。候选 \(a,b\) 的任务差异写为

\[
m_{ab}(z)=Q(\tau_a,z)-Q(\tau_b,z).
\]

当候选查询在局部可写成线性或一阶近似时，存在行算子 \(A_{ab}\) 使

\[
m_{ab}(z)\approx A_{ab}z+c_{ab}.
\]

对候选族堆叠得到 \(A_{\mathcal C}\)。定义决策等价关系

\[
z\sim_{\mathcal C}z'
\quad\Longleftrightarrow\quad
A_{\mathcal C}(z-z')=0.
\]

因此候选排序真正依赖的是商空间

\[
\mathcal Z/\ker A_{\mathcal C},
\]

而不是完整场，也不只是去除常数 gauge 的图商空间。这个定义把“哪些误差不影响决策”直接绑定到冻结候选比较算子。

### 条件性命题 1：排序保持

若真实 pair margin 为 \(m^*_{ab}\)，且

\[
|A_{ab}(\hat z-z^*)|<|m^*_{ab}|,
\]

则该 pair 的排序符号保持不变。这个条件将 field error 改写为 decision-boundary crossing 条件。

### 条件性命题 2：选择性干预

若可构造有效的 pairwise error bound \(\epsilon_{ab}\)，并且只在

\[
\widehat m_{ab}-\epsilon_{ab}>0
\]

时用 challenger 替换 native，则在 bound event 上，所覆盖的道路约束因子不会发生被证书允许的反向排序。

这些命题不保证碰撞、动态交互、交通灯或闭环安全；它们只约束 teacher 和查询算子所定义的道路任务因子。

## 5. 最小系统架构

```text
RGB histories
   -> frozen DINOv3 spatial tokens
   -> small relation head
   -> scene-conditioned relation z_hat
   -> frozen candidate query A_C
   -> pairwise margins + calibrated error bound
   -> selective arbiter: keep native / choose challenger / abstain
```

### 复杂度预算

- 冻结：DINOv3、原驾驶模型、candidate provider、NAVSIM scorer；
- 新训练组件：一个 relation head；最多再增加一个显式 uncertainty head；
- 不新增：完整 diffusion planner、语言推理模块、RL verifier、开放式几何证书重构；
- graph-potential/integrability 只作为 relation head 的可删除归纳偏置，不预设为核心创新。

### 训练目标

主目标应从全场 edge MAE 转为候选比较相关的 contrast loss：

\[
\mathcal L_{dc}
=\sum_{(a,b)}\omega_{ab}
\rho\!\left(A_{ab}(\hat z-z^*)\right),
\]

并在 source-derived appearance pairs 上约束

\[
\mathcal L_{app}
=\sum_{(a,b)}
\left|A_{ab}\hat z(I)-A_{ab}\hat z(T(I))\right|.
\]

普通 field/edge supervision 可保留为辅助正则，但不能继续作为论文的唯一主目标。

## 6. ICCV 所需的基线层级

### A. 最近邻方法基线

1. FlowDrive-style analytic risk + lane field；
2. READ-style learned continuous risk field；
3. UncAD-style uncertainty-aware map/trajectory selection（若接口可公平实现）。

公开代码不可用时，必须写成 paper-derived matched reimplementation，并列出所有无法复现的细节；不能使用原方法名称暗示官方复现。

### B. 表示必要性基线

1. DINOv3 + ordinary SDF/BEV head；
2. direct edge prediction；
3. direct edge + exact gradient projection；
4. node potential / QuotientFieldPredictor；
5. direct candidate-pair score head；
6. decision-quotient relation head。

这些模型必须匹配 backbone、训练数据、参数/FLOP 预算、checkpoint selection 和 query/readout。若 direct pair head 持平或更好，应删除完整 field 作为主创新。

### C. 鲁棒与候选选择基线

1. Native / no intervention；
2. likelihood / planner-native score；
3. zero、source-only prior、wrong-image、removed-image；
4. PEA adapted to driving backbone；
5. Uniform-KL；
6. MG-Select test-time core adaptation 或 MG-style adaptation；
7. Oracle candidate upper bound（只作 headroom，不是部署方法）。

## 7. PEA 与 MG-Select 负结果的论文资格

### PEA

项目中的冻结 PEA-MG V1 confirmation 覆盖 76 tokens、38 logs、15 sessions、11 appearances。保存结果的机械重算通过；PEA-MG 相对 Native 为负，session-cluster CI 不跨 0，并出现 new-hard。它可以成为正式的 driving-adapted TTA baseline，但必须同时说明：

- 这是 PEA 向 DiffusionDrive 视觉主干的迁移，不是原论文分类 benchmark；
- 负结论只适用于该冻结适配与协议；
- 它支持“全局 embedding alignment 可能破坏 task directions”，不证明所有 TTA 无效。

### MG-Select

- AutoVLA：test-time core adaptation，缺少 joint condition-dropout training，且 MG-Both 与 Uniform-KL 近似等价；可作为 appendix 或 matched selector baseline；
- ReCogDrive：Gaussian reverse-kernel 上的 MG-style 适配，不是原论文 action-token VLA；适合机制对照，不适合写成严格复现；
- 若要在主表使用“MG-Select”而无限定词，必须实现兼容模型上的 joint condition-dropout training，并按原论文训练与选择合同复核。

失败基线只有在实现、候选支持、评分和统计均闭合时才有价值。任何由接口不匹配、候选坍缩或未实现原训练策略导致的失败，只能记为 inconclusive adaptation。

## 8. R004H6 之后的唯一合理顺序

1. **R004H6：held-out 视觉资格。** 冻结执行，不因本审查改变协议。
2. **R004H7-A：理论等价与文献边界。** 正式证明 potential 与 projected-edge 的表示等价；冻结 decision quotient 定义、反例和 claim boundary。
3. **R004H7-B：只读 shadow decision diagnosis。** 在固定候选上比较 field MAE、DINO drift、SDF error、decision-contrast error 对 ranking flip/regret 的增量解释力。
4. **R004H7-C：条件启动公平训练。** 只有 H7-B 支持 decision contrast 后，才训练 direct edge、projected edge、potential、direct pair 和 decision-quotient heads。
5. **R004H8：冻结决策评估。** 比较 Native、PEA、likelihood、Uniform-KL、MG-style、FlowDrive-style、READ-style 与 AORTA；报告 coverage、abstention、clean harm、rank regret、DAC/DDC 与完整 PDMS 分项。

## 9. ICCV 裁决门

AORTA 只有同时满足以下条件，才有资格按方法论文推进：

1. R004H6 在未见 groups 上证明正确图像必要性、任务正确性和跨外观稳定性；
2. decision-contrast error 对真实 ranking flip/regret 的解释显著超过 field MAE 与全局 drift；
3. decision-quotient head 在公平预算下胜过 direct pair、projected edge、SDF/BEV、FlowDrive-style 与 READ-style；
4. 选择性仲裁在至少两个不同 planner/candidate providers 上产生正收益，并满足 clean non-degradation；
5. 增益不能由更大 backbone、更多候选、额外 target data 或不同 scorer 解释；
6. 所有正式 test 只打开一次，组/session 是统计单位，失败也完整报告。

如果第 2 或第 3 条失败，应把论文降级为视觉机制/评测发现，或者采用更简单的 direct pair method；不得继续围绕 quotient field 增加模块来保护既有设计。

## 10. 当前投稿判断

当前状态不是 ICCV-ready。研究问题有价值，R004H5 提供了 fit 机制证据，R004H6 正在检验 held-out 泛化；但核心算法的不可替代性、与 READ/FlowDrive 的机制差异、以及决策收益均未成立。

最有希望的论文主线不是“新的风险/势场”，而是：

> 对冻结驾驶系统，外观鲁棒性应在候选决策对比的商空间中学习和校准；完整场中大量误差与候选排序无关，而真正需要控制的是外观误差对决策边界的穿越。

该主线仍是待证伪假设，不是当前已经成立的创新结论。

## 11. 核验来源

- FlowDrive, arXiv:2509.14303，v1，2025-09-17；作者项目页与仓库截至本次核验仍标注完整代码/模型待发布。
- READ, arXiv:2609.12371，v1，2026-09-11；原文包含 matched E2E/VLA field learning 与 frozen external-planner trajectory refinement。
- PEA, ICLR 2026 OpenReview `7kLNGaAHaw`；本地 driving adaptation 与 confirmation 审计见历史实验产物。
- MG-Select, ICLR 2026 OpenReview `UD4Rw8MOEK` / arXiv:2510.05681；本地结果按 core adaptation 与 MG-style adaptation 分开解释。

