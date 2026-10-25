# AORTA 面向 ICCV/CVPR 的投稿级研究蓝图 V1

状态：`THEORY_REVIEWED__METHOD_HYPOTHESIS_FROZEN_FOR_FALSIFICATION__FORMAL_RUNS_HOLD`
范围：只读审查、理论推导和数据无关验证；B4R1、训练与封存 split 均未授权。

## A. Executive Research Thesis

### A.1 事实与假设账本

**已核验事实**

1. R004H6R1 支持旧视觉场在 historically exposed held-out development 上含非平凡视觉任务信息；
   它不证明新 relation 网络或驾驶收益。
2. B3 证明 scalar set-min 被共同路径支配；BS2 证明 2.5 s prefix 丢失晚期分叉；旧 scalar
   road-margin reranker 已停止。
3. B4 因合法候选 membership 类型错误永久 INVALID；B4R1 只完成 outcome-free readiness，正式运行
   仍为 HOLD。
4. D-JEPA、World4Scorer、GTRS/DA-WAM 类工作已经覆盖一般 relation/scorer、candidate
   conditioning、outcome supervision、history reranking 或 fallback 的重要部分。
5. 当前没有新 relation 模型的训练结果、NavDream 结果、三 planner 行为收益、clean no-harm 或
   batch-1 latency 结果。

**已证明的条件命题**

- 合法 appearance intervention 下真实任务 relation 不变；
- 表示可恢复目标当且仅当目标在表示纤维上为常数；
- signed support 对加性轨迹泛函给出精确差分；
- native-star 输出经差分可精确满足反对称与循环一致性；
- 同时 margin 误差 \(\varepsilon\) 推出选择 regret 不超过 \(2\varepsilon\)；
- simultaneous coverage event 上，正 lower bound 推出相对 native 的真实改善。

**仍未证明的假设**

- differential support 对非加性 official utility 比 generic scorer 更有效；
- source 外观轨道约束能迁移到 NavDream；
- correct visual evidence 在新 relation 模型中不可被轨迹先验替代；
- history-conditioned one-step 模型在实际 rollout history 下仍有效；
- selective rule 不会近乎全拒判；
-完整在线链路满足实时预算。

### A.2 三个候选科学问题

| 候选问题 | 重要性 | 可识别性 | 创新空间 | 裁决 |
|---|---:|---:|---:|---|
| Q-A：如何学习外观稳定道路场 | 中 | 高 | 低；现有工作密集，且 B3 否定一般 reranker 充分性 | 不选；保留 baseline |
| Q-B：如何在冻结候选集上学习更强 outcome scorer | 高 | 高 | 中低；D-JEPA/World4Scorer 已非常接近 | 必须做 strongest baseline |
| Q-C：什么视觉证据在外观变化下足以保持候选间的正确决策差异 | 高 | 中 | 窄但可辩护 | **主问题** |

选择 Q-C，因为它由项目真实反例推出，能够同时产生数学可识别性问题、可区分网络归纳偏置和明确
falsifier。Q-A 解决的是被证伪为不充分的中间对象；Q-B 虽实用，但若不提出新的 evidence
interface，难以与最近工作形成独立贡献。

### A.3 核心 insight

稳定整个表示并非必要且不充分。对候选仲裁真正需要稳定的是：**native 与 challenger 在完整时域
哪里产生不同后果，以及全局视觉中哪些证据解释这些差异。**

### A.4 一句话方法

AORTA 用完整时空 signed candidate difference 生成视觉 query，以共享全局视觉 tokens 和实际
previous-action context 回答 query，输出全局一致的 native-relative contextual utility，并只在
同时下界为正时干预冻结 planner。

## B. Scientific Problem Formulation

### B.1 统一符号

| 符号 | 含义 |
|---|---|
| \(S_t\) | 物理场景与任务状态 |
| \(A_t\) | appearance nuisance |
| \(X_t=g(S_t,A_t,\xi_t)\) | 多相机视觉观测 |
| \(Z_t=E_{vis}(X_t)\) | 共享视觉 token |
| \(V_t=\{\tau_0,\ldots,\tau_K\}\) | 合法 ragged 候选域 |
| \(h_t=\tau^{sel}_{t-1}\) | 上一周期实际选择 |
| \(U_t^{ctx}(\tau;S,h)\) | 作用域核验后的两帧候选效用 |
| \(\Delta^*_{ij}=U^{ctx}(\tau_i)-U^{ctx}(\tau_j)\) | 真实相对 margin |
| \(\nu_{ij}=\mu_i-\mu_j\) | 完整时空 signed support |
| \(q_{i0}\) | 预测 native-relative margin |
| \(r_{i0}\) | calibration 后的同时误差半径 |
| \(\mathcal R\) | 决策风险或 regret |

### B.2 目标之间的逻辑关系

下列蕴含成立：

\[
\|q-\Delta^*\|_\infty\le\varepsilon
\Longrightarrow \operatorname{Regret}\le2\varepsilon.
\]

\[
\mathcal E\land(q_i-r_i>0)
\Longrightarrow \Delta^*_{i0}>0.
\]

下列蕴含不成立：

1. appearance invariance \(\not\Rightarrow\) decision correctness。反例：\(q(X)\equiv0\) 完全稳定但在
   任一非零 margin 上错误。
2. low average MAE \(\not\Rightarrow\) correct top choice。反例：只在两个近最优候选上交换很小误差，
   平均 MAE 可趋近零但选择翻转。
3. correct one-step ranking \(\not\Rightarrow\) sequential benefit。反例：第一次错误干预改变 \(h_{t+1}\)，
   后续输入离开 teacher-forced 分布。
4. better candidate selection \(\not\Rightarrow\) closed-loop safety。候选集合可能缺失安全轨迹，官方
   proxy 也不涵盖真实交通的全部风险。

### B.3 决策风险

训练阶段的核心风险写为

\[
\mathcal R_{ctx}(\theta)
=\mathbb E\left[
\Delta^*_{i^*0}-\Delta^*_{\hat i0}
+\lambda_h\mathbf 1\{\hat i\ne0,\Delta^*_{\hat i0}\le0\}
\right],
\]

其中 \(i^*=\arg\max_i\Delta^*_{i0}\)，\(\hat i\) 是模型与仲裁规则共同选择。最终系统风险必须另在
official evaluator 与 sequential protocol 中测量，不把 \(\mathcal R_{ctx}\) 宣称为安全等价物。

## C. Formal Theoretical Foundation

每条结果的完整证明与反例入口为
`AORTA_SUBMISSION_THEORY_V1_CN.md` 与 `FORMAL_THEORY_APPENDIX_V1_CN.md`。本节给出论文正文级结构。

### C.1 Legal-orbit invariance

- **Definition:** \(T_a\) 保持 \(S,h,V,U^{ctx}\)。
- **Assumptions:** 纯外观变化，不改变 evaluator label。
- **Proposition:** \(\Delta^*(T_aX)=\Delta^*(X)\)。
- **Proof:** 两个被减效用的全部自变量不变。
- **Counterexample:** 雨雪编辑删除车辆或改变可见交通灯状态。
- **Architecture implication:** 只约束 relation orbit，不无条件对齐全部 feature。
- **Falsifier:** 变换前后官方标签或候选 membership 改变。
- **Originality boundary:** 任务不变性是定义一致性，不是新定理。

### C.2 Representation identifiability

- **Definition:** 表示纤维 \(R^{-1}(r)\)。
- **Proposition:** 存在 \(f\) 使 \(\Delta^*=f\circ R\) 当且仅当 \(\Delta^*\) 在每个纤维上为常数。
- **Proof:** 必要性来自函数复合；充分性通过对每个非空纤维定义唯一值。
- **Counterexample:** 两个视觉表示相同的场景中，遮挡后交通灯状态相反，真实 margin 符号相反。
- **Architecture implication:** global KV 与 correct/mismatch/removed control；真正不可识别时拒判。
- **Falsifier:** correct image 不优于 mismatch/removed。
- **Originality boundary:** 标准 factorization 条件；AORTA 价值在其驾驶反例与接口选择。

### C.3 Differential support

- **Proposition:** 加性泛函对 signed measure 精确线性。
- **Counterexample:** collision gate、min/TTC、交通规则与 history comfort 均可能非加性；只看
  \(\nu\) 不够。
- **Architecture implication:** \(\nu^+,\nu^-\) 用于 query；common support、global visual evidence
  与 \(h_t\) 仍保留。
- **Falsifier:** matched generic scorer 相当或更优；去掉 differential support 不降低 OOD regret。
- **Originality boundary:** Jordan decomposition 是标准数学；新意只能来自外观干预下的 evidence
  interface 及其实证增量。

### C.4 Global ranking consistency

- **Definition:** 部署只预测 \(q_0=0,q_i=q_{i0}\)，任意 pair 定义为 \(q_i-q_j\)。
- **Proposition:** 反对称、零自比较和循环和为零精确成立。
- **Proof:** 代数消去。
- **Counterexample:** 独立 pair head 可产生 \(\hat\Delta_{12}>0,\hat\Delta_{23}>0,\hat\Delta_{31}>0\)。
- **Architecture implication:** 不独立预测 complete pair graph；若使用 pair supervision，统一投影到
  同一 native-star potential。
- **Falsifier:** unrestricted pair head 在相同预算下更准且其循环不影响选择；此时一致性只保留类型安全。
- **Originality boundary:** star exactness 是重参数化，不作为结构创新。

### C.5 Error, regret and selective intervention

- **Proposition:** simultaneous \(\ell_\infty\) margin error \(\varepsilon\) 推出 regret \(\le2\varepsilon\)。
- **Proposition:** simultaneous interval event 上，正 lower bound 推出真实改善。
- **Counterexample:** 独立 marginal intervals 对选后最大候选不提供 simultaneous 保护；source
  calibration 在 NavDream 失去 exchangeability。
- **Architecture implication:** group-max calibration 与 native fallback。
- **Falsifier:** undercoverage、近乎全拒判、false intervention 不降。
- **Originality boundary:** regret bound 与 split conformal 均属标准工具。

### C.6 History rollout shift

- **Proposition:** 对 \([0,1]\) 有界损失，teacher-forced 与 rollout 风险差不超过两分布的总变差。
- **Counterexample:** 一次 AORTA 替换导致下一帧 \(h\) 未在训练出现，后续 margin 全翻转。
- **Architecture implication:** actual history token、native-history/actual-history/sequential 三臂。
- **Falsifier:** sequential regret 明显高于 teacher-forced，且不能由 candidate supply 解释。
- **Originality boundary:** distribution-shift bound 标准；其在 AORTA official history 中的闭环是任务要求。

## D. Neural Network Theory and Architecture

### D.1 三种候选架构

| 路线 | 输出 | 优点 | 致命风险 | 角色 |
|---|---|---|---|---|
| A. Absolute scorer | 每候选 \(\hat U_i\)，再作差 | 最简单、天然一致、强 baseline | 外观公共偏差未必抵消；每候选查询可能忽略 pair difference | 必做 baseline |
| B. Full-context set scorer | 集合内联合评分 | 可表达候选交互；最接近 D-JEPA | 可能自己学出差异，使 AORTA 新意消失；候选集合改变可影响单候选 score | strongest baseline |
| C. Differential evidence potential | signed pair query + global KV，输出 native-star \(q_i\) | 对准失败反例；一次视觉编码；一致输出 | inductive bias 可能无增益；query 构建有成本 | **推荐主方法** |
| D. Unrestricted pair head | 独立 \(\Delta_{ij}\) | 表达能力最大 | 循环偏好、\(O(K^2)\)、样本低效 | 删除或仅诊断 |

选择 C 不是因为它最复杂，而是它是唯一同时响应 common-min、late-fork、global context、official
history 和实时约束的最小机制。若 C 不胜 B，论文不能保留架构创新。

### D.2 Tensor contract

令 batch \(B\)、合法候选上限 \(K\)、视觉 token 数 \(P\)、每个差异分支 query 数 \(M\)、宽度 \(d\)：

- visual tokens：`Z [B, P, d]`，mask `[B, P]`；
- trajectories：`T [B, K+1, L, s]`，含 \(x,y,yaw,v,t\) 与 footprint；
- ragged candidate mask：`C [B, K+1]`；
- signed queries：`Q_pos/Q_neg/Q_common [B, K, M, d]`；
- actual-history token：`H [B, H_len, d]`，首帧另有 validity bit；
- evidence summaries：`E_pos/E_neg/E_common [B, K, d]`；
- margins/scales：`q/sigma [B, K]`，native \(q_0=0\)；
- calibrated lower bounds：`L [B, K]`。

### D.3 参数与 FLOP 估算

\[
\#\theta_{new}
=\#Q_{geo}+\#W_Q+\#W_K+\#W_V+\#G+\#H_{hist}+\#H_{scale}.
\]

若每个分支使用 cross-attention，粗略 query FLOPs 为

\[
O(3BKMPd)+O(BKMd^2),
\]

视觉 backbone 只计算一次 \(C_{vis}\)。正式 matched study 要固定 \(d,M,P\)、层数和 token budget；
参数匹配不足以替代 FLOP 与 memory matching。

### D.4 视觉 backbone 决策

DINOv3 作为冻结 evidence encoder 是合理起点，因为旧实验已有 visual necessity 信号，冻结还能避免
目标域适配混淆。但它不保证交通灯、动态体或长期任务证据完整。默认只训练小型 projector/query/head；
只有 correct-image 或动态场景分层失败明确指向 backbone 信息缺失时，才允许注册 source-only adapter。

### D.5 离散化

query 使用连续轨迹与车辆 footprint 先生成软时空 samples，再做固定数目的带权稀疏 tokens，而不是
把 64×64/96×64 网格当作方法本体。需要报告 \(M\)、时间采样、空间核宽和 quadrature refinement
误差；若两级细化改变 relation 超过预注册 tolerance，查询离散化不合格。

## E. Learning Objective and Loss Derivation

### E.1 最小损失体系

\[
\mathcal L_{val}
=K^{-1}\sum_i\operatorname{Huber}(q_i-\Delta_i^*),
\]

\[
\mathcal L_{dec}
=\max_i[q_i-q_{i^*}+\Delta_{i^*}^*-\Delta_i^*]_+,
\]

\[
\mathcal L_{orb}
=\max_a\|q(X^a)-\operatorname{sg}q(X^{clean})\|_\infty.
\]

优化为

\[
\min_\theta\mathbb E[\mathcal L_{val}+\lambda_{dec}\mathcal L_{dec}]
\quad\text{s.t.}\quad
\mathbb E[\mathcal L_{orb}]\le\delta_A.
\]

point model 冻结后，尺度分支使用 proper residual likelihood/interval score，之后再做独立 calibration。

### E.2 梯度与冲突

- Huber 在大 residual 上限制梯度，避免少数 scorer outlier 支配，但可能弱化 top-choice 边界；
  \(\mathcal L_{dec}\) 只补足当前最危险的结构化违反。
- \(\mathcal L_{dec}\) 在 tie/近 tie 处可能非光滑；使用固定 tie tolerance 与 deterministic subgradient，
  不按结果改 margin。
- orbit loss 可能把错误 clean prediction 传播到所有 appearance；stop-gradient 只稳定 anchor，
  不能修正错误，所以 task loss 必须在 clean 与合法 transformed input 上都计算。
- value 与 ranking 梯度可能冲突；预注册报告 cosine/conflict rate。若 \(\mathcal L_{dec}\) 不降低
  regret，删除它而非调更多权重。
- learned scale 可通过放大 \(\sigma\) 降低 normalized residual；proper score 的 log-scale 项和独立
  calibration 防止无代价膨胀，但不能保证非真空 sharpness。

### E.3 捷径与反例

- candidate ID/source token 不进入输入；
- mismatch 不默认作为训练负样本，避免学习跨场景检测捷径；
- trajectory-only、removed-image 与 candidate-order permutation 是强制检查；
- orbit-only、ranking-only、value-only 和完整目标均需删除消融。

## F. System and Deployment Design

数据流：

`RGB/history → shared frozen visual encoder → legal candidate provider → full-horizon signed query
→ global evidence transport → history-conditioned native-star margins/scales → calibrated lower bounds
→ native-preserving arbitration → selected trajectory/history update`。

跨 planner 只适配 candidate extraction、坐标/时间标准化和合法 mask；视觉 encoder、query operator、
relation head 原则上共享。三类 planner 分别裁决，禁止 pooled 平均掩盖失败。

无合法 history 时设置显式 invalid flag，使用 evaluator 的 one-frame 语义；除非已核验的最坏符号
证书或 calibrated lower bound 成立，否则保持 native。

部署必须保存：candidate provenance、actual previous selection、query discretization version、model
hash、calibration hash、运行时和 fallback reason。

## G. Novelty and Prior-work Audit

| 工作族 | 已覆盖 | AORTA 不能宣称 | 必须额外证明 |
|---|---|---|---|
| D-JEPA | set relation、native correction、risk/margin gate、fallback | relation/fallback 首创 | appearance differential query 的 matched 增量 |
| World4Scorer | outcome-grounded candidate state、future/history reranking | outcome/full-context scorer 首创 | correct-image necessity 与 OOD relation stability |
| GTRS/DA-WAM | trajectory token、candidate-conditioned scoring | cross-attention/scorer 首创 | signed support 的独立价值 |
| FlowDrive/READ | task/risk field 与 planning/refinement | field-to-planning 首创 | field readout 不充分后的 relation 增量 |
| PEA/MG-style | test-time/global-statistic adaptation 或内部选择 | “适配失败所以 AORTA 必然成功” | 冻结公平 baseline；负结果按复现范围表述 |
| selective/conformal work | confidence gating、native fallback、coverage | uncertainty/fallback 首创 | relation estimand 与实证非真空性 |

现有本地审计已查阅上述最接近工作的公开原文或官方页面；FlowDrive 完整代码/模型当时未公开，READ
需 matched reimplementation。PEA driving adaptation 与 MG-style 结果只能按已实现范围命名，不能
冒充原论文完整官方复现。提交前必须再次核对版本、公开实现和 bibliographic status。

## H. Claim–Evidence–Falsifier Matrix

| Claim | Hypothesis | Test | Falsifier | Decision |
|---|---|---|---|---|
| scoped target 合法 | evaluator 层级与公式一致 | source primary/replay/audit | 任一门控/聚合不匹配 | 收缩 estimand |
| contextual oracle 非平凡 | full context 含可改善候选 | group oracle | 无 headroom | 停当前 panel |
| visual evidence 必要 | correct image 提供额外信息 | mismatch/removed/trajectory-only | correct 无优势 | 停视觉主张 |
| differential bias 有用 | C 优于 A/B | matched 3-seed held-out | strongest scorer 相当/更好 | 删除网络创新 |
| full horizon 必要 | late fork 影响 relation | full vs prefix | 无增量 | 删 full-query复杂度 |
| signed branches 必要 | 正负证据早减会损失 | separated vs early subtraction | 相当 | 合并分支 |
| history一致 | actual history 改变 target | native/actual/sequential | rollout 崩溃 | history修复或停 |
| appearance 泛化 | source orbit 学到可迁移 invariance | sealed NavDream | OOD regret/flip 崩溃 | 停鲁棒性主张 |
| 行为增益 | selection 改善 official behavior | 三 planner clean/OOD | \(G_{net}\) 无正下界/new-hard | 停系统主张 |
| selective可靠 | interval 非真空且少误替换 | independent calibration | undercoverage/全拒判 | 删除可信机制 |
| 实时 | query 增量可部署 | batch-1 profile | P95 超预算 | 降级离线 |

## I. Publication-Ready Paper Story

### I.1 标题候选

**AORTA: Appearance-Robust Trajectory Arbitration via Differential Decision Evidence**

只有 matched、NavDream 与行为门通过后保留 Appearance-Robust、Differential 和 Arbitration 三个词。

### I.2 English abstract draft

Appearance shifts can change the decisions of end-to-end driving planners even when their visual
representations remain stable. We argue that robustness should therefore be defined over the
task relation between candidate trajectories, rather than over a global feature or road field.
We introduce AORTA, a plug-in arbitration framework for frozen planners. AORTA represents each
native–challenger pair by its signed, full-horizon spatiotemporal occupancy difference, uses this
difference to query a shared global visual representation, and predicts a history-conditioned
native-relative task margin. Its native-star output is antisymmetric and globally consistent, while
a group-calibrated simultaneous lower bound permits intervention only when the evidence supports
an improvement. The design is motivated by two audited failure modes of field-based arbitration:
common-path minima erase candidate differences, and short temporal prefixes miss late divergence.
We evaluate whether this differential-evidence bias provides value beyond capacity- and
compute-matched relational scorers through correct/mismatched-image controls, unseen appearance
shifts, three planner families, clean non-inferiority, and batch-one latency. Theoretical guarantees
are explicitly conditional on the scoped evaluator and calibration population; they do not imply
arbitrary-OOD or real-world safety.

该摘要是设计稿，结果句必须在实验后填写，当前不得投稿使用为已完成摘要。

### I.3 Contributions

1. 一个由 common-min 与 late-fork 反例推出的 appearance-interventional differential-evidence
   estimand/interface；
2. 一个共享视觉编码、完整时域 query、history-conditioned 且全局一致的 native-relative operator；
3. 一套把 relation error、sequential history、选择性干预、跨 planner 行为和实时性闭合的验证合同。

第 2 条必须胜过 matched scorer；第 3 条若只是严谨实验协议，可放在系统贡献而非算法首创。

### I.4 三张主图

1. **Paradox:** stable/perfect field + common-min/late-fork 仍产生错误排序。
2. **Operator:** one visual encoding → signed full-horizon queries → global KV → integrable star margins
   → positive lower-bound intervention → history update。
3. **Causal evidence:** correct-image necessity、matched-scorer regret、NavDream flip、sequential
   false intervention、\(G_{net}\)、clean harm 与 P95 的同一证据链。

### I.5 30 秒与 90 秒

**30 秒：** 稳定感知不等于稳定决策。AORTA 不再稳定整幅道路图，而用 native 与 challenger 的
完整时空差异去查询全局视觉证据，预测带真实历史上下文的相对任务收益；证据不足就保留原决策。
论文生死取决于它是否在 NavDream 和三类 planner 上胜过信息与算力匹配的通用 scorer。

**90 秒：** 外观变化改变 planner 决策；旧 field 在 perfect input 下仍因 common-min 和短前缀删除
候选差异；因此 estimand 改为 contextual native-relative utility；signed full-horizon support 只负责
提问，全局视觉回答非局部后果；native-star 输出保证排序一致，actual previous action 闭合两帧目标；
task risk 与 orbit constraint 同时保证正确和稳定；同时 lower bound 才允许替换；最后用 matched
D-JEPA/World4Scorer、correct/mismatch/removed、NavDream、三 planner、clean no-harm、sequential
history 和 P95 决定主张是否成立。

## J. Critical Review and Revision

### J.1 Round 1 P0 defects

1. 把局部 \(7/8+1/8\) 分解写成完整官方目标，作用域过强；
2. 独立 pair head 只有反对称，没有多候选循环一致性；
3. native-history 训练与 AORTA-history 推理存在 rollout shift；
4. B4R1 one-frame STOP 被错误解释为 full-context panel STOP；
5. differential support 仍可能只是 generic scorer 可自行学习的 feature engineering。

### J.2 已完成修订

- 将目标改名为 scoped \(U^{ctx}\)，完整 EPDMS 只作最终行为评估；
- 部署输出改为 native-star potential，任意 pair 由差分得到；
- 增加 actual-selected-history 与 sequential evaluation 合同；
- 将 B4R1 接口有效性和 one-frame 科学 outcome 分离；
- 把 matched full-context scorer 提升为单一 novelty falsifier。

### J.3 Round 2 审稿裁决

| 维度 | /10 | 结论 |
|---|---:|---|
| Problem fidelity | 9 | 仍聚焦 appearance-to-decision gap |
| Method specificity | 9 | estimand、tensor、loss、推理和停止规则清楚 |
| Contribution quality | 7 | 新颖性窄，必须靠 matched 增量 |
| Frontier leverage | 8 | 复用冻结 foundation vision，不强加趋势模块 |
| Feasibility | 7 | full-target materialization 与 query latency 未知 |
| Validation focus | 9 | falsifier 明确且没有 benchmark 堆叠 |
| Venue readiness | 6 | 核心实验为空 |
| **Overall** | **7.8** | **REVISE；理论可推进，论文不可投稿** |

### J.4 rapidPeer 13 问简答

1. 理论与问题匹配，但只支持条件主张。
2. 原创空间存在于 appearance differential evidence，不在 relation/fallback。
3. 问题有意义；最简单竞争解释是 matched scorer 已足够。
4. 当前最重要改进不是加模块，而是目标作用域与 sequential history。
5. 方法学框架正确；官方 evaluator 作用域仍须机械核验。
6. absolute subtraction 和 full-context set scorer 都可能更好，必须先实现。
7. 已主动承认不可识别、候选供给、OOD coverage 和闭环边界。
8. 目前仅有处理方案，没有经验闭环。
9. 最清楚表述：稳定感知并不保证知道哪条轨迹更好。
10. 正文只保留 \(\Delta^*,\nu,Z\) 三个核心对象，其余放附录。
11. 三图见 I.4。
12. 还需最新 closest-work 版本、官方 evaluator scope 与 sequential/offline 理论实证。
13. 若视觉证据机制成立，优先 ICCV/CVPR；若主要贡献落在仲裁/风险，CoRL/ICRA/T-IV 更合适。

## K. Open Problems and Research Decisions

### K.1 P0 未解决问题

1. \(U^{ctx}\) 与完整 EPDMS 的精确代码边界；
2. contextual full-target 是否具有非平凡 oracle headroom；
3. differential operator 是否超越 matched full-context scorer；
4. actual-selected-history 的训练覆盖与 sequential shift；
5. NavDream 下 source calibration 的失覆盖程度；
6. 完整 batch-1 latency。

### K.2 下一项最高信息价值工作

在不解除 B4R1 HOLD 的前提下，先完成**数据无关 evaluator-scope audit 预注册**：列出
\(U^{1f},H,C_2,U^{ctx}\)、乘性门、过滤、归一化和最终 EPDMS 的调用图与逐项 replay 条件。同时把
B4R1 解释改为“one-frame 子问题”，不再让其科学 STOP 阻断 contextual target 资格。

用户明确授权正式运行后：

1. 运行冻结 B4R1，仅裁决接口和 one-frame 信号；
2. 只要接口有效，独立 materialize scoped contextual target；
3. contextual oracle 有信号后，先实现 absolute scorer 与 matched full-context scorer；
4. 再训练 AORTA，单一问题是 differential query 是否提供增量；
5. 通过后才开封 NavDream、sequential history、三 planner 与时延。

### K.3 五项明确裁决

1. **THEORY STATUS：** 条件定理与反例链已闭合；evaluator scope、非加性充分性、OOD 与 rollout
   前提未证明。
2. **NOVELTY STATUS：** 存在窄而可辩护的算法假设；尚无证据证明独立创新成立。
3. **ARCHITECTURE STATUS：** 推荐架构由真实失败与理论自然导出；可积 native-star 和 history
   修正使接口比 v2 更严谨。
4. **RESEARCH READINESS：** 已具备进入“冻结目标核验与 matched 实验预注册”的科学条件；不具备
   跳过 B4R1/HOLD 或直接做 NavDream/行为实验的条件。
5. **PUBLICATION POTENTIAL：** 若 matched scorer、NavDream、三 planner、clean no-harm、sequential
   与 P95 全部通过，可形成 ICCV/CVPR poster 级强方法候选；oral 需要非常清晰且显著的跨 planner
   现象与简单统一机制。若 differential 增量失败，则降低为严谨评测/系统论文或转投机器人/智能车领域。
