# AORTA 最终实验计划：从理论对象到跨规划器行为

**状态：PLAN_ONLY__FORMAL_B4R1_HOLD__SEALED_SPLITS_UNTOUCHED**

**问题：**视觉外观变化下，如何为不同冻结驾驶规划器提供任务充分、可信且实时的决策辅助，而不是只获得稳定但不影响决策的表征。

**方法主命题：**共享视觉编码上的完整时空 native–challenger 差异查询，比容量和计算匹配的通用候选 scorer 更能保持外观干预下的 full-task relation，并减少错误替换。

**从属命题：**group-simultaneous lower-bound 仲裁在非真空 coverage 下减少有害干预；证据不足时保留 native。

**日期：**2026-10-25。

本文是条件实验路线，不授权任何正式运行。B4R1 仍需用户明确授权；calibration/test/NavDream/target 继续封存。

## 1. Claim Map

| Claim | 为什么重要 | 最低可信证据 | 对应 Block |
|---|---|---|---|
| P1：differential evidence 是外观鲁棒决策关系的有效且必要接口 | 这是相对 D-JEPA/World4Scorer-style generic scorer 剩余的主要创新候选 | 合法 full target；matched 数据/参数/FLOP/token/seed；held-out 与 NavDream 上更低 regret、pair-sign error、false intervention；support 删除实验复现机制 | B0、B1、B2 |
| S1：evidence-qualified arbitration 能在不全拒判时减少错误替换 | 这是从预测到可信动作的桥 | 独立 group calibration；simultaneous empirical coverage、sharpness、risk–coverage；OOD 失覆盖诚实报告；clean harm 降低 | B4 |
| P2：方法能改变真实评估行为且具有跨架构与实时价值 | 防止论文停留在表示或离线 scorer | NavDream/正式 OOD 上至少三类 planner；Official PDMS、DAC、碰撞、组级配对差异、hard-failure 净改善；clean 非劣；batch-1 P50/P95 | B3、B4 |

### 必须排除的竞争解释

1. 收益只来自更大模型、更多 token、更多训练或更强 DINO；
2. 收益来自 candidate provider 增加了更好候选，而不是选择器；
3. correct-image 与 mismatch/removed 相当，模型主要使用轨迹或数据集先验；
4. OOD 提升以 clean 退化为代价；
5. selective 方法只是近乎全拒判；
6. NavDream 或正式 target 被用于调 architecture、appearance、threshold 或 hard-case 定义；
7. 只在一种 planner 上成立，不能支持通用接口；
8. 在线成本实际是每个候选重复视觉编码。

## 2. Paper Storyline

### 主文必须证明

- 稳定道路场/短前缀为何不足：B3 common-min 与 BS2 late-fork 反例；
- full-task relation 的合法标签与非平凡 oracle opportunity；
- differential evidence 相对 strongest matched generic scorer 的增量；
- correct image、appearance invariance 和 task correctness 同时成立；
- NavDream + 三 planner 的行为净收益、clean no-harm 与实时性；
- selective arm 的 risk–coverage，而不是只给理论保证。

### 附录支持

- one-frame 的 \(1/7\) worst-case sign-certified subdomain；
- factor heads、复杂 temporal encoder、额外 backbone 的条件消融；
- appearance severity、候选数、速度、曲率和困难类型分层；
- calibration 次序统计量、有限样本资格与全部失败 strata；
- 完整 latency trace、显存和预处理明细。

### 明确删除

- 不再训练旧 scalar road-margin reranker；
- 不恢复开放式几何证书工程；
- 不把 Hodge、conformal、antisymmetry、DINO 或 cross-attention 单列贡献；
- 不做没有改变审稿人判断的长 baseline 清单；
- 不用 target-domain TTA 挽救 source-only 主张；
- 不以更大 ROI、更多候选或完整 planner 重训替代核心 relation 问题。

## 3. 五个核心实验块

## Block B0：Estimand 与 oracle 资格

- **Claim tested：**合法候选域上存在非平凡任务监督；最终 full target 可复算且必须包含 previous-action context。
- **为什么存在：**没有合法且有信息量的 target，任何网络训练都没有科学意义。
- **数据/角色：**
  - source fit 42 recording groups；
  - 当前冻结 672 anchors、ragged eligible candidate domain；
  - development/calibration/test/NavDream/target 不读取。
- **Runs：**
  1. 已预注册 B4R1 candidate-local one-frame oracle；
  2. B4R1 接口有效后，独立预注册并 materialize scoped augmented-context target；one-frame
     GO/HOLD/STOP 不替代该资格；
  3. current one-frame 与 full target 的 sign、top-choice、regret、\(1/7\) certified coverage；
  4. minimal previous-action geometry token 的解释度诊断。
- **主指标：**
  - B4R1 已冻结四项 gate；
  - full target pair-sign/top-choice 反例计数；
  - full oracle opportunity、group-level mean gain；
  - zero-challenger、tie、temporal-ambiguous 分母完整性。
- **成功条件：**
  - B4R1 实现接口有效，one-frame outcome 按原协议完整保留；
  - full target primary/replay/audit exact；
  - full target 含足够非 tie relation，不由极少数组驱动。
- **失败解释：**
  - B4R1 interface INVALID：停止并修复接口，不解释科学结果；
  - B4R1 scientific STOP：只停止 one-frame supervision；仍须独立资格化 contextual target；
  - full target 几乎无信号：优先审查 candidate supply/任务定义，不搜索更复杂网络；
  - previous context 产生反例：current-only 模型永久降为 baseline。
- **目标产物：**Appendix label table；官方目标定义图；不是主方法结果。
- **优先级：**MUST-RUN。

## Block B1：网络创新与必要性隔离

- **Claim tested：**完整时空 differential query 在相同资源下优于通用 candidate relation/scorer。
- **为什么存在：**D-JEPA、World4Scorer、DA-WAM/GTRS 已覆盖 generic relation、outcome scorer 和 candidate conditioning；不做 matched comparison 就没有网络创新。
- **数据/角色：**
  - source train；
  -未用于训练的 source development 只用于模型选择，但必须标记 historically exposed；
  - final test/NavDream 不参与 architecture 或超参数选择。
- **三类 baseline family：**
  1. **Proxy/perception family：**trajectory-only、zero/prior、旧 field/SDF/BEV readout；
  2. **Generic scorer family：**容量匹配的 independent scorer、D-JEPA-style set relation、World4Scorer-style candidate state；
  3. **AORTA family：**共享视觉编码 + full differential query + full-margin head。
- **AORTA 决定性消融：**
  - full support vs 2.5 s prefix；
  - signed positive/negative vs unsigned/early subtraction；
  - global KV vs local-only；
  - ordinary comparator vs exact antisymmetric comparator；
  - current-only vs minimal previous-action token；
  - task-only vs task+orbit；
  - shared single encoding vs per-candidate repeated encoding，仅用于精度/时延边界。
- **公平合同：**
  - 相同 frozen backbone features、training examples、ragged candidates、full labels；
  - 参数/FLOP/token 数在预注册 tolerance 内匹配；
  - 相同 optimizer search budget 与 3 个最终 seeds；
  - 先单 seed 做实现/明显失败筛查，不能据此报告主结果；
  - candidate provider 完全冻结。
- **主指标：**
  - group-level full-margin MAE/normalized error；
  - pair-sign error、top-choice error、selection regret；
  - false intervention；
  - correct-minus-mismatch/removed；
  - seed 稳定性。
- **成功条件：**
  - AORTA 相对 strongest matched generic scorer 在预注册 primary endpoint 上组级置信区间支持改善；
  - 优势在至少两个独立决策指标上一致，不只来自一个阈值；
  - 删除 differential support 复现预期失败，且参数/计算差异不能解释收益。
- **失败解释：**
  - generic scorer 相当或更优：删除 differential-support 网络创新，采用最强简单 scorer 并降低论文定位；
  - 只有更多参数版本有效：不支持结构必要性；
  - task+orbit 只减 drift、不减 regret：删除 orbit 或重审合法干预。
- **主文目标：**Table 1 方法隔离；Figure 2 反例—架构—消融。
- **优先级：**MUST-RUN。

## Block B2：NavDream 外观机制泛化

- **Claim tested：**模型依赖正确场景视觉证据，并在未见外观中保持任务关系，而非只记忆轨迹/空间先验。
- **为什么存在：**source controlled appearance 的稳定性不能替代真实未见外观泛化。
- **数据/角色：**
  - architecture、checkpoint、appearance、tie rule、metrics 全冻结后一次性打开 NavDream OOD；
  - recording group 为统计独立单位；
  - 若 NavDream 曾被历史访问，必须另设最终 untouched target 或降低独立测试主张。
- **Compared systems：**
  - strongest proxy/perception baseline；
  - strongest matched generic scorer；
  - frozen AORTA；
  - correct、geometry-matched mismatch、removed/trajectory-only controls。
- **主指标：**
  - appearance-wise full relation error、pair-sign、top-choice、regret；
  - correct image 相对 mismatch/removed 的配对差值；
  - relation drift与 task error 联合图；
  - worst group、failure group、severity trend；
  - visual evidence/support/abstention 分层。
- **成功条件：**
  - correct image 优势的组级置信下界为正；
  - AORTA 相对 strongest matched scorer 的 OOD regret/false-intervention 改善不由单一 appearance 或少数组驱动；
  - clean 与 OOD 同时报告，稳定性不掩盖错误关系。
- **失败解释：**
  - correct 与 mismatch/removed 相当：视觉必要性失败，HOLD 并定位轨迹先验；
  - source有效、NavDream崩溃：停止当前 appearance-robustness 主张；
  - 只在弱 appearance 成立：限制适用范围，不称通用天气鲁棒。
- **主文目标：**Table 2 上半部分；Figure 3 relation correctness–stability。
- **优先级：**MUST-RUN。

## Block B3：三类 planner 的行为净收益与 clean no-harm

- **Claim tested：**relation 改善实际改变驾驶行为，并能跨回归、diffusion/generative、generate-and-select/scoring planner 复用。
- **为什么存在：**离线 relation 指标不是驾驶收益；单 planner 不能支持通用接口。
- **冻结 planner classes：**
  1. regression planner；
  2. diffusion/generative planner；
  3. generate-and-select/scoring planner。
- **公平合同：**
  - planner 权重、native output、candidate provider 和候选数冻结；
  - AORTA 只 rerank/abstain，不重训 planner；
  - native、strongest generic scorer、AORTA 使用完全相同 candidates；
  - candidate supply 和 selection gain 分开记账；
  - clean/OOD、planner、appearance 与 group 全部预先列出。
- **hard failure 冻结定义：**
  - baseline-hard 集合在方法 outcome 前由 official metrics 与 appearance-caused 条件定义；
  - 必须是同一物理场景 clean 不 hard、目标 appearance 下 native hard，且 candidate set 中存在可挽救机会；
  - rescued 与 new-hard 使用同一冻结集合和同一阈值；
  - 不删除 zero-challenger、abstained 或失败 groups。
- **主指标：**
  - Official PDMS、DAC、collision/TTC及适用的 NAVSIM v2 指标；
  - recording-group paired change；
  - \(G_{\mathrm{net}}=(N_{\mathrm{rescued}}-N_{\mathrm{new-hard}})/N_{\mathrm{baseline-hard}}\)；
  - false intervention、intervention coverage、candidate-supply ceiling、selection regret；
  - clean group-level non-inferiority 与新增 hard failures。
- **统计合同：**
  - PDMS 最小有意义效应与 clean 非劣界由冻结 baseline 组间方差和功效分析确定；
  - 在目标 split 前冻结效应、alpha、bootstrap/permutation、multiplicity 与 missing-data 规则；
  - 报告每 planner 与预先定义的总体层级，不以事后 pooled result 掩盖失败 planner。
- **成功条件：**
  - 顶会目标下，三类 planner 必须分别达到预注册的组级可信收益标准，不能只依赖 pooled 平均；
  - 预注册总体 OOD endpoint 同时获支持；
  - \(G_{\mathrm{net}}\) 的组级置信下界为正或超过最小有意义效应；
  - clean non-inferiority 通过且新增 hard failure 受控。
- **失败解释：**
  - 只在一种 planner 有效：降级为该架构特定方法；
  - candidate supply 解释全部收益：不能归因于 AORTA selector；
  - OOD 提升但 clean 劣化：主方法失败，不用事后 gating 掩盖；
  - PDMS 提升但 hard-failure 无净改善：只能主张平均指标改善。
- **主文目标：**Table 2 完整 cross-planner behavior；Figure 4 hard-failure transitions。
- **优先级：**MUST-RUN。

## Block B4：可信性、实时性与失败边界

- **Claim tested：**选择性机制非真空、能减少错误替换，并在真实在线预算下运行。
- **为什么存在：**数学 coverage 可通过全拒判获得；离线缓存吞吐不能证明部署可行。
- **Calibration protocol：**
  - 独立 recording-group calibration；
  - 模型、candidate policy、normalization、appearance 与 score 固定；
  - group maximum 同时覆盖 appearances/anchors/legal challengers；
  - 次序统计量超样本资格时取无穷并全拒判；
  - NavDream/新 planner 只报告 empirical coverage，不宣称 source guarantee。
- **Reliability metrics：**
  - group empirical simultaneous coverage；
  - interval radius/sharpness；
  - risk–coverage 与 selective regret；
  - false-improvement/false-intervention；
  - abstention 原因和困难场景分层。
- **Runtime protocol：**
  - 同一硬件、batch 1、warmup 后固定重复次数；
  - 端到端包含图像解码/预处理、一次 backbone、support construction、\(K\) queries、comparison、arbitration 与同步；
  - P50/P95/P99、峰值显存、吞吐；
  - \(K\) 与 horizon scaling；
  - shared encoding 对 per-candidate repeated encoding；
  - 不用缓存 features 作为主延迟结果。
- **预算冻结：**
  - 先测 native planner 与 strongest baseline 的部署延迟分布；
  - 根据目标系统周期与 baseline slack 冻结 AORTA P95 上限，不在结果后填写；
  - 若现有硬件无法代表部署场景，同时报告绝对时延与相对 overhead。
- **成功条件：**
  - calibration coverage 与预注册 sharpness/coverage 下界同时成立；
  - selective arm 相对 forced replacement 降低 false intervention；
  - P95 不超过冻结部署预算，视觉主干成本不随 \(K\) 重复。
- **失败解释：**
  - 近乎全拒判：删除可信仲裁实证贡献，只保留条件定理；
  - OOD undercoverage：不得声称 NavDream guarantee，分析失覆盖；
  - latency 超预算：降级为离线方法；只有保持 B1/B2 机制的压缩方案才可另行注册。
- **主文目标：**Figure 5 risk–coverage/latency；Appendix profiler table。
- **优先级：**MUST-RUN。

## 4. Run Order and Milestones

| Milestone | 目标 | Runs | 决策门 | 计算/时间合同 | 主要风险 |
|---|---|---|---|---|---|
| M0 | 合法 one-frame oracle | B4R1 primary/replay/audit | 原冻结 GO/HOLD/STOP | 无固定 wall cutoff；禁止自动第三次运行 | scorer/候选域接口 |
| M1 | scoped contextual target 闭合 | target-scope audit + materialization + replay/audit + alignment | M0 interface valid；signal 非平凡且 ledger exact | 由 M0 实测 throughput 外推，不预填 GPU 小时 | evaluator scope/history shift |
| M2 | 最强 baseline 与最小网络 | 单 seed smoke；随后 3-seed matched B1 | generic scorer 对照合法；AORTA有增量才保留 | 训练作业数按 frozen matrix；先筛 bug 后正式 seeds | 计算不公平/实现差异 |
| M3 | 视觉机制与 NavDream | held-out controls；一次性 B2 | correct-image necessity + OOD relation | checkpoint/metrics 冻结后才开封 | 场景记忆/目标污染 |
| M4 | 三 planner 行为 | B3 clean+OOD | OOD endpoint、\(G_{\mathrm{net}}\)、clean non-inferiority | 每 planner 使用同一 candidate/scorer cache policy | candidate supply 混淆 |
| M5 | 可信性与部署 | independent calibration + B4 profiler | non-vacuous risk–coverage + P95 | calibration group eligibility；同硬件 batch-1 | 全拒判/时延超限 |
| M6 | 论文完整性 | experiment audit、paper claim audit、强反方 review | 无致命 claim/evidence mismatch | 无新增调参 | 选择性汇报 |

## 5. 第一批实际动作

这些动作按顺序执行，后一步不得绕过前一步：

1. 用户明确授权后，按现有 frozen preregistration 运行 B4R1；未授权保持 HOLD。
2. 若 B4R1 interface valid，无论其 one-frame GO/HOLD/STOP，先注册 evaluator-scope audit 与
   contextual-target materialization/alignment；不直接训练网络。
3. full target 合法后，冻结 B1 的 exact model budgets、tie tolerance、optimizer search 与 seed roster。
4. 先复现 strongest generic scorer，再运行 AORTA；不能只做弱 baseline。
5. B1 通过后才开封 NavDream 机制评估；B2 通过后才进入三 planner 行为。

## 6. Compute and Data Budget

### 已知硬限制

- B4R1：无固定 wall cutoff；保留异常失败、进度与输出上限；
- 正式主方法与 strongest matched scorer：默认 3 seeds；
- 主视觉 backbone 每场景只编码一次；
- 不能修改现有 dinov3 环境；长任务使用隔离 worktree；
- 实验产物进入 NAVSIM_EXP_ROOT 与实验登记系统。

### 不凭空填写的量

当前没有合法 full-target materialization throughput、最终网络显存或 batch-1 latency，因此不填写虚假 GPU-hours。M1 后使用以下公式冻结预算：

\[
H_{\mathrm{total}}
=H_{\mathrm{label}}
+3(H_{\mathrm{generic}}+H_{\mathrm{AORTA}}+H_{\mathrm{core\ ablations}})
+H_{\mathrm{evaluation}}.
\]

每项由一次 dry-run 的 anchors/s、steps/s、峰值显存与 I/O 占比外推，并乘预注册安全系数。不能用结果好坏决定追加 seeds。

### 数据角色

| Role | 允许用途 | 禁止用途 |
|---|---|---|
| source train | 拟合模型和冻结范围内 augmentation | 报最终泛化 |
| historically exposed development | 调试、模型选择、功效/方差估计 | 声称 untouched test |
| independent calibration | 固定模型后的 uncertainty quantile | 再选模型或阈值 |
| NavDream/target OOD | 一次性主泛化与行为评估 | architecture/appearance 调参 |
| final clean/test | clean non-inferiority 与最终确认 | 任何训练选择 |

## 7. Risks and Mitigations

| 风险 | 预先处置 |
|---|---|
| B4R1 one-frame 无 oracle opportunity | 停止 one-frame supervision；由独立 contextual oracle 再裁决 panel |
| one-frame 与 full target 大量翻转 | full labels + previous token 成为必需，不用代理标签 |
| native-history 与 actual-history/rollout 分歧 | 停止 teacher-forced 行为主张；修复历史覆盖或停止 sequential claim |
| generic scorer 同样有效 | 删除 differential-support 创新，采用简单模型并降低定位 |
| correct image 无优势 | 停止视觉机制主张，分析轨迹/场景先验 |
| NavDream 崩溃 | 停止当前 appearance-robustness 主张，不做目标域适配补救 |
| 三 planner 异质 | 分 planner 设置主门并完整报告；不能用 pooled 平均掩盖失败。若仅部分架构通过，降低通用性主张 |
| clean harm | 主方法失败；只有预注册 selective mechanism 可决定是否干预 |
| calibration 全拒判 | 删除实证可信性贡献 |
| P95 超预算 | 降级离线；压缩另行预注册且保持机制 |
| 结果很多但故事散 | 主文只保留 P1、S1 和行为终点；其余放附录或删除 |

## 8. 主文表图合同

| 位置 | 内容 | 必须回答的问题 |
|---|---|---|
| Figure 1 | common-min/late-fork 反例 → differential evidence operator → native fallback | 为什么稳定 perception 不够？ |
| Table 1 | matched proxy/generic/AORTA relation 与 ablation | 为什么需要这个网络接口？ |
| Figure 2 | correct/mismatch/removed + appearance drift vs regret | 模型真的使用正确视觉信息吗？ |
| Table 2 | NavDream × 三 planner × clean/OOD behavior | 真正改变了什么？ |
| Figure 3 | hard-failure rescued/new-hard transitions | 平均分数之外是否减少关键失败？ |
| Figure 4 | risk–coverage + P50/P95/K scaling | 可信性是否非真空、能否实时？ |

## 9. Final Checklist

- [x] 主问题、唯一 estimand 与两项以内贡献候选已冻结；
- [x] 标准数学与项目创新边界已写明；
- [x] B4R1 授权边界保持；
- [ ] 合法 full target 与 oracle opportunity 已验证；
- [ ] strongest generic scorer 已公平复现；
- [ ] differential evidence 的增量已隔离；
- [ ] correct-image necessity 在新 relation head 上成立；
- [ ] NavDream 未见外观通过；
- [ ] 三类 planner 的行为净收益通过；
- [ ] clean non-inferiority 通过；
- [ ] simultaneous calibration 非真空；
- [ ] batch-1 P50/P95 通过；
- [ ] experiment audit 与 paper claim audit 完成。

当前总体状态：**PLAN_READY__EVIDENCE_CHAIN_INCOMPLETE__NOT_ICCV_READY**。
