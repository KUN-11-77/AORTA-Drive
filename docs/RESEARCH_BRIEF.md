# AORTA-Drive Research Brief

> **2026-10-26 R005E2S：`MECHANICAL_PASS__RELATION_ONLY_BINDING_SECONDARY__PARENT_STOP_IMMUTABLE`。**
> 冻结V2 checkpoints在S2 disjoint-geometry与S3 hash-random donor输入干预下完成6-fold primary/replay，
> exact replay、parent state、fold/donor隔离、finite与sealed-role门全部通过。S2/S3相对correct的relation
> MSE差为`+0.021940 [0.015855,0.028243]`与`+0.033896 [0.025969,0.042040]`，但regret差的CI均跨零；
> 因此只支持donor-robust平均relation binding，不支持decision-sufficient视觉信息。R005E2B父STOP不变；
> 按预结果第一性决策树，只允许预注册最小decision-aligned objective qualification，不直接训练DQ或读NavDream。

> **2026-10-26 R005E2S donor sensitivity：`IMPLEMENTATION_FROZEN__SECONDARY_ONLY__PARENT_STOP_IMMUTABLE`。**
> 在R005E2B主门严格STOP后，按既有预注册冻结实现S2 disjoint-geometry与S3 hash-random两种
> donor输入干预；复用既有V2 checkpoints，不训练、不选模、不读取封存角色。正式链为6-fold
> primary、同卡exact replay与独立机械audit，统计以42个recording groups为单位。无论结果为何，
> 均不得改写R005E2B的`STOP_CURRENT_VISUAL_MAIN_CLAIM`；本实验只区分scene-binding结论是否依赖
> primary donor构造。实现提交`73b5e46d5e06e064d1da60007d3af6e8878bf746`，469项仓库测试通过。

> **2026-10-26 R005E2B视觉可恢复性：`MECHANICAL_PASS__UNSUPPORTED__STOP_CURRENT_VISUAL_MAIN_CLAIM`。**
> 6 folds×3 arms×3 seeds的54个primary与54个replay全部完成，prediction/model-state exact、OOF/fold/
> target/provenance/finite/sealed-role门通过。V2 correct relation MSE相对mismatch改善`-0.008010`
> `[-0.014059,-0.001854]`，但相对removed的CI跨零；decision regret相对removed/mismatch均未改善，
> mismatch intervention也未可靠伤害regret，`rho_vis=-0.025449`。父裁决不可由post-hoc分析改写。
> 只读失败解剖显示435/672 anchors有exact top-2 tie、平均best-set大小2.81；视觉平均关系信号与
> choice utility改善近乎不相关。当前最合理假设是relation回归信号未对齐稀疏决策边界，但尚未证明
> 应使用任何特定新loss或Differential Query。详见`experiments/R005/R005E2B_RESULTS.md`。

> **2026-10-25 Best-Paper 实现合同：`MANDATORY_PROJECT_CONTRACT__NO_AWARD_GUARANTEE`。**
> 后续方法实现、训练、评估、审计与论文主张统一执行
> `docs/BEST_PAPER_IMPLEMENTATION_STANDARD_V1_CN.md`：数学对象、因果/坐标/角色不变量、模块五问、
> matched baseline、公平预算、组级统计、exact replay、端到端时延、失败分类与claim边界均为硬门。
> 已冻结的R005E2B不做中途修改，完成后按该标准审计。该合同保证研究过程达到候选级严谨性，
> 不保证正结果、录用或Best Paper奖项。

> **2026-10-25 Best-Paper first-principles blueprint：`DECISION_QUOTIENT_NORTH_STAR__EMPIRICAL_QUALIFICATION_PENDING`。**
> 顶级论文主线进一步收紧为“视觉观测的决策充分商空间”：对固定候选与因果历史，两个观测仅在
> 诱导相同native-relative contextual-relation分布时等价。外观鲁棒不再定义为全局feature相同，
> 而定义为跨外观保持该条件分布。AORTA用完整时空signed candidate difference提出query、全局视觉
> 回答、同时下界决定是否干预。该构型只保留一个主贡献和一个从属机制；Best Paper资格取决于
> decision-sufficiency gap、matched differential advantage与跨外观/跨planner行为闭环三项发现。
> 详见`refine-logs/2026-10-25_aorta_best_paper_blueprint/AORTA_BEST_PAPER_FIRST_PRINCIPLES_BLUEPRINT_V1_CN.md`。

> **2026-10-25 Best Paper readiness audit：`STRONG_PROBLEM_AND_THEORY_SEED__BEST_PAPER_EVIDENCE_NOT_YET_PRESENT`。**
> 以高于普通ICCV接收的标准审查后，当前具备有价值的问题重述、第一性理论入口和严格实验纪律，
> 但尚缺三项决定性发现：可推广的decision-sufficiency gap、Differential Query对matched closest
> scorers的不可替代优势，以及NavDream×跨planner×sequential×latency的行为闭环。后续不增加并列
> 模块，只按R005E2→matched architecture→OOD/behavior顺序裁决。详见
> `refine-logs/2026-10-25_aorta_best_paper_audit/AORTA_BEST_PAPER_READINESS_AUDIT_V1_CN.md`。

> **2026-10-25 第一性原理 canonical 化：`FIRST_PRINCIPLES_RELATION_SYSTEM_RECORDED__EMPIRICAL_CORE_UNPROVEN`。**
> 当前主线已从潜在世界、外观干预、合法候选与上下文效用出发，统一推出 native-relative estimand、
> 视觉条件增量、观测纤维可辨识性、完整时空 signed differential query、全局视觉回答、条件分布、
> 同时下界仲裁与实时约束。R005E2 被明确定位为视觉信息资格门而非最终 AORTA；后续必须依次通过
> matched architecture、appearance、reliability、sequential/cross-planner 与 latency gates。完整规范见
> `docs/AORTA_FIRST_PRINCIPLES_THEORY_V1_CN.md`。该整理不修改任何冻结实验、阈值或科学结果。

> **2026-10-25 训练前理论再核验：`THEORY_COHERENT__VALID_FALSIFICATION_GATE__POSITIVE_RESULT_NOT_GUARANTEED`。**
> 当前链条已严格区分 `U_ctx`、完整 policy value 与驾驶收益，并以带 ragged mask 的 Bayes 风险恒等式
> 证明 R005E2 检验的是“给定候选/历史后，视觉是否改变 margin 条件均值”。有限网络仍含逼近、估计
> 与优化误差，故任何严谨理论都不能预告 correct visual 必然胜出；可以保证的是 GO/MIXED/STOP/INVALID
> 均有冻结解释。合成无真实 endpoint 的 learnability sanity 已证明视觉—候选通路可学习简单关系，
> 但不构成真实科学证据。另已澄清 `rho_vis` 的分子是 decision-regret 降低而非 relation MSE，量纲
> 一致且未改变门槛。完整核验见
> `refine-logs/2026-10-25_aorta_theory_revalidation/AORTA_THEORY_SYSTEM_REVALIDATION_V1_CN.md`。

> **2026-10-25 R005E2 history-semantics fail-closed repair：** 首次scientific bundle资格构建在
> 任何GPU训练或科学指标产生前失败，因为原实现错误地把9点、0.5秒的previous native planner
> 输出当作41点输入。服务器只读核验确认：R005E1B登记的previous official simulated states为
> `41x11`，独立重算SHA逐字节一致。冻结修复将历史定义为该状态轨迹去除0.5秒已过期前缀后的
> `states[5:41]` rear-axle SE(2)，再变换至current-anchor ego frame，得到36点、0.1秒的因果重叠
> context；禁止插值/padding。candidate/history时间通道均使用真实`0.1s/4s`尺度，不再按各自长度
> 拉伸。该修复仅闭合`U_ctx`条件变量与网络输入语义，不提供视觉、决策或驾驶收益证据；须以新
> 资格目录完成primary/replay exact及机械审计后才能启动54-job科学链。

> **2026-10-25 R005E2A outcome-free资格门：`MECHANICAL_PASS__GO_TO_R005E2B_RUNNER_FREEZE__SCIENTIFIC_CLAIM_UNTESTED`。** 672-anchor冻结DINO cache完成primary/replay：42 groups、8,064个唯一特征、1,344个投影产物逐文件重开，逻辑manifest/feature/projection全部byte-exact，checkpoint、151个DINO源码与预处理hash闭合；sealed roles和科学标签未访问。S2 disjoint-geometry与S3 hash-random共14,784个non-gating donor assignments在科学输出前冻结并审计PASS。单fold真实I/O资源演练完成448/448 backward steps，峰值365.6MB、总耗时37.999s，模型/AdamW状态/恢复前向exact。旧审计器错误假定`P95 >= mean`，由冷启动长尾反例否定；提交`81a2cb5`修复后对同一结果重审PASS，未重跑。随后预算复核又发现原`0.570 GPU-hours`漏计refit与epochs；同一测量纠正为training-only 1.282 GPUh下界、51.299 GPUh最坏上界（均不含validation/test/I/O/audit），并把正确算术加入硬门。外部review unavailable。现在只解锁R005E2B runner与auditor冻结；正确视觉是否提供增量仍未检验，Differential Query、NavDream与驾驶收益继续HOLD。详见`experiments/R005/R005E2A_PREFLIGHT_STATUS.md`与`experiments/R005/R005E2A_EXPERIMENT_AUDIT.md`。

> **2026-10-25 R005E1B contextual oracle：`SUPPORTED_ON_SOURCE_FIT_NATIVE_HISTORY_UCTX__GO_TO_VISUAL_RECOVERABILITY_PREREGISTRATION__NO_TRAINING_YET`。** 修复后的672-anchor因果上下文、ragged候选域与previous metric cache完成primary、完整replay和机械审计：4,536候选的已保存components、分数、统计与裁决复算一致，ledger exact，全部数值有限，sealed roles未访问。group-equal contextual oracle gain为`0.0166504`，42-group bootstrap 95% CI `[0.0087722,0.0262034]`，42/42 groups存在strict native improvement。filtered one-frame相对`U_ctx`仍有`252/672=37.50%` top-set disagreement、`1673/13402=12.48%` pair-sign disagreement，忽略context的group-equal mean regret为`0.0157405`；这证明one-frame监督不等价于登记的`U_ctx`，但尚未通过matched消融证明某一种previous-action token编码是任何learner的必要形式。官方two-stage mapping覆盖0/672，`Q_pi_ref`分支fail-closed，结果不支持RGB可恢复性、架构优越性、完整伪闭环价值、NavDream、驾驶收益或安全。独立完整性复核裁决为`WARN`：EC未从原始state arrays独立重算，运行时source import与系统级sealed访问也未被本地归档单独证明。下一步只解锁visual recoverability的预注册，不直接训练。完整结果与审计见`experiments/R005/R005E1B_RESULTS.md`和`experiments/R005/R005E1B_EXPERIMENT_AUDIT.md`。

> **2026-10-25 R005E0目标语义审计：`RAW_ONE_FRAME_NE_FILTERED_CONTEXT_NE_TWO_STAGE_VALUE`。**
> B4R1登记的raw `pdm_score`在human filter前形成，而官方完整评估从filter后components重新聚合；
> posthoc机械审计发现506/4536 candidates、111/672 anchors不一致，导致473/13402 pair-sign与
> 55/672 top-set差异。原B4R1 GO不变，但只支持其预注册raw one-frame对象。后续严格区分
> `F_raw^{1f}`、过滤后`F_tilde^{1f}`、带真实previous trajectory的`U_ctx`、固定后续策略的
> two-stage action value `Q_pi_ref`及策略级`J(pi)`。该条为当时的E0裁决；R005E1B现已完成，但网络训练、
> development/calibration/test/NavDream/target继续锁定。

> **2026-10-25 B4R1正式结果：`SUPPORTED_ON_SOURCE_FIT_ONE_FRAME__GO_TO_CONTEXTUAL_PREREGISTRATION__NO_TRAINING_YET`。** 合法ragged candidate domain上primary/replay与本地复算全部PASS；42/42 groups存在nonzero pair并至少有一个strict native improvement，group-equal oracle gain为`0.0202745`，42-group bootstrap 95% CI `[0.0117815,0.0301722]`。该证据只资格化candidate-local one-frame oracle，不支持RGB、full contextual target、NavDream、训练、驾驶收益或安全。下一步仅解锁evaluator-scope/contextual materialization的独立预注册。

> **2026-10-25 投稿版理论审核：`SCOPED_TARGET__INTEGRABLE_STAR__ROLLOUT_HISTORY`。**
> rapidPeer 式 P0 审核修正三处边界：`7/8+1/8` 只在冻结 evaluator 逐项核验的两帧候选效用层
> 使用，不未经证明称完整 EPDMS；部署输出改为同一 native-star margin 向量，任意 challenger
> relation 由差分得到，精确避免循环偏好；previous selected trajectory 被视为干预后的内生状态，
> 必须分别评估 native-history、actual-selected-history 与 sequential rollout。B4R1 interface
> invalid 仍阻断后续，但其 one-frame GO/HOLD/STOP 不再代替 contextual target 裁决。正式运行
> 继续 HOLD。投稿版闭环见
> `refine-logs/2026-10-25_aorta_submission_theory/AORTA_PAPER_BLUEPRINT_V1_CN.md`。

> **2026-10-25 frontier novelty correction：`D_JEPA_CLOSEST_BASELINE__NOVELTY_NARROWED`。**
> 官方 arXiv 正文审查确认，D-JEPA 已覆盖 decision-local gap、完整候选关系、native-relative bounded
> correction、risk penalty、margin gate 与 native fallback；World4Scorer 又覆盖 outcome-grounded
> candidate state、future anchor、negative bank 与 previous-plan reranking。故 AORTA 不得再以“关系
> 学习”“冻结 planner 上的 relative head”或“可信 fallback”本身作为创新。当前唯一可防守的主命题
> 收紧为：在驾驶 appearance intervention 下，以完整时空 signed candidate difference 提出视觉
> query，是否比 matched D-JEPA/World4Scorer-style scorer 更能保持 full-task relation，并减少 OOD
> 错误替换而不伤害 clean。若 matched study 不胜，删除 differential-support 网络创新主张。B4R1、
> 网络训练、NavDream 和行为收益状态均未改变，仍 HOLD。详见
> `references/2026-10-25_frontier_decision_scorer_novelty_audit.md` 与
> `refine-logs/2026-10-25_aorta_post_b3/ICCV_CORE_STORY_V5_AFTER_DJEPA_CN.md`。

> **2026-10-25 contextual-row语义收紧（已由R005E0进一步修正）：`U_CTX_REQUIRES_PREVIOUS_CONTEXT`。**
> filter后row utility满足`U_ctx=7/8 F_tilde+1/8 M*EC`，故上一周期实际selected trajectory属于
> `U_ctx`上下文，不能再由经验alignment决定是否存在。最小previous-action geometry token将进入该relation
> 输入；复杂temporal encoder仍由matched消融决定。对任意候选对，`|Delta_one|>1/7`严格保证full
> sign一致；one-frame top choice的最坏full regret至多`1/8`，但该界并不足以授权其作为训练真值。
> B4R1仍只作低成本raw oracle gate；后续监督须由R005E1在`U_ctx`或合法`Q_pi_ref`之间裁决，任何实际
> sign/top-choice反例都否定current-only普遍充分性。详见
> `OFFICIAL_TARGET_ALIGNMENT_AND_TEMPORAL_CONTEXT_V2_CN.md`。

> **2026-10-25 B4R1开封前来源审计：`SERVER_PROVENANCE_REPREFLIGHT_PASS__FORMAL_HOLD`。**
> 正式 outcome 前已补齐 primary/replay 的status、commit、config、frozen-input、implementation、
> sealed-role与full-PDMS boundary核验。随后在任何 scorer outcome 产生前，用户明确将资源合同
> 修订为无固定 wall cutoff；候选、标签、阈值、seed、统计和裁决均未改变。原冻结来源提交
> `53f16c1`在服务器同一NAVSIM环境12/12合同测试通过；scorer未调用，任何科学outcome与
> development/calibration/test/NavDream/target均未访问。preflight artifact SHA-256为
> `4477fe27…b9191`。该结果只说明正式B4R1已经具备更强的可审计执行资格，明确授权前仍HOLD。

> **2026-10-25 理论—网络—行为闭环合同：`FORMAL_CLOSURE_CANDIDATE__EMPIRICAL_CORE_UNPROVEN`。**
> 当前主线已压成五层建筑：合法外观干预、native-relative estimand、完整时域差异支持、同时关系
> 不确定集合、native-preserving 实时执行。六个条件命题分别约束 relation 输出、signed support、
> global visual context、task+orbit loss、`2 epsilon` regret bridge 与 simultaneous lower-bound
> arbitration；每个网络模块均绑定已观察失败和删除实验。顶会故事只保留
> appearance-interventional differential-support relation learning 一项主贡献与可信仲裁一项从属
> 机制。该闭环不新增经验主张；R005 target alignment、matched scorer、NavDream、三 planner、
> clean no-harm 和 batch-1 latency 仍全部是硬门。详见
> `refine-logs/2026-10-25_aorta_post_b3/THEORY_ARCHITECTURE_CLOSURE_V1_CN.md`。

> **2026-10-25 relation仲裁语义分离：`DATA_FREE_CONTRACT_PASS__NO_EMPIRICAL_CLAIM`。** 历史
> `native_preserving` 处理绝对安全区间并要求先证明native不安全；新理论处理
> `challenger-minus-native` margin，只要求同时lower bound严格为正。两者已禁止复用同一接口。
> 新增数据无关参考实现，现有11项性质测试；与既有decision-quotient、legacy arbitration和
> calibration合计29项定向测试通过，strict mypy通过。测试覆盖条件性真实改善、NaN/Inf
> fail-closed、零challenger、`2 epsilon` appearance drift与常数坍缩反例、42-group 95%有限/99%
> 全拒判边界，以及one-frame `1/7`严格符号证书、边界反例与`1/8` top-choice regret界。
> 本地`dinov3` conda入口失效，验证使用Python 3.13 +显式`PYTHONPATH=src`；未修改环境，服务器正式
> 环境不受影响。该结果不证明coverage、NavDream、网络或驾驶收益，也不授权B4R1。

> **2026-10-25 two-frame上下文roster审查：FORMAL_OUTCOME_FREE_INTERFACE_PASS。**
> 现有672-anchor fit roster内0/672存在≤0.55 s相邻token，直接调用官方aggregator会把two-frame
> 权重全部置零并伪造full=one-frame的100%对齐；但从261个官方raw logs逐token恢复时，672/672
> anchors均有唯一合法previous row，42/42组为16/16，gap为0.498078–0.500686 s。后续alignment
> 必须使用augmented-context roster，禁止把anchor index当时间。已实现fail-closed resolver、summary、
> runner与8项测试；冻结提交06d3ab0的primary/replay 672行ledger逐字节一致，结果SHA-256为
> 2ae90f5c…512c。该结果只闭合上下文接口，B4R1与网络训练状态不变。

> **2026-10-25 ICCV/Oral叙事与完整目标审查：`COHERENT_HYPOTHESIS__NOT_YET_ORAL_READY`。**
> 当前唯一故事收紧为“稳定感知不推出可区分决策；应在完整时域native–challenger差异支持上学习
> 外观稳定的任务关系，并在不可识别或同时下界不足时保留native”。官方源码进一步确认完整分数
> 在存在合法相邻帧时含two-frame extended comfort：`F_full=(7/8)F_one+(1/8)H*C2`；无相邻帧
> 时官方实现退化为one-frame。因此previous-action
> context可能是任务定义强制的必要输入，而不是任意加模块。B4R1通过后必须先做冻结alignment；
> matched scorer、NavDream、跨planner、clean no-harm与时延未通过前，状态仍非ICCV/Oral ready。
> 详见`ICCV_ORAL_STORY_READINESS_V4_CN.md`与`OFFICIAL_TARGET_DISCOVERY_AND_ALIGNMENT_V1_CN.md`。

> **2026-10-25 scorer因子与surrogate对齐审查：`ONE_FRAME_ORACLE_ONLY__TRAINING_TARGET_HOLD`。**
> 官方源码确认B4R1 one-frame score为NC/DAC/DDC/TLC乘积与EP/TTC/LK/HC加权平均的乘积，并省略
> two-frame extended comfort。差异支持因此只作为comparison query geometry，不能等同完整utility；
> 主模型继续直接预测native-relative scalar margin，factor head仅作DA-WAM匹配消融。B4R1即使
> 通过，也必须先预注册one-frame对完整official target的pair-sign、top-choice、regret与two-frame
> harm alignment gate；通过前禁止用B4R1标签训练主网络。详见ADR-0072与scorer alignment文档。

> **2026-10-25 顶会叙事与网络反方审查：`DIFFERENTIAL_SUPPORT_RELATION_V2__HYPOTHESIS_ONLY`。**
> 现有 Shared–Difference–Interaction v1 的机制解释不成立为结构保证：非线性共享查询不推出候选
> 共同外观偏置，自由 interaction 可绕过 difference 分支。B3 common-min、BS2 晚期分叉与
> BS4A-D0 无证据差异共同导出新的条件候选：用完整时空 signed occupancy contrast
> `nu_i0=mu_i-mu_0` 的 challenger-exclusive、native-exclusive 与 common support 生成几何 query，
> 但保留全部视觉 token 作为 key/value，以反对称 relation operator 输出合法 margin。二次反方审查
> 已修正过强结论：D0 的直接投影缺失不等于全局不可观测，provenance 只进入 scale/calibration；
> 只有全部可访问表示相同而真实 margin 符号相反，才构成不可识别二世界并要求拒判。
> DA-WAM 已覆盖 candidate-conditioned scorer，The Constant Eye 已覆盖冻结 DINO/NavDream，
> RiskWorld/Expand-and-Prune 已覆盖 selective replacement，因此只有 matched scorer 对照、NavDream、
> 跨 planner、clean no-harm 与时延共同证明差异支持的增量，才有资格写成 ICCV/CVPR 主贡献。B4R1
> 正式运行仍未获授权，网络训练继续 HOLD。张量、mask、前向、loss、推理和 metamorphic tests 已
> 写入实现合同，但只能在 B4R1 通过后实施。详见 ADR-0071 与 v2/v3 方法故事文档。

> **2026-10-25 selective calibration定理收紧：`GROUP_SIMULTANEOUS__FINITE_SAMPLE_LIMIT_EXPLICIT`。** 校准单位冻结为recording group；每组先对appearance/anchor/legal challenger的归一化误差取最大值，再用`k=ceil((n+1)(1-alpha))`次序统计量。若未来独立calibration有42组，95%覆盖可用有限`k=41`，99%需要`k=43>n`，无额外假设只能给无穷半径并全拒判；现有fit/development不能回收作calibration。禁止把672 anchors当独立样本，也禁止把source exchangeability外推为NavDream或跨planner保证。详见理论命题4.1与ADR-0070。**

> **2026-10-25 R1 scorer语义复检：`CANDIDATE_LOCAL_SCORE_DIFFERENCE__SERVER_PREFLIGHT_PASS__FORMAL_HOLD`。** 官方调用为每条轨迹单独构造`[metric-cache PDM baseline, evaluated trajectory]`并返回pred score；native与challenger不是联合评分。标签现严格定义为两个candidate-local one-frame scores之差。runner在评分前强制`PROGRESS_MODE=eval`，primary/replay/audit均记录核验；提交`08081cc`的服务器11/11合同测试、implementation/official source hashes与`gen_gt` fail-closed反例通过，未调用scorer或读取outcome。正式R1仍未授权。详见`H7BS4B4R1_SCORER_SEMANTICS_AUDIT.md`和ADR-0069。**

> **2026-10-25 minimal-estimand correction：`RELATION_FIRST__PARAMETERIZATION_LATER`。** 当前部署对象正式收紧为合法ragged候选域上的native-relative task margin向量，而不是预设node potential或complete graph。其因果/信息论目标是在保持decision sufficiency时删除同场景无关外观信息；核心loss直接监督relative margin，appearance项作为source-frozen约束，correct/mismatch/removed controls排除常数与道路先验。由同时margin误差可直接得到`regret <= 2 epsilon`，无需谱隙。complete-score/Hodge只保留为matched消融候选。详见`PREFERENCE_QUOTIENT_THEORY_V0_CN.md`与ADR-0068。**

> **2026-10-25 preference结构创新预死亡审查：`STAR_GRAPH_EXACTNESS_VACUOUS__HODGE_NOT_A_CORE_CLAIM`。** 若监督/部署只比较native与各challenger，star incidence \(D_\star\) 对任意relative vector均满射：取`s0=0, si=q_i`即可，因此node-score difference与直接输出native-relative vector只是重参数化，star上没有cycle可约束。complete-graph edges又由native-conditioned scalar代数派生，不能在实验前宣称独立任务结构。后续只有在B4R1通过后，以native-vector、star-score、complete-score、free complete-edge进行参数/FLOP/数据/seed匹配，并在held-out regret/selection而非train cycle residual上证明优势，才允许保留Hodge/结构贡献；否则主线收缩为appearance-quotient relation target + simultaneous-uncertainty selective arbitration。详见`refine-logs/2026-10-25_aorta_post_b3/STRUCTURAL_NOVELTY_PREMORTEM_V1_CN.md`。**

> **2026-10-25 B4接口失败已转化为B4R1候选域一致性预注册，尚未运行。** B4 的48个失败
> anchors实际pose全部bit-exact；错误是168个冻结intent-ineligible challengers仍进入oracle。
> B4R1因此把候选membership提升为estimand的一部分：每个context的合法顶点集为native加预B4
> 冻结eligible challengers；标签、pair graph、Laplacian、oracle与后续loss必须共享同一集合。
> 15个零challenger anchors保留并贡献零gain/零pair rate，禁止complete-case deletion。R1沿用B4
> 原阈值和bootstrap，B4 outcome禁止作为设计输入；已实现可变候选域summary、fail-closed门、
> runner/audit与合成反例测试。冻结提交`9664e4b`的服务器zero/partial/full/sentinel interface smoke
> 已PASS，但未自动启动第三次正式运行。详见
> `experiments/R004H7/H7BS4B4R1_PREREGISTRATION.md`。

> **2026-10-25 R004H7-BS4B3：`STOP_ROAD_MARGIN_AS_GENERAL_RERANKER`。** 预注册
> temporal first-fork 对2,033个common-min pairs均可定义，但只恢复170个exact-nonzero branch
> contrasts；组等权恢复率8.26%，95% bootstrap CI为6.34–10.29%，远低于严格多数门。固定
> 2.5 s suffix只恢复464个。spatial-exclusive在1,568个双侧非空pair上全部恢复，但465个nested
> pairs至少一侧独占集合为空，总体适用率77.13%低于95%门，不能事后替代primary。机械重放
> 672/672 exact；learned field、DINO、outcome、development及封存数据均未读取。该结果停止
> global/fixed/temporal scalar-min道路裕度作为一般候选reranker；H7C/network/NavDream/decision
> evaluation继续HOLD。下一步先查新并审查新的decision-sufficient数学对象，不自动搜索架构。

> **2026-10-25 post-B3 literature gate：普通替代方案已排除。** CVPR 2019 Oral NMP 已使用时空
> cost volume沿轨迹累加，故signed occupancy integral不是新贡献；CVaR/EVaR已有成熟风险规划；
> GTRS、DA-WAM与TOAD已覆盖trajectory tokenizer、candidate-conditioned latent、hard negatives、
> off-proposal scorer与test-time search。当前唯一值得继续查验的候选是appearance-quotient
> preference：直接学习外观干预下的native-relative task score differences。scalar-score
> coboundary仅是待检验的complete-graph参数化，在native-star上不构成结构约束。该方向尚未通过标签合法性、oracle ceiling、
> direct-pair/GTRS/DA-WAM匹配消融、NavDream或时延门，不授权实现。详见
> `refine-logs/2026-10-25_aorta_post_b3/LITERATURE_AND_METHOD_PIVOT.md`。

> **2026-10-25 H7BS4B4预注册：先验证标签与oracle，不训练网络。** 新候选数学对象只有在
> source-fit 上存在合法、可复算且非平凡的native-relative task utility时才有实现资格。由于NAVSIM
> progress在当前评分面板内归一化，7个候选不得一次批量评分；每个候选必须独立与同一native构成
> candidate-local `[PDM baseline, trajectory]` proposal set，复现candidate-specific reactive traffic与one-frame PDM；native和challenger分别评分后作差，并非联合评分。B4冻结42组×16 anchors、完整
> 独立重放和group-level oracle-gain裁决；two-frame comfort、完整official PDMS、跨planner、NavDream
> 与网络收益均不在本轮主张内。详见`experiments/R004H7/H7BS4B4_PREREGISTRATION.md`。

> **2026-10-24 post-D0理论迁移审查（已被B3/B4与ADR-0067收紧）：复杂度按必要性而非模块数量治理。** 当时候选核心链为
> Hodge-consistent quotient representation、support-conditioned correlated graph uncertainty、
> exact shared-variable robust readout与selective Simplex fallback。离散外微分解释potential
> 的gauge与可积性；信息充分性排除“稳定但无用”；联合集合保留公共候选相关性；拒判把证据
> 不足变成正式行为。Sheaf、连续operator、PEA升级、depth和diffusion/RL均为条件模块，须由
> 独立失败证据授权。B3 已停止该 scalar-road-margin readout；ADR-0067 又证明 native-star exactness
> 空泛，因此这段只保留为历史设计记录，不代表当前采用架构。最近资格门为96×64 teacher-critical observability与oracle uncertainty
> non-vacuity；二者通过前不训练新架构。详见`THEORY_TRANSFER_AUDIT.md`与
> `SYSTEM_THEORY_V0_5_CANDIDATE.md`。

> **2026-10-24 R004H7-BS4A-D0：difference-only直接证据修复被否定。** 42 fit groups、
> 672 anchors、3864个native–challenger pairs全部完成精确集合归因；3678个几何不同pair中
> 有8个非空对称差区域完全无直接projective evidence，分布于5 groups/5 anchors，差异仅
> 1–2 nodes但公共区有5–25个支持节点。原2个blind native均无合法challenger，说明原门还
> 混合了candidate supply与decision support。机械重放0失败；H7BS4A STOP不变。下一理论
> 候选改为部分可观测quotient field的不确定集合与共享变量鲁棒比较/拒判，尚未获teacher、
> 网络或决策证据；训练继续HOLD。

> **2026-10-24 R004H7-BS4A：`STOP_OR_REDESIGN_VISUAL_SUPPORT_INTERFACE`。** fit-only
> outcome/teacher-free审计覆盖42 groups、672 anchors、4536个合法候选与261个源日志；
> 42/42 groups达到12/16门，但有2个native候选的全部查询节点均无直接projective
> evidence，按预注册硬门严格STOP。机械重放ledgers、summary和日志hash为0失败。全网格
> 平均支持72.27%、候选节点平均89.28%；实际使用新增`x>=32m`区域的1104个候选，其扩展
> 节点支持率均为100%，故失败不来自96×64远距扩展，而来自两个非扩展native查询。
> teacher构建与训练保持HOLD；只允许一次不读teacher/outcome的候选公共/差异区域归因，
> 不得删例或放宽门槛。完整记录见`experiments/R004H7/H7BS4A_RESULTS.md`。

> **2026-10-24 R004H7-BS3R1：`DEVELOPMENT_FULL_DOMAIN_PASS`。** 固定前向96×64、
> 1m、完整4秒域在24个historically exposed development groups上支持384/384 native与
> 2206/2206 eligible challengers，0 anchors发生候选域越界；24/24 groups达到至少12/16
> 有效panel。另9个anchors无eligible challenger，属于供给而非ROI失败。机械审计重放
> 384 ledgers为0失败，外部reviewer unavailable。该结果只闭合离散空间定义域；视觉
> 可观测性、预测精度、1m分辨率、真实P50/P95时延、决策相关性、驾驶/安全收益和
> test/NavDream/target泛化均未成立。下一步仅允许预注册视觉可观测性和时延资格；
> H7-B2/H7-C与训练继续HOLD。完整记录见`experiments/R004H7/H7BS3_RESULTS.md`。

> **2026-10-24 R004H7-BS2：`PREFIX_UNIVERSAL_SUFFICIENCY_FALSIFIED`。** outcome-blind
> fit诊断覆盖42 groups、672 anchors、3864 eligible native/challenger pairs：128 pairs、
> 36/42 groups的2.5秒lattice footprint集合完全相同而4秒集合不同，严格否定prefix
> set-min的通用决策充分性；主0.5m阈值下151 pairs、40/42 groups晚于2.5秒分叉，完整
> 区分节点平均34.21%未被prefix保留。机械重算0失败。原panel-level规则会选择前移
> 64×64，但它仍丢290/3864 eligible challengers，故不能作为完整域；冻结字典中首个
> 100%保留fit候选的是前向96×64（6144节点）。后者仍需development、视觉可观测性与
> P50/P95时延资格，不是网络效果或创新结论。H7-B2/H7-C继续HOLD。完整记录见
> `experiments/R004H7/H7BS2_RESULTS.md`。

> **2026-10-24 R004H7-BS1：`DEVELOPMENT_SUPPORT_PASS`。** outcome-blind fit选择在冻结的
> 1.0--4.0秒字典中选出2.5秒为最大共同可行前缀：622/672 anchors有效且42/42 groups
> 达到至少12/16；3.0秒仅37/42 groups通过。随后一次性development资格复核得到
> 360/384有效anchors、24/24 groups通过，机械审计对384 artifacts复算0失败。fit首次结果
> 写入因辅助intent字段含NaN而失败；672个既有artifacts随后经official execution逐bit重放
> 与第二层机械audit恢复，未重生成科学panel。外部reviewer unavailable。该结果只支持
> 2.5秒prefix的离散查询定义域充分；它不证明2.5秒包含4秒候选排序所需的信息，因此
> H7-B2继续HOLD，先完成outcome-blind晚期分叉、set-min不可区分性与完整时域ROI诊断；
> 不证明decision relevance、视觉正确性、驾驶/安全收益、完整4秒支持或创新性。完整记录见
> `experiments/R004H7/H7BS1_RESULTS.md`。

> **2026-10-24 R004H7-BS0：`FORWARD_HORIZON_SUPPORT_MISMATCH_OBSERVED_POSTHOC`。** outcome-blind归因重读384个已审计panels：110个native越界与770个candidate ROI failure均只越过前向`x_max`；native首次越界最早在2.1s、中位3.05s，所需前向扩张中位10.016m、最大26.693m；所有初始pose平移/yaw残差为0。23个可配对groups内，native-invalid相对valid的path length平均高24.300m而曲率proxy未增加，说明complete-case样本明显偏向短行程。panel在appearance fields前构建，故视觉证据不能解释本次119个invalid，但ROI内视觉证据充分性仍未测。独立机械audit复现119 identities与全部边界归因，0失败；外部reviewer unavailable。新增定理明确：若候选集合越出网络定义域，完整set-min在无域外假设时不可识别，裁剪会改变estimand。该阶段当时保持prefix选择、H7-B2和H7-C为HOLD；后续状态见上方H7BS1。完整记录见`experiments/R004H7/H7BS0_RESULTS.md`。

> **2026-10-24 R004H7-B：`INSUFFICIENT_PANEL_COVERAGE`。** 冻结24-group/384-anchor shadow run已完成且机械重算384/384通过，但只有265 anchors有效、11/24 groups达到每组至少12/16有效anchors的预注册完整性门；110 anchors因native越出冻结ROI而无效，另9个无有效challenger。candidate级770次失败来自oriented footprint离开64×64、1m、`[-32,32]^2`查询域，冻结provider四类rejection均为0。按协议未计算Spearman/AUROC，因此C13既未获得支持也未被关联性结果否定；H7-C继续HOLD。只允许先注册不读取outcome的support-adequacy diagnosis，不得事后删组、降阈值或改候选。calibration/test/NavDream/target仍封存，外部reviewer为`REVIEW_UNAVAILABLE`。完整记录见 `experiments/R004H7/H7B_RESULTS.md`。

> **2026-10-24 R004H7-A：`PASS_DATA_FREE_THEORY_GATE`。** exact decision quotient 的共同偏移不变性、set-min oscillation bound、\(2\eta\) regret bound、potential/cut-space等价、全局非线性反例与tie policy共290k+登记案例0失败；完整仓库274 tests与新增模块strict mypy通过。该结果只闭合数学接口，不提供视觉或驾驶证据。当时仅授权 H7-B 实现；其后续实现资格见上一条，24-group payload读取、H7-C训练、calibration/test/target仍HOLD。外部reviewer仍为`REVIEW_UNAVAILABLE`。完整记录见 `experiments/R004H7/RESULTS_H7A.md` 与 `H7A_AUDIT.md`。

> **2026-10-24 R004H6R1 held-out visual gate：`SUPPORTED_ON_HELDOUT_DEVELOPMENT`。** 冻结模型在 24 个 historically exposed、但未参与当前 checkpoint 拟合/选点的 development recording groups（384 anchors）上完成一次性评估；两条预注册主张均通过：clean task MAE `0.3699m`，优于 fit-only prior `0.4532m` 与 zero `0.5543m`；mismatch-minus-correct `+0.1478m [0.1391,0.1569]`，removed-minus-correct `+0.1855m [0.1681,0.2029]`，各确认条件失败组均为 0。clean replay mismatch 为 0，calibration/test/target 未访问。legacy float32-artifact auditor 的 1 个 secondary salient-edge mismatch 已由独立登记、非重跑的 R004H6A exact-teacher audit 处置：384/384 hashes、teacher projections 与 metric bundles 全部通过，并复现 `SUPPORTED`；阈值、tolerance、数据、模型和结果均未改变。规定的独立 reviewer backend 仍不可用，状态 `REVIEW_UNAVAILABLE`。不进行第三次运行；R004H7 只解锁预注册，决策收益继续 HOLD。完整记录：`experiments/R004H6/RESULTS_R1.md`、`MECHANICAL_AUDIT_R1.md`、`R004H6A_AUDIT_RESULT.json`、`EXPERIMENT_AUDIT.md`。

> **2026-10-24 PEA-derived method decision:** the audited PEA failure is treated as evidence that unsigned source-moment alignment can move task-critical directions, not as evidence that all adaptation is useless. A task-constrained PEA is specified only as a conditional backup: freeze the planner candidates, project a PEA proposal through decision-margin constraints, and abstain when a lower-bound improvement is unavailable. It is not authorized for implementation before R004H6 and the read-only H7 decision-relevance diagnosis; the single primary thesis remains decision-quotient robustness. Full specification: `refine-logs/2026-10-24_aorta_iccv_positioning/PEA_DERIVED_METHOD_DECISION_CN.md`.

> **2026-10-24 ICCV baseline/novelty audit:** FlowDrive and READ are now mandatory closest-method references, but their published headline scores are not treated as direct wins/losses unless sensor, backbone, data, candidate, and scorer contracts are aligned. PEA is eligible only as a frozen driving-adapted TTA baseline; the current AutoVLA and ReCogDrive MG results remain respectively `test-time core adaptation` and `MG-style adaptation`, not strict official MG-Select reproductions. The provisional method thesis moves from graph quotient alone to a decision quotient defined by frozen candidate contrasts; this is not a qualified contribution until R004H6 passes and R004H7 shows that decision-contrast error predicts and reduces ranking flips beyond field MAE, SDF/BEV, direct edge + projection, direct pair, FlowDrive-style, and READ-style controls. Full audit: `refine-logs/2026-10-24_aorta_iccv_positioning/BASELINE_AND_NOVELTY_AUDIT_CN.md`.

> 状态：AORTA-v8.3.2 V0 的 Gate A 与 R004E-A/B/C 已完成。R004H5 首次给出经过机械复算的正向 fit 视觉机制证据：42 groups/672 anchors 全部完成，learned task MAE 约0.324m、zero 0.550m、teacher cosine约0.771，三种受控外观漂移约0.0087–0.0102m；一次性审查中 learned 在42/42组上优于 leave-one-group-out teacher prior。该结果只支持受控 fit 机制。R004H4P5R 在预注册30分钟上限内未完成首个哨兵，当前严格几何迭代因 computability 停止且无科学几何结果。下一步只允许冻结模型的独立视觉泛化资格实验；决策收益保持 HOLD。development、official calibration、test 与 target 仍封存，正式 coverage、真实天气/OOD、驾驶收益、Gate B 与创新性未成立。

## 1. 问题锚点

AORTA-Drive 研究 NAVSIM v1/v2 上冻结端到端规划器的 source-only appearance robustness。对同一物理场景、驾驶上下文与合法候选域，外观变化不应改变完整任务对 native 与 challenger 的真实相对评价。

核心问题是：

> 能否从合法外观干预中学习evaluator-aligned、任务充分的native-relative relation，并只在同时证据
> 支持真实相对改善时修改冻结planner，从而在NavDream与多类planner上减少困难失败，同时保持
> clean非劣与实时可用？

道路场、SDF、BEV 与 one-frame score 只保留为先验、对照或诊断对象。项目长期目标与统一验收合同见 `docs/AORTA_RESEARCH_CONSTITUTION_V1_CN.md`。

## 2. 核心假设

设合法外观观测为 \(X^a=g_a(S)\)。令R005资格化后的训练目标为
\(T^\dagger\in\{U_{ctx},Q_{\pi_{ref}}\}\)，最终估计对象为

\[
\Delta^*_{i0}(S,h,\pi_{ref})
=T^\dagger(S,\tau_i,h,\pi_{ref})-T^\dagger(S,\tau_0,h,\pi_{ref}).
\]

当前候选机制假设：完整时域signed differential support \(\nu_{i0}=\mu_i-\mu_0\)能提出候选差异
查询，全局视觉证据能回答该查询，反对称relation head能比通用matched scorer更稳定地恢复外观
干预下的evaluator-aligned margin；同时误差集合再把估计误差连接到regret与native-preserving arbitration。

该假设可能失败：差异支持可能没有增量信息，全局视觉仍可能不可识别，`U_ctx`可能不足以代表
two-stage action value，固定continuation又可能不足以代表部署策略；matched D-JEPA/World4Scorer-style
scorer可能同样有效，同时区间也可能因OOD失覆盖而全拒判。候选供给、关系预测、选择损失、拒判与最终行为必须分别报告。

## 3. 冻结范围

- NAVSIM v1/v2；
- frozen visual backbone 与至少三类被审计 planner；
- source-only train/development/calibration，最终 test 与 NavDream/target 按角色封存；
- 合法ragged candidate domain、native-relative scoped `U_ctx`；若反事实资格通过则使用固定continuation `Q_pi_ref`；
- 当前候选网络：完整时域差异支持 query、全局视觉 key/value、反对称 relation head、最小 previous-action context；
- 默认保留 native，只有同时 lower bound 严格为正才授权替换；
- batch-1 P50/P95、显存、候选扩展成本与 clean non-inferiority 均为最终硬门。

## 4. Non-goals

- 不恢复完整 HD map 或绝对道路安全零点；
- 不联合训练或修改原 planner；
- 不把任一道路 proxy 等同于完整官方任务或真实道路安全；
- 不把 source conformal coverage 写成任意 appearance OOD 保证；
- 不把 DINO、SDF、BEV、cross-attention、conformal、difference constraints 或轨迹条件化单独宣称为创新；
- 不把 development 写成最终未触碰测试集，也不以事后筛样、改阈值或删失败组制造主张。

## 5. 当前条件理论链

\[
\text{合法外观等价类}
\Rightarrow
\text{可识别的 evaluator-aligned relation}
\Rightarrow
\text{同时 margin 误差与 }2\varepsilon\text{ regret bridge}
\Rightarrow
\text{positive lower-bound intervention}
\Rightarrow
\text{OOD hard-failure net reduction + clean no-harm + realtime}.
\]

既有结果只支持旧视觉场在 held-out development 上包含非平凡视觉任务信息，以及若干数学/软件接口正确；不证明新 relation head、NavDream、跨 planner、驾驶收益、clean no-harm、实时性或网络创新。

## 6. 验收门

1. **Estimand gate:** B4R1 先验证合法 one-frame 接口与信号；接口有效后独立核验 evaluator scope
   并 materialize contextual target。只有 contextual oracle 也无非平凡机会才停止当前 panel；
2. **Representation gate:** correct image necessity、未见组 relation 泛化与非坍缩同时成立；
3. **Novelty gate:** differential-support relation 在 matched D-JEPA/World4Scorer-style 与普通 scorer 上取得可靠增量，否则删除架构创新；
4. **Robustness gate:** NavDream 各外观上 relation/regret/false intervention 不系统性崩溃；
5. **Behavior gate:** 至少三类 planner 上报告总体收益，并要求 \(G_{\mathrm{net}}\) 的预注册组级置信下界为正或超过最小有意义效应，同时分离 candidate supply 和 selection；
6. **No-harm/runtime gate:** clean group-level non-inferiority 与 batch-1 P50/P95 预算同时通过；
7. **Integrity gate:** 角色封存、预注册、机械重算、独立审查与 claim-evidence 映射闭合。

任一主门失败先降低相应主张；不得用新增模块、事后阈值或子集替代原始裁决。

## 7. 当前裁定

- 长期理论与验收合同：已冻结为可证伪研究宪法，不等于方法通过；
- 旧视觉场 held-out mechanism：有支持，但不能回填新 relation 网络；
- 旧 scalar road-margin reranker：已停止；
- 当前 differential-support relation：反例驱动的待证候选，查新后创新范围显著收窄；
- 最近实验门：R005E1A资格通过，R005E1B完成并GO到visual recoverability预注册；R005E1C因官方two-stage mapping覆盖`0/672`而fail-closed；R005E2A outcome-free门已全部闭合：6-fold cross-fit、S1主donor、S2/S3敏感性donor、三臂参数/FLOPs、672-anchor DINO exact cache与448-step单fold资源演练均机械PASS。用户已授权进入冻结视觉增量实验，但当前仓库尚缺正式scientific runner与独立auditor，因此只GO到R005E2B runner冻结，不得先读取结果；
- B4R1 interface valid 后顺序：target-scope audit/contextual materialization → matched scorer →
  NavDream → actual-history/sequential evaluation → 三 planner 行为 → clean no-harm → batch-1 latency；
- 当前投稿裁定：`NOT_ICCV_READY / NOT_ORAL_READY`。
