# AORTA-Drive Research Brief

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

AORTA-Drive 研究 NAVSIM v1/v2 上的 source-only appearance robustness。对同一物理场景与驾驶意图，外观变化可能改变冻结端到端 planner 的输出，却不应改变道路可行结构。

核心问题是：

> 能否学习一个与候选无关的道路差分场，在绝对安全零点不可靠时，为任意冻结 planner 的 challenger 相对 native 提供 nominal road-margin non-degradation 证书，并只在证书成立时干预？

## 2. 核心假设

完整道路 signed-distance field 的共同 offset 可能在视觉退化下不可辨识，但局部 clearance differences 可能仍可学习。若 source session 上的 graph-gradient residual 能被同时校准，则一个 field coverage event 可以覆盖任务图内所有自适应 swept-set pairs，并经 challenger 外覆盖/native 内点的非对称几何合同，认证每次授权干预达到冻结的相对改进阈值 \(\gamma_{task}\ge0\)。

该假设可能失败；遮挡、全局 calibration maximum、长路径累积和图错设都可能令证书失去非平凡性。即使证书有效，冻结 candidate provider 也可能没有提供更优候选；candidate supply 与 certificate recall 必须分别报告。

## 3. 冻结范围

- NAVSIM v1/v2；
- frozen DINOv3-H+ 与被审计 planner；
- source-only train/development/calibration；
- 一个具有冻结多相机节点投影/缺失证据协议的 candidate-free node-potential decoder；
- V0 fixed physical scale control 与 v8.4 candidate-independent observation-and-edge scale candidate、single native、exact multi-source solver；
- 预注册 ego-centric ROI、grid、continuous teacher/map error ledger、geometry checker、candidate provider、资格层与 session time set；
- 默认保留 native，仅认证 relative nominal-road non-degradation。

## 4. Non-goals

- 不恢复完整 HD map 或绝对道路安全零点；
- 不联合训练或修改原 planner；
- 不把道路 margin 等同于碰撞、规则、舒适性、真实执行或完整闭环安全；
- 不把 source conformal coverage 写成任意 appearance OOD 保证；
- 不把 DINO、SDF、split conformal、difference constraints、Dijkstra 或 min-plus transform 单独宣称为创新；
- 不在理论/idea 阶段启动 pilot 或长跑实验。

## 5. V0 条件理论链

\[
\text{source-session edge event}
\Rightarrow
\text{all-pair discrete robust bounds}
\Rightarrow
\text{map/geometry/numerical correction}
\Rightarrow
\text{all authorized interventions reach }\gamma_{task}.
\]

既有观察只支持继续研究任务空间接口，不证明 v8.3.2 的局部差分假设、校准锐度、系统收益或创新性。

## 6. 验收门

1. **Gate A:** 数学 oracle、multi-source 实现、geometry checker 和数值误差方向全部通过；
2. **Gate B:** 合法 source-only predictor 产生可接受的 session-tail residual 与非平凡米制证书，并分离 candidate supply、certificate recall 与 selection loss；
3. **Gate C:** 收益存在于具有连续几何意义的 non-inclusion candidates；
4. **Gate D:** 在同一 field event 下迁移到未见 candidate policy/frozen planner；
5. 完成 uncertainty-only 与 end-to-end 两层公平 baseline；
6. 完成独立 novelty-check。

Gate A/B 失败则不进入完整评测；公平 baseline 或查新失败则降级论文定位，不追加模块挽救主张。

## 7. 当前裁定

- V0 条件理论规范：可冻结；
- Gate A primitive/生成器：通过；R004 anchor-aligned teacher instantiation：R004E-A 通过；
- 实际系统：未验证；
- 顶会方法原创性：未授权；
- 下一阶段 A（独立视觉泛化资格）：冻结 R004H5 的 checkpoint、外观算子、metrics 与阈值，只允许在预注册的未见 source recording groups 上一次性检验任务保持；同时冻结错配图像/场景置乱与输入/DINO 扰动强度账本。不得用该阶段调模型，也不得把受控外观命名为真实天气。
- 下一阶段 B（决策收益）：HOLD。R004H4P5R 已在30分钟资源上限内停止且没有完整哨兵结果；当前不恢复细网格、候选修改或128-anchor扩展。只有独立视觉泛化与可计算的决策接口分别闭合后，才注册驾驶收益实验。
