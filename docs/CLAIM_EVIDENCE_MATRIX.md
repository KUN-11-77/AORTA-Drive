# Claim–Evidence Matrix

任何数字或比较在没有注册实验与不可变结果引用前都保持 `proposed`。

| ID | Intended claim | Required evidence | Falsifier | Status |
|---|---|---|---|---|
| C1 | source-session edge event 可同时覆盖声明时刻的全图差分 residual | 独立 session calibration/validation；simultaneous coverage、sample-size eligibility 与 sharpness | undercoverage、split reuse 或阈值空泛 | proposed |
| C2 | exact min-plus 实现与 robust set-min infimum 一致 | 小图枚举、LP/dual oracle、shift/zero-edge/inclusion cases | 任一方向错误或数值乐观 | Gate A synthetic pass |
| C3 | map/geometry/numerical ledger 给出有效 continuous correction | continuous teacher、node-label error、解析 geometry bound、protocol hash 与数值预算 | 任一通过 gate 的乐观反例 | Gate A PASS：legacy route falsified；R002E raw-cover/rotation-free geometry、R002F exact transform、R002G 164 caches × 4096 nodes 完整账本通过 |
| C4 | local graph 在相同合法 predictor/coverage 下比 absolute/global gauge 更紧 | Layer-1 uncertainty-only control，exact readout | 优势消失或仅来自 baseline handicap | proposed |
| C5 | 完整 edge-trained system 优于合法 absolute/global/CDF 最近邻 | Layer-2 matched-capacity/source comparison | 等价或更差 | proposed |
| C6 | 所有授权干预达到 \(\gamma_{task}\) relative improvement，并产生 meaningful non-inclusion utility | session \(R_\gamma\)、\(S_{better}\)、\(R_{cert|better}\)、\(R_{opportunity}\)、\(R_{selected}\) | relative failure、供给近零、证书近空或只靠 inclusion | proposed |
| C7 | candidate-free event 迁移到未见 candidate policy/frozen planner | 同一校准事件下 held-out policy/planner test | coverage/utility 随候选机制崩溃 | proposed |
| C8 | 运行代价可接受 | 冻结图规模上的 field 与 multi-source profiler | 不满足预注册资源门 | proposed |
| C9 | graph potential differences 是 appearance transfer 的相关机制 | paired same-scene \(D_{app}\)、task error、non-collapse controls、session-tail residual 与 empirical target coverage | 稳定只来自塌缩、task error 不优于平凡场、\(D_{app}\) 与证书退化无关或目标域严重失效 | supported on held-out development：R004H5 fit 机制已机械 PASS；R004H6R1 在 24 groups/384 anchors 上同时通过 held-out task 与 correct-image necessity，clean task MAE `0.3699m`、fit-prior `0.4532m`、zero `0.5543m`，mismatch-minus-correct `+0.1478m [0.1391,0.1569]`，removed-minus-correct `+0.1855m [0.1681,0.2029]`。R004H6A exact-teacher audit 在不重跑推理、不修改阈值或 tolerance 的条件下通过 384/384 hashes、projections 与 metric bundles，并复现 `SUPPORTED`。外部 reviewer unavailable；未触碰最终 test/target，也无 coverage/driving/safety 证据 |
| C10 | observation-conditioned bounded edge scale 可在冻结 potential 上被稳定学习 | proper-loss fit、cap diagnostics、pooled/uniform controls 与后续 common-denominator task endpoint | learned proper loss 不优于 uniform、边界塌缩或 learned 不优于 pooled | partial：raw-interface H2 falsified；gauge-repaired H2R1 `0.001564 < 0.380000` 且 zero cap；pooled/task endpoint 未验证 |
| C11 | frozen Stage-P activation magnitude is the dominant conditioning defect in first H2 | fixed-anchor exact gradients and non-mutating first-step raw/RMS counterfactual | fewer than 6/8 activation-scale votes or normalization does not remove first-step cap entry | supported for registered instance：R004H2D 8/8 votes; not yet a scale-quality claim |
| C12 | mean proper-scale fit translates into sharper simultaneous calibrated budgets | independent session-max proxy thresholds and common-denominator task endpoint | rank-max threshold cancels scale gains or learned utility ≤ controls | proxy warning：learned median budget `5.7849 m` > uniform `2.6315 m`; task-level status unresolved until H4 |
| C13 | exact decision-contrast error 比全局 field/DINO error 更贴近候选 ranking flip 与 regret | R004H7A theory oracle；完整时域、合法支持域上的read-only shadow diagnosis | 数学 oracle 失败、panel coverage不足、前缀丢失候选区分信息、直接视觉证据接口失败或组级关联不优于 controls | unsupported for current scalar-min family：B1 joint uncertainty无额外符号识别；B2显示2033/2219零差异由共同min支配；B3 temporal first-fork只恢复170/2033，组级CI 0.0634–0.1029，spatial-exclusive适用率仅0.7713。机械audit PASS，但general road-margin reranker STOP；尚未读取outcome，故新decision-sufficient对象与ranking关联均未建立，H7C HOLD |
| C14 | 旧 quotient-field scalar reranker 在公平容量与预算下减少候选 ranking regret/flip | 原计划R004H7C six-arm study | B1/B2/B3 oracle chain显示该readout在训练前已不具一般决策充分性 | retired before training：B3后禁止恢复旧QP-DQ训练；后续不得用新preference结果回填该主张 |
| C15 | source-only合法候选域上的candidate-local one-frame score differences合法且当前panel存在正oracle ceiling | 候选membership、score domain与统计分母一致的两次完整materialization；每条轨迹独立在固定PDM-baseline proposal set中评分；`PROGRESS_MODE=eval`；42-group oracle gain | 接口/候选域/环境模式不一致、零/稀疏偏好、bootstrap oracle gain下界不为正或少于21组存在native improvement | B4永久INVALID：48 anchors含168个intent-ineligible challengers且正向diagnostics不可解释。R1已完成合法域、scorer语义和progress-mode outcome-free server preflight，但formal primary/replay/audit尚未授权或运行；不是完整official PDMS或learned evidence |
| C16 | complete-graph relational structure相对native-vector/star-score/free-edge controls具有必要的held-out优势 | 合法域label/oracle通过后，容量、FLOP、数据、optimizer、seed匹配的native-vector/star-score/complete-score/free-complete-edge研究；主要看held-out regret与selection error，cycle residual仅诊断 | star上exactness已证明vacuous；若star-score≈native-vector且complete structure无可靠held-out优势，或收益来自更多edges/参数/算力，则删除Hodge/结构贡献 | theory warning：native-star上任意edge vector均可积，score difference只是重参数化；complete edges又是由native-conditioned scalar代数派生。B4R1通过前禁止训练，当前不具结构创新证据 |
| C17 | 同时误差集合上的lower-bound仲裁可条件性地保证native improvement并在证据不足时拒判 | recording-group max residual的split calibration；固定模型/候选机制；coverage/sharpness、false-improvement与OOD失覆盖反例 | anchor伪独立、split复用、置信事件不同时覆盖候选、lower bound实现错误、coverage在新planner/appearance下失效却仍宣称保证 | conditional theorem candidate：有限样本次序统计量已闭合；若未来独立calibration有42 groups，可给有限95% group-simultaneous quantile，但无假设99%只能无穷半径/全拒判。当前未取得该calibration证据，source coverage不得外推为NavDream guarantee |
| C18 | 合法候选域上的native-relative relation可同时保留任务信息并删除无关外观变化 | 合法task loss、paired-appearance relation drift、correct-vs-mismatch/removed、held-out/NavDream ranking regret；同一ragged domain贯穿label/loss/inference | 常数或先验达到同等结果、correct image无增益、relation稳定但regret不降、候选域污染或OOD外观下失真 | formal estimand only：因果/信息论目标与anti-collapse反例已写明；R004H6只支持旧field的视觉必要性，不能回填新relation head。B4R1与后续训练/泛化尚未执行 |

## Claim language policy

- `guarantees` 只用于完整陈述假设的定理；
- `shows` 只用于已注册、可追溯结果；
- `safe` 必须附带 nominal road constraint 范围；
- source coverage 不写成 arbitrary-OOD coverage；
- V0 规范冻结不写成系统已验证或论文原创性成立。
