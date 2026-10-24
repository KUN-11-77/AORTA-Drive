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
| C15 | source-only native-conditioned preference labels 合法且当前panel存在正oracle ceiling | R004H7BS4B4两次完整materialization、panel exact reconstruction、candidate-specific official one-frame scorer、42-group oracle gain | 接口不一致、零/稀疏偏好、bootstrap oracle gain下界不为正或少于21组存在native improvement | preregistered；运行中，不是完整official PDMS或learned evidence |
| C16 | exact coboundary preference head 相对自由pair head具有必要的结构优势 | label/oracle gate通过后另行预注册的matched-capacity、matched-data、three-seed study；cycle residual、regret、appearance flip、planner transfer与latency | direct-pair/GTRS-style/DA-WAM-style control等价或更好，或优势来自参数/算力/标签差异 | proposed hypothesis only；B4通过前禁止训练 |
| C17 | 同时误差集合上的lower-bound仲裁可条件性地保证native improvement并在证据不足时拒判 | 明确 simultaneous event 的证明；独立calibration coverage/sharpness；false-improvement与coverage；OOD失覆盖反例 | 置信事件不同时覆盖候选、lower bound实现错误、coverage在新planner/appearance下失效却仍宣称保证 | conditional theorem candidate；source calibration不得外推为任意NavDream guarantee |

## Claim language policy

- `guarantees` 只用于完整陈述假设的定理；
- `shows` 只用于已注册、可追溯结果；
- `safe` 必须附带 nominal road constraint 范围；
- source coverage 不写成 arbitrary-OOD coverage；
- V0 规范冻结不写成系统已验证或论文原创性成立。
