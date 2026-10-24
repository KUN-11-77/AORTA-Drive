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
| C13 | exact decision-contrast error 比全局 field/DINO error 更贴近候选 ranking flip 与 regret | R004H7A theory oracle；完整时域、合法支持域上的read-only shadow diagnosis | 数学 oracle 失败、panel coverage不足、前缀丢失候选区分信息或组级关联不优于 controls | inconclusive；2.5秒最终路线已否定。H7BS3R1现支持前向96×64完整离散域：development上384/384 native与2206/2206 eligible challengers无越界，24/24 groups过门；但视觉可观测性、时延与C13关联终点尚未检验。H7-B2/H7-C HOLD，外部reviewer unavailable |
| C14 | decision-quotient supervision 在公平容量与预算下减少候选 ranking regret/flip | R004H7C six-arm three-seed matched study；potential/projected-edge/direct-pair明确拆分 | QP-DQ 不优于 edge-only/SDF/direct-pair，或优势来自容量/选择不匹配 | HOLD；C13尚未通过，H7BS1只解决2.5秒prefix定义域，禁止训练 |

## Claim language policy

- `guarantees` 只用于完整陈述假设的定理；
- `shows` 只用于已注册、可追溯结果；
- `safe` 必须附带 nominal road constraint 范围；
- source coverage 不写成 arbitrary-OOD coverage；
- V0 规范冻结不写成系统已验证或论文原创性成立。
