# AORTA Final Validation Tracker

状态：**R005E1B_GO_SOURCE_FIT_UCTX__VISUAL_RECOVERABILITY_PREREGISTRATION_UNLOCKED__TRAINING_LOCKED**

| Run ID | Milestone | Purpose | System / Variant | Split | Primary Metrics | Priority | Status | Stop / Go |
|---|---|---|---|---|---|---|---|---|
| R004H7-BS4B4R1 | M0 | 合法raw one-frame oracle | official candidate-local scorer | source fit 42 groups | frozen oracle gates | MUST | COMPLETED_GO：42/42 nonzero、42/42 strict improvement、gain `0.0202745 [0.0117815,0.0301722]`；机械 PASS | 只支持source-fit pre-filter `F_raw`；不授权训练 |
| R005E0 | M1 | target-source scope audit | raw/filter/context/two-stage source alignment | source code + frozen B4R1 ledger | source fidelity、semantic gap | MUST | COMPLETED：source scope PASS；raw-filter equivalence FAIL | B4R1 GO保留但只适用于`F_raw`；禁止直接训练 |
| R005E1A | M1 | contextual target资格 | single evaluator root + history/candidate/mapping validity | source fit metadata | hashes、causal history、counterfactual validity | MUST | COMPLETED_PASS：672/672 causal histories与previous caches可用；官方two-stage mapping覆盖0/672 | `U_ctx`可物化；`Q_pi_ref` fail-closed |
| R005E1B | M1 | scoped `U_ctx` materialization | filter后components + native previous history | source fit | exact replay、headroom、raw/filtered/context flip-regret | MUST | COMPLETED_GO：gain `0.0166504 [0.0087722,0.0262034]`；42/42 strict-improvement groups；primary/replay/audit PASS | 只解锁视觉可恢复性预注册；不授权直接训练或行为主张 |
| R005E1C | M1 | fixed-continuation two-stage value | candidate-specific `Q_pi_ref` | source fit + official mapping | exact replay、`U_ctx`/`Q` disagreement | SHOULD | STOP_UNDEFINED_ON_SOURCE_FIT | 官方映射覆盖0/672；禁止近似填充，不影响scoped `U_ctx`子问题 |
| R005E2 | M2 | 视觉可恢复性资格 | trajectory/history-only、correct/mismatch/removed RGB | source train/held-out source groups | contextual relation、sign、regret、视觉增量 | MUST | PREREGISTRATION_UNLOCKED__NO_TRAINING_YET | correct RGB无独立增益则停止视觉主张 |
| R005A-PROXY | M2 | 简单代理基线 | trajectory/prior/field-SDF-BEV | source train/dev | relation、sign、regret | MUST | LOCKED_BY_R005E2 | 建立最低线 |
| R005A-GENERIC | M2 | 最强最近基线 | D-JEPA/World4Scorer-style matched scorer | source train/dev | relation、sign、regret、latency | MUST | LOCKED_BY_R005E2 | 必须先于主方法完成 |
| R005A-AORTA | M2 | 主方法 | differential evidence relation | source train/dev | relation、sign、regret、false intervention | MUST | LOCKED_BY_R005E2 | 不胜 generic 删除主创新 |
| R005A-ABL-SUPPORT | M2 | 结构必要性 | full/prefix、signed/unsigned、global/local | source train/dev | held-out regret | MUST | LOCKED_BY_M1 | 不支持则删对应模块 |
| R005A-ABL-CONTEXT | M2 | contextual/future-value input | current-only/minimal token/temporal/fixed continuation | source train/dev | `U_ctx`/`Q_pi_ref` regret、false intervention | MUST | LOCKED_BY_M1 | 最简单获胜者保留 |
| R005A-ABL-LOSS | M2 | correctness vs invariance | task/task+orbit/orbit-only | source train/dev | task error、drift、regret | MUST | LOCKED_BY_M1 | drift好但regret差判失败 |
| R005B-CONTROLS | M3 | 视觉必要性 | correct/mismatch/removed/trajectory-only | held-out source | paired error/regret | MUST | LOCKED_BY_M2 | correct无优势则HOLD |
| R005C-NAVDREAM | M3 | 未见外观泛化 | generic vs AORTA | sealed NavDream | group regret、sign、false intervention | MUST | LOCKED_BY_M2 | 崩溃则STOP鲁棒性主张 |
| R006A-REG | M4 | 回归式 planner 行为 | native/generic/AORTA | clean + OOD | PDMS、DAC、collision、G_net | MUST | LOCKED_BY_M3 | 分 planner 裁决 |
| R006A-DIFF | M4 | diffusion planner 行为 | native/generic/AORTA | clean + OOD | PDMS、DAC、collision、G_net | MUST | LOCKED_BY_M3 | 分 planner 裁决 |
| R006A-SELECT | M4 | scoring planner 行为 | native/generic/AORTA | clean + OOD | PDMS、DAC、collision、G_net | MUST | LOCKED_BY_M3 | 分 planner 裁决 |
| R006A-HISTORY | M4 | 历史 rollout 一致性 | native-history/actual-history/sequential | clean + OOD | contextual regret、false intervention、error propagation | MUST | LOCKED_BY_M3 | teacher-forced 不能替代 sequential |
| R006B-CAL | M5 | simultaneous calibration | fixed/learned scale + fallback | independent calibration/OOD | coverage、sharpness、risk–coverage | MUST | LOCKED_BY_M2 | 全拒判则删实证贡献 |
| R006C-LAT | M5 | 在线时延 | shared vs repeated encoding | deployment hardware | P50/P95/P99、memory、K scaling | MUST | LOCKED_BY_M2 | 超预算降级离线 |
| R006D-AUDIT | M6 | 最终完整性 | mechanical + scientific + claim audit | all frozen outputs | hash、recompute、claim fidelity | MUST | LOCKED | 有不一致不发布结论 |

## Tracker Rules

1. LOCKED 不得因“代码已经准备好”自动改为 RUNNING。
2. B4R1、R005E0、R005E1A与R005E1B已完成；R005E1C因官方mapping覆盖为零而停止。R005E2只解锁预注册，尚未授权训练。
3. 任一 run 开始前补齐 commit、config、input hashes、role ledger、seed、wall cap 与异常停止规则。
4. partial scientific metrics 不用于修改同一 run 的 threshold、model、candidate 或 split。
5. 失败 run 永久保留；修正版使用新 ID，不能覆盖原裁决。
6. tracker 记录状态，不代替对应 preregistration、result 与 audit artifact。
