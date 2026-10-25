# AORTA Final Validation Tracker

状态：**B4R1_FORMAL_RUNNING__LATER_RUNS_LOCKED**

| Run ID | Milestone | Purpose | System / Variant | Split | Primary Metrics | Priority | Status | Stop / Go |
|---|---|---|---|---|---|---|---|---|
| R004H7-BS4B4R1 | M0 | 合法 one-frame oracle | official candidate-local scorer | source fit 42 groups | frozen oracle gates | MUST | RUNNING：`7050c6e`，无固定 wall cutoff | interface valid 才进入 M1；科学 outcome 只裁决 one-frame |
| R004H7-BS4B5F | M1 | scoped contextual target materialization | evaluator-scope audit + augmented previous context | source fit | exact replay、contextual signal、flip/regret | MUST | LOCKED_BY_M0_INTERFACE_VALID | contextual 无信号才停当前 panel |
| R005A-PROXY | M2 | 简单代理基线 | trajectory/prior/field-SDF-BEV | source train/dev | relation、sign、regret | MUST | LOCKED_BY_M1 | 建立最低线 |
| R005A-GENERIC | M2 | 最强最近基线 | D-JEPA/World4Scorer-style matched scorer | source train/dev | relation、sign、regret、latency | MUST | LOCKED_BY_M1 | 必须先于主方法完成 |
| R005A-AORTA | M2 | 主方法 | differential evidence relation | source train/dev | relation、sign、regret、false intervention | MUST | LOCKED_BY_M1 | 不胜 generic 删除主创新 |
| R005A-ABL-SUPPORT | M2 | 结构必要性 | full/prefix、signed/unsigned、global/local | source train/dev | held-out regret | MUST | LOCKED_BY_M1 | 不支持则删对应模块 |
| R005A-ABL-CONTEXT | M2 | full-target context | current-only/minimal token/temporal | source train/dev | full regret、false intervention | MUST | LOCKED_BY_M1 | 最简单获胜者保留 |
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
2. B4R1 的 HOLD 只有用户明确授权才能解除。
3. 任一 run 开始前补齐 commit、config、input hashes、role ledger、seed、wall cap 与异常停止规则。
4. partial scientific metrics 不用于修改同一 run 的 threshold、model、candidate 或 split。
5. 失败 run 永久保留；修正版使用新 ID，不能覆盖原裁决。
6. tracker 记录状态，不代替对应 preregistration、result 与 audit artifact。
