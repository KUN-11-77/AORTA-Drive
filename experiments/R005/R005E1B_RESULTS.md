# R005E1B：source-fit contextual oracle 结果

日期：2026-10-25。

冻结提交：`1e5362ef6d65dd7e568466317ed347aef312fa75`。

裁决：`SUPPORTED_ON_SOURCE_FIT_NATIVE_HISTORY_UCTX__GO_TO_VISUAL_RECOVERABILITY_PREREGISTRATION__NO_TRAINING_YET`。

## 1. 执行完整性

- primary 与独立完整 replay 均覆盖 42 个 source-fit recording groups、672 anchors、4,536 个合法候选和 13,402 个候选对；
- 3,864 个 eligible challengers 被评分，15 个 zero-challenger anchors 保留在完整分母中，ineligible candidate score count 为 0；
- primary/replay 的逐 anchor ledger 精确一致；机械审计从已保存components重新计算 4,536 个候选的 filtered one-frame score 与 contextual utility，summary 和裁决均精确一致；EC未从原始state arrays独立重算；
- 两份完整结果均无 NaN/Inf，退出码为 0，归档 SHA-256 全部复核通过；
- development、calibration、test、NavDream 与 target 均未访问；`Q_{\pi_{ref}}` 未被构造，也未声明完整 official pseudo-closed-loop PDMS。

## 2. 冻结主结果

| 指标 | 结果 |
|---|---:|
| Group-equal mean contextual oracle gain | `0.0166504` |
| 42-group bootstrap 95% CI | `[0.0087722, 0.0262034]` |
| 存在 nonzero relation 的 groups | `42/42` |
| 存在 strict native improvement 的 groups | `42/42` |
| Group-equal nonzero-pair fraction | `0.6318311` |
| 42-group bootstrap 95% CI | `[0.6028049, 0.6596517]` |
| Filtered one-frame 与 contextual top-set disagreement | `252/672 = 37.50%` |
| Filtered one-frame 与 contextual pair-sign disagreement | `1673/13402 = 12.4832%` |
| Raw one-frame 与 contextual top-set disagreement | `279/672 = 41.5179%` |
| Raw one-frame 与 contextual pair-sign disagreement | `2128/13402 = 15.8782%` |
| 忽略 context 的 group-equal mean regret | `0.0157405` |
| Native hard-failure anchors / hard rescues | `60 / 8` |

预注册 GO 门全部通过：42/42 groups 可定义、严格改善组数高于 21、oracle-gain 置信下界严格大于 0，且信号不是通过删除 zero-challenger 或 complete-case 筛选产生。

## 3. 科学含义

本实验支持两个受限结论：

1. 在冻结的 source-fit 合法候选 panel 上，native-history teacher-forced contextual row utility `U_ctx` 存在非平凡、可复算的候选选择空间；
2. 对登记的`U_ctx`而言，previous-action context不是可随意删除的目标自变量。即使先做 official human filter，忽略 two-frame context 仍改变 `37.50%` anchors 的 top set，并在 `12.48%` 候选对上改变关系符号。

因此，最终 relation learner 的最低合法监督对象应是 `U_ctx`。因果 previous-action geometry token是下一阶段要与history-only/current-only及更复杂temporal encoder做matched消融的最低成本实现；本实验未证明该特定编码形式对任何learner都必然必要。raw或filtered one-frame score只能保留为消融和低成本诊断，不能再作为`U_ctx`的等价监督替代品。

## 4. 严格边界

本结果不证明：

- RGB 或冻结视觉 backbone 能恢复 `U_ctx`；
- differential-support query 优于容量、数据与 FLOP 匹配的通用 scorer；
- `U_ctx` 等同于 fixed-continuation `Q_{\pi_{ref}}`、完整 pseudo-closed-loop value 或策略级 `J(\pi)`；
- NavDream 泛化、跨 planner 行为收益、clean no-harm、实时性或驾驶安全。

E1A 已证明官方 two-stage mapping 对本 source-fit current/previous pair 的覆盖为 `0/672`，故 `Q_{\pi_{ref}}` 分支按预注册 fail-closed；禁止用插值、近似 continuation 或当前 row 结果冒充完整两阶段价值。

## 5. 下一步

只解锁 R005E2 的预注册，不直接训练：

1. 冻结 `U_ctx` label bundle、ragged candidate membership、previous-action token 与 split roles；
2. 先做 trajectory/history-only、correct RGB、mismatched RGB、removed RGB 的视觉可恢复性实验；
3. correct RGB 在 held-out source groups 上必须显著优于 trajectory/history-only 与 mismatch/removed，且改善集中于 near-tie、context-flip 与 hard-failure strata；
4. 该门通过后，才开展 Differential Query 对 matched D-JEPA/World4Scorer-style full-context scorer 的核心结构对照。

原始归档：`experiments/results/R005E1B_1E5362E/`。

独立完整性复核为`WARN`而非无保留PASS，限制见`R005E1B_EXPERIMENT_AUDIT.md`。该限制不改变冻结数值或GO到预注册的裁决，也不授权直接训练。
