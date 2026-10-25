# R005E1B 实验完整性审计

日期：2026-10-25  
审计对象：`experiments/results/R005E1B_1E5362E/`  
冻结源码：`1e5362ef6d65dd7e568466317ed347aef312fa75`  
总体裁决：`WARN`  
科学裁决：`SUPPORTED_ON_SOURCE_FIT_NATIVE_HISTORY_UCTX__GO_TO_VISUAL_RECOVERABILITY_PREREGISTRATION__NO_TRAINING_YET`

## 1. 审计结论

没有发现数字不匹配、删除zero-challenger、将网络预测伪装成ground truth、隐藏raw/filtered差异或将`U_ctx`冒充完整PDMS的证据。primary/replay逐anchor ledger、summary和decision精确一致；本地复算、归档哈希、有限值与封存角色字段均通过。

本审计不能给无保留`PASS`，原因有三项：

1. mechanical audit从保存的components复算filtered score、`U_ctx`、聚合统计和裁决，但没有从原始simulation state arrays独立重算two-frame EC；primary与replay也共享生产实现；
2. 冻结官方源码快照及其哈希与E1A账本一致，但当前归档不足以独立证明正式运行时实际import的模块路径/hash，也不能由程序自报布尔量单独证明系统级从未访问封存数据；
3. 外部cross-family reviewer后端不可用，本次为新上下文审查者的独立核验，不能声称满足跨模型家族复核。

这些限制不要求自动重跑R005E1B；它们限制可写主张，并进入R005E2预注册的provenance与独立复算要求。

## 2. 六维审计

| 维度 | 裁决 | 证据与边界 |
|---|---|---|
| A. Ground-truth provenance | `WARN` | 监督来自冻结NAVSIM simulator/scorer与metric cache，不是待训练模型输出；但runtime import来源和raw-log immediate predecessor无法由归档完全重建。 |
| B. Score normalization | `PASS` | `filtered_one_stage`与`U_ctx`分母固定为14与16；保留官方候选相关progress归一化语义，未按本方法表现事后缩放。 |
| C. Result existence | `PASS_WITH_DOCUMENTATION_FIX` | 42 groups、672 anchors、4,536 candidates、13,402 pairs及所有主数字独立吻合；陈旧状态文字已在本提交修正。 |
| D. Dead code / verification | `WARN` | 核心路径实际调用并保存；但audit复用生产函数，EC未由原始states独立复算，`sentinel_replay_exact`字段不能替代完整replay证据。 |
| E. Scope | `WARN` | 仅42个source-fit groups；group bootstrap不是672-anchor独立样本，也不是多seed。sealed-access字段是运行声明，不是系统级访问证明。 |
| F. Classification | `PASS` | `simulation_only / evaluator-derived proxy`；不是真实道路安全GT、human evaluation或完整pseudo-closed-loop value。 |

## 3. 已核验数值

| 指标 | 独立核验值 |
|---|---:|
| Groups / anchors | `42 / 672` |
| Candidates / eligible challengers | `4536 / 3864` |
| Pairs / zero-challenger anchors | `13402 / 15` |
| Group-equal mean contextual oracle gain | `0.01665036604823182` |
| Group bootstrap 95% CI | `[0.008772239329326128, 0.02620335071293309]` |
| Groups with strict native improvement | `42/42` |
| Filtered/context top-set disagreement | `252/672` |
| Filtered/context pair-sign disagreement | `1673/13402` |
| Group-equal mean ignored-context regret | `0.01574054136529422` |

## 4. Claim impact

- `SUPPORTED`：冻结source-fit、native-history账本存在上述`U_ctx` contextual oracle headroom；ragged候选域、42×16范围和15个zero-challenger anchors均保留；raw/filtered相对contextual存在报告的排序分歧。
- `NEEDS_QUALIFIER`：运行时source binding、即时上一帧与sealed-access只能说冻结账本/自报字段通过；不能说获得系统级独立证明。
- `NOT_ESTABLISHED`：某一种previous-action token编码对任何learner的必要性；必须由current-only/minimal-token/matched-temporal消融检验。
- `UNSUPPORTED / NOT_TESTED`：RGB可恢复性、Differential Query架构优势、NavDream、跨planner、完整`Q_{pi_ref}`、驾驶收益或安全。

## 5. 下一步约束

R005E2只能先完成预注册。正式运行前必须把runtime module provenance写入不可变产物，并明确视觉可恢复性实验的标签来源、split、correct/mismatch/removed控制、group-level统计和停止规则。若需要把EC复算升级为更强完整性保证，应新增独立raw-state verifier；不得改写本次R005E1B结果或用第三次运行覆盖现有裁决。
