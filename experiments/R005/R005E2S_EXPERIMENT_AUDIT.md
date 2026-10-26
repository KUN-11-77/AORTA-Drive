# R005E2S 实验完整性审计

日期：2026-10-26

机械审计：`PASS`

科学分类：`RELATION_ONLY_BINDING_SECONDARY`

父裁决：`STOP_CURRENT_VISUAL_MAIN_CLAIM`（不可变）

完整性状态：`WARN`（外部 cross-model reviewer backend 当前不可用）

## A. Ground-truth provenance：PASS（限定作用域）

target与candidate mask来自R005E2B冻结、可复算的source-fit `U_ctx` bundle；不由R005E2S预测生成。
本轮只替换visual donor identity。`U_ctx`仍不是完整NAVSIM two-stage policy value。

## B. Score normalization：PASS

使用未归一化relation MSE和decision regret；bootstrap作用于42个recording-group配对差值。DINO RMS
与donor geometry distance只作干预强度账本，不进入阈值、donor选择或科学分类。

## C. Result existence and replay：PASS

6 primary与6 replay jobs全部存在并完成；预测数组bit-exact；同fold绑定同物理GPU；parent checkpoint
文件与tensor state逐个重验；run目录SHA manifest在服务器与本地均通过。

## D. Dead-code / metric execution：WARN

冻结审计器实际生成登记的MSE、regret、42-group CI、donor距离、top-choice flip与四分支分类。由于规定的
独立cross-model reviewer工具当前不可用，未宣称完成外部逐行dead-code审查；内部机械审计不能冒充
独立科学复核。

## E. Scope：WARN，但主张已收缩

范围仅为42个source-fit groups、672 anchors、冻结V2 checkpoints和两个预构造donor规则。没有新训练、
独立未见场景、NavDream、跨planner、sequential、clean no-harm或在线时延结果。

## F. Evaluation classification

`simulation_only + official_evaluator_derived_contextual_target + source_fit_group_OOF + frozen_checkpoint_input_intervention`。

## 总裁决

- 实现、数据、重放和本地复算完整性：`PASS`；
- 外部独立完整性复核：`REVIEW_UNAVAILABLE`，总完整性为`WARN`；
- 二次科学分类：`RELATION_ONLY_BINDING_SECONDARY`；
- 父实验：继续`UNSUPPORTED / STOP_CURRENT_VISUAL_MAIN_CLAIM`；
- 不允许第三次donor搜索或用本结果挽救父门。

## Claim impact

- “scene-bound平均relation信号对S2/S3 donor规则稳健”：`SECONDARY_SUPPORTED`；
- “correct visual降低decision regret”：`UNSUPPORTED`；
- “Differential Query/外观鲁棒/驾驶收益”：`UNTESTED`。
