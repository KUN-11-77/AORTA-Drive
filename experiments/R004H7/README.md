# R004H7 — Decision-Quotient Structure Necessity

R004H7 检验 AORTA 的核心结构是否必要，而不是继续优化 R004H6 的视觉 MAE。

执行顺序固定为：

1. H7-A：数学资格与无数据 oracle；
2. H7-B：冻结模型的只读 shadow decision diagnosis；
3. H7-C：仅在 A/B 通过后启动容量匹配训练。

H7-A 已通过无数据 oracle 与机械复核。H7-B 已在冻结的24-group development panel上
完成，但因只有265/384 anchors有效、11/24 groups达到每组至少12个有效anchors的门槛，
按预注册裁决为 `INSUFFICIENT_PANEL_COVERAGE`；相关性与AUROC未计算。机械重算384/384
panels通过，外部reviewer不可用。H7-C不得训练。完整合同和结果见 `THEORY.md`、
`PREREGISTRATION.md`、`RESULTS_H7A.md`、`H7B_IMPLEMENTATION_QUALIFICATION.md` 与
`H7B_RESULTS.md`。

H7-B coverage failure后的唯一已注册下一步是
`H7BS_SUPPORT_ADEQUACY_PLAN.md`：先在fit groups上以outcome-blind规则选择最大固定前缀，
再一次性检查development coverage。当前仅完成设计，implementation与payload均HOLD。
