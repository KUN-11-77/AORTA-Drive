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
再一次性检查development coverage。其H7BS-0 post-hoc归因阶段已经完成并通过独立机械
复算：全部ROI失败均为晚时刻前向`x_max`越界，origin合同闭合，且invalid样本明显偏向
长前向行程。完整结果见`H7BS0_RESULTS.md`。H7BS1随后以fit-only规则选择2.5秒为最大
共同可行前缀，并在24/24 development groups上一次性通过支持资格：360/384 anchors有效，
机械复算0失败。完整记录见`H7BS1_RESULTS.md`。该结果只证明2.5秒prefix可查询，不证明
完整候选的决策信息充分；H7-B2继续HOLD，必须先完成H7BS2候选晚期分叉、prefix
node-set不可区分性与完整时域ROI诊断。H7-C训练仍HOLD。

H7BS2现已完成并否定2.5秒最终路线：36/42 fit groups中存在128个prefix节点集合相同、
完整集合不同的精确反例。注册的panel-level成本规则虽选出前移64×64，但它仍拒绝
290/3864 eligible challengers；按候选无关算子的完整定义域要求，首个fit候选是前向
96×64，保留3864/3864 queries。完整记录见`H7BS2_RESULTS.md`。只允许下一步H7BS3对
该单一ROI做development支持资格；网络训练、H7-B2与H7-C继续HOLD。

H7BS3R1现已在24个historically exposed development groups上通过一次性完整域资格：
384/384 native与2206/2206 eligible challengers全部受前向96×64支持，24/24 groups达到
12/16门槛，机械复算0失败。9个无eligible challenger的anchors属于候选供给而非ROI
失败。完整记录见`H7BS3_RESULTS.md`。该结果只解锁视觉可观测性与真实时延协议设计；
网络训练、H7-B2与H7-C仍HOLD。

H7BS4随后区分了空间包含与视觉证据：B0B在fit teacher上确认596/4536 candidates、39/42
groups存在unsupported possible minimizer，说明missing evidence可能控制set-min。B1进一步用
最强oracle比较joint Lipschitz、shared-node box与candidate-independent intervals；机械重放通过，
但三者都只识别1301/3864个有符号pair，joint相对box的组级增益为0且仅5/42 groups达到覆盖门。
裁决为`ORACLE_SUPPORT_SET_NEAR_VACUOUS_REDESIGN_INTERFACE`。H7-C继续HOLD；下一步只允许
fit-only H7BS4B2归因2219个exact-zero teacher contrasts，不启动新网络训练。
