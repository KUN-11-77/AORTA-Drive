# R004H7-BS4A：fit候选投影可观测性结果

## 裁决

`STOP_OR_REDESIGN_VISUAL_SUPPORT_INTERFACE`。机械执行成功，但严格科学门失败；不得用
`2/4536`很小或全网格平均支持率较高为理由改成PASS。

## 结果

- 42 fit groups、672 anchors、4536个native/eligible candidate queries全部记账；
- 42/42 groups达到继承的至少12/16 anchors组级门；
- 2个候选的全部查询节点均无直接camera-history patch证据，触发预注册硬失败；
- 96×64全网格平均projective support为`72.2693%`；
- 候选查询节点平均支持率为`89.2787%`；
- 1104个候选实际使用新增前向区域`x>=32m`，其扩展查询节点支持率均为100%。

两个零证据候选均为native：分别只有12和10个离散查询节点，且新增前向区域节点数均为0。
因此当前失败不是96×64前向扩展导致的远距证据缺失。它说明“完整候选footprint中的每个
候选都必须至少有一个直接观测节点”与实际决策对比之间仍可能错配，特别是候选共享的局部
区域不能未经证明地当作区分候选所需的视觉证据。

## 完整性

机械审计从261个源日志和冻结panel artifacts重新计算全部672 ledgers：anchor ledgers、
源日志hash与summary全部精确一致，0失败。

- result SHA-256：`4671032f4d22d37d067c83e0c298e592467e13601788b55dae53f8a7055cc8cc`；
- audit SHA-256：`6e062a3bfc17d36f7824668f29473ac2d0074371a0c7a8e7da61f1f104f71919`；
- runtime：`102.46s`；
- external reviewer：`REVIEW_UNAVAILABLE`。

## 下一步边界

H7BS4B teacher构建、96×64网络训练、H7-B2与H7-C保持HOLD。只允许登记一次post-hoc、
outcome/teacher-free归因：检查两个失败是否来自候选公共区域，以及native/challenger的
对称差查询区域是否具有直接证据。该诊断不得覆盖H7BS4A原始STOP，也不得事后改门。

projective support不是物理可见性；本实验不证明道路任务正确性、NavDream鲁棒性、时延、
决策收益或安全。
