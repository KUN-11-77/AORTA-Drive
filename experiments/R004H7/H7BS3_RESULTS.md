# R004H7-BS3R1：前向96×64完整候选域development资格结果

## 裁决

`DEVELOPMENT_FULL_DOMAIN_PASS`。冻结的前向96×64、1 m节点域在24个historically
exposed development groups上容纳全部4秒native和intent-eligible challenger离散查询。
这只闭合空间定义域，不授权网络训练、H7-B2或H7-C。

## 结果

- 24 groups、384 anchors全部进入分母；
- native支持：`384/384`；
- eligible challengers支持：`2206/2206`；
- 任一eligible candidate越界的anchors：`0`；
- 有至少一个eligible challenger的有效panels：`375/384`；另外9个anchors是候选供给为空，
  不是ROI失败；
- 达到至少12/16有效panels的groups：`24/24`，最差组恰为12；
- 运行时间：`16.033 s`。

该结果与H7BS2 fit结果共同支持：前向96×64是当前冻结字典中首个在fit与development均
完整保留候选查询的空间合同。它有6144节点，相对原64×64增加50%，但静态节点数不能
替代真实端到端P50/P95时延、显存和吞吐测量。

## 完整性

初始H7BS3在读取trajectory payload前因身份manifest字段不匹配fail closed；R1只将身份
来源改为已冻结的source split manifest，并新增真实manifest测试，未改变ROI、候选、门槛
或数据。机械审计重读384个哈希绑定ledger并复算支持与组级裁决，0失败。

- result SHA-256：`0015b3bc74755161eba2ffa6a1e087412d302ef87b4cbbcf92903ffebb3ef350`；
- audit SHA-256：`08b5bf838a5457bdaf29165886bbb4e625a6082192d384ed2db12663575542ea`；
- external reviewer：`REVIEW_UNAVAILABLE`。

## 主张边界与下一门

本实验未读取视觉场、teacher、regret/flip、calibration、test、NavDream或target。它不证明
96×64远场可被相机可靠观测，不证明decoder精度、1 m分辨率、decision relevance、驾驶
收益、安全或ICCV创新性。

下一步只允许预注册两个实现前资格门：

1. 距离分层的投影/teacher/视觉可观测性审计；
2. 同一硬件、同步计时的DINO、decoder、query与arbitration端到端P50/P95、显存和吞吐。

两门通过前不训练96×64网络；H7-B2与H7-C继续HOLD。
