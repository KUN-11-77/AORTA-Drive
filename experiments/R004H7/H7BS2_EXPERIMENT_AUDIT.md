# R004H7-BS2 实验完整性审查

状态：`MECHANICAL_PASS__SCIENTIFIC_REVIEW_UNAVAILABLE__ROI_SELECTION_MIXED`。

- provenance：只读 H7BS1 已哈希绑定的 fit trajectory artifacts；禁用 payload 未访问；
- scope：42 groups、672 anchors 全保留，3864 eligible pairs 全记账；
- recomputation：672 ledgers 与完整 summary 精确重放，0 failure；
- statistics：recording group 为独立单位，pair/anchor 仅为组内重复；
- normalization：未计算视觉分数或使用自归一化；
- external review：规定的跨模型 reviewer backend 不可用，不能以机械审计冒充科学复核。

Prefix falsifier 与 late-divergence 数字得到机械支持。ROI 部分为 `MIXED`：预注册
panel-level cost rule 确实选择前移64×64，但该规则没有要求所有 eligible challengers
保留，因而不足以支撑“完整候选域”。290/3864 eligible challengers 被该方案拒绝；
前向96×64才是冻结字典中首个100%保留方案。后者属于基于fit结果和定理条件形成的下一
阶段候选，必须重新预注册并在development一次性资格化，不能回写成H7BS2原始主终点。
