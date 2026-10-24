# R004H7-BS4B0A：前向 96×64 teacher 生成器资格结果

状态：`PASS_IMPLEMENTATION_QUALIFICATION_ONLY`。

## 冻结结果

- 两次独立完整构建均完成 164/164 个 source cache；
- 每次生成 1,007,616 个节点标签（164×6144）；
- 两次 record ledger 完全一致；
- 164 对 artifact 的字节哈希与数组逐项一致；
- 非有限值、误差方向与 artifact hash 失败均为 0；
- primary/replay 用时分别为 371.912 s 与 370.172 s；
- 最大总数值误差账本为 \(4.6913963943\times10^{-10}\) m；
- 机械审计：`PASS`；外部独立 reviewer：`REVIEW_UNAVAILABLE`。

原始服务器工件保留在
`/data/khsong/aorta_runs/r004h7bs4b0a_357de95/`；本地保存两个 manifest、审计 JSON 与
运行日志，完整 NPZ 保持在服务器，避免把运行工件提交进源码仓库。

## 重要的角色边界修正

这 164 个 cache 是 R002G 的 teacher 实现资格样本，不是 H7BS4A-D0 的 672 个 fit anchors。
因此本结果只证明同一精确 teacher 实现可以在前向 96×64 网格上确定性运行，**不能**把这些
标签与 D0 候选跨 token 连接，也不能直接进行 teacher-critical observability 检验。

下一步必须使用 H7BS4A 的精确 `(log_name, scene_token)` 与 R004E 的 1440-cache manifest
一一连接，只选择 42 个 train fit groups 的 672 个对应 cache，重新生成其 96×64 工件。旧
64×64 标签只能用于相同 token 的前 64 列逐位回放检查，不允许插值、近邻或跨场景复用。

## 允许与禁止的结论

允许：`forward96 exact-teacher generator qualified on the frozen 164-cache source set`。

禁止：视觉可观测、teacher-criticality、决策充分性、模型精度、OOD 泛化、实时性、驾驶收益、
安全性或架构创新。H7BS4A 的 `STOP_OR_REDESIGN_VISUAL_SUPPORT_INTERFACE` 保持不变。
