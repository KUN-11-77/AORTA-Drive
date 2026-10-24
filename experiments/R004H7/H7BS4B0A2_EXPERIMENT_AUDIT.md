# R004H7-BS4B0A2 Experiment Audit

**Date:** 2026-10-25  
**External reviewer:** `REVIEW_UNAVAILABLE`  
**Mechanical verdict:** `PASS`  
**Scientific integrity verdict:** `PROVISIONAL_SCOPE_WARN`

跨模型 reviewer backend 当前不可用，因此该结果只有内部机械审计，不冒充独立科学复核。

- **Ground-truth provenance — PASS（限定对象）：** 标签来自每个精确 token 的冻结 NAVSIM
  cache polygon，不来自模型输出；这仍只是 dataset-cache 几何，不是现实道路真值。
- **Normalization — PASS：** 不计算或归一化任何模型性能分数。
- **Existence/replay — PASS：** manifest、audit 与日志存在；672 个 artifact 全部重开检查，
  4,128,768 节点、672 个前缀逐位匹配、0 failure。
- **Role isolation — PASS：** 672 anchors 全部为 train fit；精确 join 拒绝 duplicate、missing
  与 non-train 记录；没有用 B0A 的异 token 标签替代。
- **Scope — WARN：** 该结果只资格化 fit teacher 工件；它不是 teacher-criticality、泛化、
  网络效果、时延或驾驶收益证据。

结论：`fit teacher artifacts SUPPORTED_MECHANICALLY`；其余方法主张保持 `NOT_EVALUATED`。
