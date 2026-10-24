# R004H7-BS4B1 Experiment Audit

**Date:** 2026-10-25  
**External reviewer:** `REVIEW_UNAVAILABLE`  
**Mechanical verdict:** `PASS`  
**Scientific integrity verdict:** `PASS_WITH_SCOPE_LIMIT__PRIMARY_HYPOTHESIS_UNSUPPORTED`

跨模型 reviewer backend 当前不可用；以下为内部机械重放与主张边界审计，不冒充独立同行复核。

- **Ground-truth provenance — PASS（限定对象）：** oracle 只在 projectively supported nodes 读取
  exact-token NAVSIM teacher intervals；hidden labels 只用于最终 containment 审计。它是冻结任务几何
  teacher，不是模型输出，也不是现实世界安全真值。
- **No score normalization — PASS：** 没有驾驶分数或模型分数；interval、strict sign 与 harmful
  intervention 直接按冻结数值规则计算，没有以预测统计量归一化。
- **No phantom result — PASS：** primary、replay、audit 和日志均存在；672 anchor ledgers、summary
  与 root contract 独立重放 exact，failure count 0。
- **Scope completeness — PASS：** 672 anchors、42 recording groups 与 3,864 pairs 与冻结合同一致。
- **Numerical integrity — PASS：** 最大可行势放宽 `1.04361e-14 m`，低于 `1e-9 m` fail-closed
  上限；teacher inclusion、containment 与 nesting failure 均为 0。
- **Role isolation — PASS：** `forbidden_payload_accessed=false`；未读取 learned field、DINO values、
  planner outcomes、development、calibration、test、NavDream 或 target。
- **Statistical scope — WARN：** bootstrap 单位为 42 个 fit recording groups；这是当前冻结样本的
  oracle 机制裁决，不是未见场景或总体保证。
- **Scientific scope — WARN：** interval 更窄不等于额外 sign identifiability；0 harmful interventions
  是 oracle fit 观察，不是部署安全保证。

结论：实现与机械证据可信，但预注册科学假设不通过。必须保留
`ORACLE_SUPPORT_SET_NEAR_VACUOUS_REDESIGN_INTERFACE`，不得以 width 改善替换 primary endpoint，
不得启动 H7C learned joint-uncertainty 训练。
