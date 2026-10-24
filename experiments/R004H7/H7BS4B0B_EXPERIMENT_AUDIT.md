# R004H7-BS4B0B Experiment Audit

**Date:** 2026-10-25

**External reviewer:** `REVIEW_UNAVAILABLE`

**Mechanical verdict:** `PASS`

**Scientific integrity verdict:** `PASS_WITH_SCOPE_LIMIT`

跨模型 reviewer backend 当前不可用；以下为内部机械审计和主张边界检查，不冒充独立同行复核。

- **Ground-truth provenance — PASS（限定对象）：** teacher 数值来自 exact-token 冻结 NAVSIM
  cache polygon 与已资格化的 forward96 标签；不是预测字段，也不是现实世界安全真值。
- **No score normalization — PASS：** 没有驾驶分数、模型分数或 normalization；possible-minimizer
  由向外舍入区间的必要充分条件直接计算，ties 保留，未加入 epsilon。
- **No phantom result — PASS：** primary/replay/audit 产物和日志存在；672 anchor ledgers、summary
  独立重放 exact，failure count 0。
- **Scope completeness — PASS：** 672 anchors、4,536 candidates、3,864 pairs、3,678 distinct
  pairs 和 8 个 D0 zero-support pairs 均与冻结合同完全一致。
- **Role isolation — PASS：** 只读取 train fit teacher、候选几何和 projective support；
  `forbidden_payload_accessed=false`。未读取 learned field、DINO 值、outcome、development、
  calibration、test、NavDream 或 target。
- **Statistical scope — WARN：** 这是 42 个 fit groups 上的确定性机制诊断；13.1393% candidate
  和 39/42 group 是该冻结样本的观察，不是总体发生率置信保证。
- **Scientific scope — WARN：** projective support 不是遮挡感知物理 visibility；teacher-critical
  不等于 decision-critical，更不等于驾驶伤害或安全收益。

结论：`FIT_TEACHER_CRITICAL_PROJECTIVE_INCOMPLETENESS_SUPPORTED`。允许进入 B1 oracle
uncertainty non-vacuity 的预注册；H7C、OOD 与驾驶主张保持 `HOLD`。
