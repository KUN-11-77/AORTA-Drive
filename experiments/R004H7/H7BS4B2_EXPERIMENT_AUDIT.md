# R004H7-BS4B2 Experiment Audit

**Date:** 2026-10-25  
**External reviewer:** `REVIEW_UNAVAILABLE`  
**Mechanical verdict:** `PASS`  
**Scientific integrity verdict:** `PASS_WITH_POST_RESULT_SCOPE_LIMIT`

- **Ground-truth provenance — PASS（限定对象）：** exact-token frozen NAVSIM forward96 teacher；
  不是模型输出或现实安全真值。
- **No score normalization — PASS：** exact-zero、集合交并与最小值直接计算；无预测依赖归一化、
  无 epsilon tie。
- **No phantom result — PASS：** primary/replay/audit/log 存在；672 anchor ledgers、summary、root
  contract exact，failure count 0。
- **Scope completeness — PASS：** 672 anchors、42 groups、3,864 pairs 与 2,219 B1 exact-zero
  contrasts 全部复现。
- **Category partition — PASS：** 三个 exact-zero 类别互斥且穷尽 2,219 pairs；分类优先级在运行前冻结。
- **Role isolation — PASS：** `forbidden_payload_accessed=false`；未读取 learned field、DINO、outcome、
  development、calibration、test、NavDream 或 target。
- **Statistical/scientific scope — WARN：** 同一 fit panel 的 post-B1 诊断只提供 redesign 资格，
  不提供独立确认或 fork-aware 方法收益。

结论：common-min domination 是当前 exact-zero 的主导机械原因；允许预注册 fork-aware oracle
资格实验，H7C learned-network training 继续 HOLD。
