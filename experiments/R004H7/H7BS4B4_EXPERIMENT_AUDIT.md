# R004H7-BS4B4 Experiment Audit

**Date:** 2026-10-25  
**External reviewer:** `REVIEW_UNAVAILABLE`  
**Mechanical verdict:** `PASS`  
**Scientific-interface verdict:** `FAIL`  
**Overall decision:** `INVALID_PREFERENCE_LABEL_INTERFACE`

- **Ground-truth provenance — PASS_WITH_SCOPE：** labels 来自 source-fit exact-token NAVSIM cache
  上的 candidate-specific one-frame PDM scorer；不是完整 official PDMS，不是现实安全真值。
- **Score normalization — PASS_WITH_SCOPE：** 每个 challenger 独立与同一 native 二元评分，避免
  7-candidate progress normalization 改写 estimand；但 one-frame PDM 自身含官方 pair-relative
  progress normalization，已在协议中明示。
- **Result existence — PASS：** primary、full replay、audit、logs、failure attribution 与 bundle
  均存在并有 SHA-256；两次 ledger 与 summary exact。
- **Dead/phantom computation — PASS：** 4,704 candidate outcomes 均写入 ledger，审计重算 summary；
  无 phantom metric。
- **Scope — PASS_FOR_EXECUTION：** 42 groups × 16 anchors 完整执行；统计单位是 group。该范围只
  能资格 source-fit oracle，不能支持泛化、NavDream、驾驶收益或顶会主张。
- **Role isolation — PASS：** development、calibration、test、NavDream、target 未访问。
- **Candidate-domain integrity — FAIL：** 168 个冻结 intent-ineligible challengers 仍进入 oracle
  汇总；48 anchors 因 eligibility 非全真被报告为 reconstruction failure。合法 panel membership、
  实现资格门和统计汇总使用了不同候选域。
- **Reconstruction — PASS AS SEPARATE FACT：** 48/48 被标记失败的 anchors 实际 pose 重建逐 bit
  exact、最大误差 0；但该事实不能修复候选域污染。
- **External independence — UNAVAILABLE：** 当前会话没有可用的跨模型 reviewer backend；不得把
  内部机械审计表述为独立科学复核。

## Claim impact

- C15 source preference label legality：`UNSUPPORTED_FOR_CURRENT_RUN`；
- C15 positive oracle ceiling：`INVALID / NOT INTERPRETABLE`；
- C16 preference quotient architecture/loss necessity：`HOLD / NO TRAINING`；
- visual generalization、cross-planner、full PDMS、NavDream、safety：`NOT TESTED`。

## Required action

保留本轮失败记录。若继续，只能新预注册候选域一致的 R1；禁止删除失败门后复用原 summary，
禁止自动第三次正式运行。
