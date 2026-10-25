# R004H7-BS4B4R1 Experiment Audit

**Date:** 2026-10-25

**External reviewer:** `REVIEW_UNAVAILABLE`

**Mechanical verdict:** `PASS`

**Scientific-scope verdict:** `SUPPORTED_ON_SOURCE_FIT_ONE_FRAME_ONLY`

**Frozen decision:** `GO_TO_MATCHED_STRUCTURE_PREREGISTRATION__NO_TRAINING_YET`

- **Ground-truth provenance — PASS_WITH_SCOPE：** labels 来自 source-fit exact-token NAVSIM cache
  上的 candidate-local one-frame PDM scorer。它不是完整 contextual/official EPDMS，也不是现实安全
  真值。
- **Candidate-domain integrity — PASS：** 只评分预 B4 冻结 membership 中的 3,864 个 eligible
  challengers；15 个 zero-challenger anchors 保留在 672-anchor 分母中；ineligible score count 为 0。
- **Score semantics — PASS_WITH_SCOPE：** 每条 candidate 独立相对于固定 PDM baseline proposal set
  评分，`PROGRESS_MODE=eval`；native-relative margin 由两个 candidate-local scores 相减得到。
- **Result existence — PASS：** primary、full replay、progress、logs、exit code、source identity、
  remote audit 与本地复算均存在并有 SHA-256。
- **Replay/recompute — PASS：** 两次完整 ledgers exact；primary/replay summary 均从 ledger 复算
  exact；root、provenance、estimand 与 progress-mode contracts exact。
- **Finite values — PASS：** 两份完整 result JSON 无 NaN/Inf；退出码 0。
- **Scope — PASS_FOR_REGISTERED_QUESTION：** 42 fit groups × 16 anchors 完整执行，以 recording group
  为独立统计单位。结果支持 source-fit one-frame oracle gate，不支持 held-out、RGB、NavDream、
  full-context、planner behavior、latency 或 safety。
- **Role isolation — PASS：** development、calibration、test、NavDream、target 未访问。
- **External independence — UNAVAILABLE：** 当前环境无跨模型 reviewer backend；本审计不得表述为
  独立科学同行评审。

## Claim impact

- C15 label legality 与 positive one-frame oracle ceiling：`SUPPORTED_ON_SOURCE_FIT`；
- C20 one-frame 作为最终 contextual target：仍 `REJECTED_AS_UNIVERSAL`；
- C16/C19 differential structure necessity：`NOT TESTED / NO TRAINING`；
- visual generalization、cross-planner、full EPDMS、NavDream、safety：`NOT TESTED`。

## Required action

冻结并保留本结果。下一步只允许预注册 evaluator-scope/contextual target materialization；禁止用
B4R1 标签直接训练最终网络，禁止自动第三次运行，禁止读取封存 split。
