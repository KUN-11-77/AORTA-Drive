# Experiment Tracker

| Run ID | Milestone | Purpose | System/Variant | Split | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| R007A0 | M0 | 服务器资产与SHA核验 | DINOv3-H+ / Constant Eye implementations | N/A | MUST | PARTIAL_HOLD | DINO/source verified；trained Constant Eye checkpoint未定位 |
| R007V0S | M0 | topology audit +合法swap机械smoke | 2 anchors, real DINO stages | source discovery | MUST | IN_PROGRESS | source/DINO hashes verified；Constant Eye TransFuser checkpoint待定位 |
| R007V0D | M1 | 接口有限干预发现 | topology-legal sequential/factorial audit | 28 discovery groups | MUST | HOLD | 等trained checkpoint与R007V0S real forward |
| R007V0P | M2 | DINO evidence distribution probe smoke | final/last-four/intermediate/full/compressed/skip；correct/mismatch/removed | 1 fold | MUST | HOLD | 双目标与层候选规则已预注册；等R007V0D |
| R007V0F | M2 | 正式6-fold probe | matched readers, 3 seeds | 28/7 | MUST | HOLD | outer不可读 |
| R007V0W | M2 | 冻结tail写入目标审计 | native/final-feature/output/task四目标，同reader-writer-cut | 28/7 | MUST-IF-INCREMENT | HOLD | 仅V0F视觉任务增量通过后解锁 |
| R007V0T | M2 | 任务上下文时序2×2 | compressed/raw-full × context-free/conditioned，等预算 | 28/7 | MUST-IF-INCREMENT | HOLD | 主量为risk difference-in-differences |
| R007V0R | M2 | 应不变/应响应成对实验 | appearance/task/scene/background valid-label pairs | 28/7 | MUST-IF-VALID | HOLD | 无可靠反事实标签则UNRESOLVED |
| R007V0C | M2 | 一次性确认 | frozen audit and labels | 7 outer-confirmation | MUST | HOLD | 只允许一次 |
| R007V1 | M3 | 机制导出方法对照 | conditional single method | source group-heldout | MUST | HOLD | 仅V0通过解锁 |
| R007V2 | M4 | zero-shot行为闭环 | final frozen pipeline | NavDream/多planner，至少一个真实VLM/VLA | MUST | HOLD | 仅V1通过解锁；DINO本身不算VLM |
| R007V1-VJP | M3 | task-shaped restriction必要性 | VJP vs Gram/direct residual | source | NICE | CUT/HOLD | 只有额外证据才恢复 |
