# Experiment Tracker

| Run ID | Milestone | Purpose | System/Variant | Split | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| R007A0 | M0 | 服务器资产与SHA核验 | DINOv3-H+ / Constant Eye implementations | N/A | MUST | PARTIAL_HOLD | DINO/source verified；trained Constant Eye checkpoint未定位 |
| R007V0S | M0 | topology audit +合法swap机械smoke | 2 anchors, real DINO stages | source discovery | MUST | IN_PROGRESS | source/DINO hashes verified；Constant Eye TransFuser checkpoint待定位 |
| R007V0D | M1 | 接口有限干预发现 | topology-legal sequential/factorial audit | 28 discovery groups | MUST | HOLD | 等trained checkpoint与R007V0S real forward |
| R007V0P | M2 | full/compressed/skip probe smoke | correct/mismatch/removed | 1 fold | MUST | HOLD | 等R007V0D |
| R007V0F | M2 | 正式6-fold probe | matched readers, 3 seeds | 28/7 | MUST | HOLD | outer不可读 |
| R007V0C | M2 | 一次性确认 | frozen audit and labels | 7 outer-confirmation | MUST | HOLD | 只允许一次 |
| R007V1 | M3 | 机制导出方法对照 | conditional single method | source group-heldout | MUST | HOLD | 仅V0通过解锁 |
| R007V2 | M4 | zero-shot行为闭环 | final frozen pipeline | NavDream/多planner | MUST | HOLD | 仅V1通过解锁 |
| R007V1-VJP | M3 | task-shaped restriction必要性 | VJP vs Gram/direct residual | source | NICE | CUT/HOLD | 只有额外证据才恢复 |
