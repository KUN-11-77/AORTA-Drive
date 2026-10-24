# Experiment Tracker

| Run ID | Milestone | Purpose | Split | Priority | Status | Notes |
|---|---|---|---|---|---|---|
| R004H7BS4A | M1 | 候选查询直接projective evidence | 42 fit groups | MUST | STRICT_STOP | 2/4536零证据native；机械PASS |
| R004H7BS4A-D0 | M1 | 公共区域与候选差异区域失败归因 | 同一fit artifacts | MUST | COMPLETE_MIXED_FAIL | 8个distinct pair差异区零证据；2 blind native无challenger；A裁决不变 |
| R004H7BS4B0A | M2a | 前向96×64精确teacher生成器双构建资格 | 164 source qualification caches | MUST | COMPLETE_PASS_LIMITED | 每次164/164与1,007,616节点；机械PASS；token不匹配fit，禁止跨token使用 |
| R004H7BS4B0A2 | M2b | 672精确fit token的forward96 teacher工件 | 42 train fit groups | MUST | COMPLETE_PASS_LIMITED | 672/672、4,128,768节点、672前缀bitwise exact；机械PASS；只资格工件 |
| R004H7BS4B0B | M2c | interval-critical teacher节点可观测性 | 同一42 fit groups | MUST | COMPLETE_SUPPORTED_LIMITED | 596/4536 candidates、39/42 groups关键盲；1/8 D0盲差异可控min；机械PASS |
| R004H7BS4B1-R0 | M2d-preflight | Oracle联合集合首次启动 | 无结果 | MUST | STOPPED_PRE_RESULT | 发现可行势数值放宽未进入PASS门；无result artifact；不可解释 |
| R004H7BS4B1 | M2d | Oracle联合不确定集合非空洞性 | 同一42 fit groups | MUST | COMPLETE_UNSUPPORTED_REDESIGN | 机械PASS；仅5/42组达覆盖；joint-box sign-identifiability差0、CI [0,0]；转零差异归因 |
| R004H7BS4B2 | M2e | global set-min零差异结构归因 | 同一42 fit groups | MUST | COMPLETE_COMMON_MIN_DOMINANCE | 2033/2219零差异为共同min，42/42组；机械PASS；fork-aware oracle设计GO |
| R004H7BS4B3 | M2f | fork-aware task operator oracle资格 | 同一42 fit groups | MUST | COMPLETE_STOP_GENERAL_RERANKER | temporal仅170/2033恢复，group CI 6.34–10.29%；spatial适用率77.13%；机械PASS；不训练网络 |
| R004H7BS4B4 | M2g | source preference标签合法性与oracle ceiling | 同一42 fit groups；development/test/NavDream封存 | MUST | PREREGISTERED_NOT_RUN | 每候选独立与native成对复现one-frame PDM；先验裁决oracle gain与组级支持；不训练网络 |
| R004H7BS4C | M3 | 64/96/128同步时延与显存 | source-only | MUST | HOLD | 周期预算需先核验 |
| R004H7BS4D | M4 | 一次性development确认 | 24 exposed development groups | MUST | HOLD | A/B/C冻结后 |
