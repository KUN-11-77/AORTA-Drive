# Experiment Tracker

| Run ID | Milestone | Purpose | Split | Priority | Status | Notes |
|---|---|---|---|---|---|---|
| R004H7BS4A | M1 | 候选查询直接projective evidence | 42 fit groups | MUST | STRICT_STOP | 2/4536零证据native；机械PASS |
| R004H7BS4A-D0 | M1 | 公共区域与候选差异区域失败归因 | 同一fit artifacts | MUST | COMPLETE_MIXED_FAIL | 8个distinct pair差异区零证据；2 blind native无challenger；A裁决不变 |
| R004H7BS4B0A | M2a | 前向96×64精确teacher生成器双构建资格 | 164 source qualification caches | MUST | COMPLETE_PASS_LIMITED | 每次164/164与1,007,616节点；机械PASS；token不匹配fit，禁止跨token使用 |
| R004H7BS4B0A2 | M2b | 672精确fit token的forward96 teacher工件 | 42 train fit groups | MUST | PREREGISTERED | 精确join；旧64前缀仅作同token bitwise replay；120分钟上限 |
| R004H7BS4B0B | M2c | interval-critical teacher节点可观测性 | 同一42 fit groups | MUST | HOLD | B0A2通过后另行冻结；不以标签调阈值 |
| R004H7BS4C | M3 | 64/96/128同步时延与显存 | source-only | MUST | HOLD | 周期预算需先核验 |
| R004H7BS4D | M4 | 一次性development确认 | 24 exposed development groups | MUST | HOLD | A/B/C冻结后 |
