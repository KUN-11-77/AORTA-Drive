# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R005E3A-M0 | M0 | loss与mask资格 | soft-regret synthetic | synthetic | loss/grad/tie/mask | MUST | TODO | CPU tests first |
| R005E3A-M1 | M1 | 单job资源与重放 | correct soft-regret | one source outer fold | exact/finite/runtime | MUST | HOLD | after implementation tests |
| R005E3A-M2 | M2 | 正式目标比较 | parent MSE vs soft-regret | 42-group OOF | regret/recovery | MUST | HOLD | no partial metrics |
| R005E3A-M3 | M3 | 场景绑定输入干预 | correct/S1/S2/S3/removed | same OOF panels | intervention regret | MUST | HOLD | same checkpoint |
| R005E3A-M4 | M4 | 完整审计 | primary/replay | all | provenance/exact/CI | MUST | HOLD | one audit only |
