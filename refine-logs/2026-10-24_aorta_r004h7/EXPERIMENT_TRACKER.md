# R004H7 Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R004H7A | M0 | 理论与oracle资格 | exact decision quotient | synthetic only | invariant/bound failure count | MUST | PASS | 290k+登记案例0失败；外部reviewer unavailable |
| R004H7B | M1 | 完整4秒决策相关性 | frozen R004H6 field | 24 development groups | regret association / flip AUROC | MUST | INSUFFICIENT_PANEL_COVERAGE | 265/384有效、11/24 groups过门；未计算关联；原结果保持不变 |
| R004H7BS1 | M1a | outcome-blind固定前缀支持资格 | frozen 64×64 panel | 42 fit → 24 development groups | 每组有效anchor数 | MUST | DEVELOPMENT_SUPPORT_PASS | fit选出2.5秒；development 360/384、24/24 groups过门；机械audit PASS；只证明可查询，不证明决策充分 |
| R004H7BS2 | M1b | 候选区分充分性与完整时域ROI诊断 | frozen realized trajectories only | 42 fit groups | late divergence / exact prefix node equivalence / 4s support-cost frontier | MUST | PREFIX_ROUTE_FALSIFIED | 128 exact反例遍及36/42组；96×64首个保留3864/3864 eligible queries；audit PASS |
| R004H7BS3 | M1c | 前向96×64完整候选域资格 | frozen realized trajectories only | 24 development groups one-shot | all eligible queries supported / panel coverage | MUST | PREREGISTRATION | 禁止field、teacher、outcome；通过后仍需视觉与时延门 |
| R004H7B2 | M1d | 完整时域决策相关性 | future qualified full-horizon field | same 24 development groups | regret association / flip AUROC | MUST | HOLD | 2.5秒路线停止；等待H7BS3及视觉/时延资格 |
| R004H7C | M2 | 结构必要性 | 6 matched arms × 3 seeds | 42 fit / 8 selection / 16 proxy | regret / flip / cost | MUST | HOLD | 等待H7B SUPPORTED |
| R004H8A | M3 | 跨provider与驾驶收益 | frozen winner | later frozen panels | clean harm / regret / driving metrics | LATER | HOLD | 不属于本轮 |
