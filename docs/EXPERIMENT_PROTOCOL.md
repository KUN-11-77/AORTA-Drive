# Experiment Protocol Boundary

> 本文件只定义进入实验前的顺序与禁止事项。具体样本量、run matrix、预算和命令必须由后续 `experiment-plan` 生成。

## Required order

1. **Configuration freeze:** ROI、resolution、(|\Omega|)、(|E|)、(T(Z))、edge scale、geometry checker、数值误差预算和 splits。
2. **Gate A — theorem/geometry audit:** LP oracle、multi-source equivalence、shift/zero-edge/inclusion cases、proof-carrying geometry checks。
3. **Gate B — source nontriviality:** 使用合法 source-only predictor；允许训练小型 probe，但禁止 GT 自拟合冒充视觉 residual。
4. **Gate C — meaningful non-inclusion:** 连续 footprint 差异、严格 GT margin improvement、匹配资格和偏离预算。
5. **Gate D — candidate/planner transfer:** 同一 field event 下测试 held-out candidate policy 与 frozen planner。
6. 只有 Gate A/B 通过后，才能规划正式 decoder training 与完整 NAVSIM evaluation。

## Baseline fairness

- Layer 1：同一合法 absolute-trained predictor，只改变 uncertainty geometry；
- Layer 2：各方法使用合法训练目标，匹配 backbone、capacity、source data 与 selection；
- absolute、global gauge、local graph 与具体 CDF/joint field 均使用 exact 或方向正确的 conservative relative set-min readout；
- 不得让 absolute baseline 使用任意 gauge 的 edge-only predictor；
- 不得用 target labels 选模型、阈值、图或 checker。

## Reporting units

- calibration/coverage 主单位：独立 session；
- frame、corruption copy 和同 route 片段不得冒充独立 session；
- 主失败事件：同一 (T(Z)) 上的 nominal continuous road failure (H)；
- 同时报告 coverage、sharpness、intervention、meaningful non-inclusion、geometry rejection、数值误差与运行成本。

## Prohibited claims before evidence

- 任意 OOD coverage；
- 已验证实时；
- 完整车辆安全；
- 优于 CDF/DriveReferee/scalar referee；
- 顶会原创或“首次”。
