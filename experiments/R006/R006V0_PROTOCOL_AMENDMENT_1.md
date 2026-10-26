# R006V0 协议修正 1：矩阵自由候选层资格

日期：2026-10-26  
状态：`FROZEN_BEFORE_FORMAL_SCIENCE`

## 触发原因

R006V0S 证明 C1–C4 都是完整且可替换的因果接口，但正式 task-weighted basis 还要求真实 LTF 的矩阵自由
JVP/VJP 在登记精度内自洽。R006V0J 在任何 outer-fold、layer/rank coverage、restoration 或科学 endpoint
可见前完成该机械资格。

## 唯一修正

- 正式候选层从 `{C1,C2,C3,C4}` 收缩为 `{C3 planner memory, C4 trajectory query}`；
- C1/C2 作为“full-patch mechanically valid、matrix-free operator unqualified”失败记录保留；
- 不允许在正式运行中重新启用 C1/C2。

## 不变部分

42 source-fit groups、6-fold nested isolation、三种受控 appearance、rank `{4,8,16,32}`、task-weighted
目标、solver、A0–A7、16 seeds、group bootstrap、layer/rank稳定性、PASS/STOP 和全部主张边界均不改变。

## 解释边界

该修正由自动微分算子资格而非科学效果选择触发，不表示 C3/C4 更有效，也不表示 C1/C2 理论上无介导。
若 C3/C4 在正式 coverage 或 scene-specific controls 中失败，裁决仍为 STOP，不再回到 C1/C2 搜索。
