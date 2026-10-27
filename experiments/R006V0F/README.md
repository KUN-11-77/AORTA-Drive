# R006V0F experiment record

状态：`MECHANICAL_PASS__RANK_ESTIMATION_UNSTABLE__PARENT_STOP_IMMUTABLE`。

## 问题

R006V0 已有效否定固定最小rank跨折共识，但尚不能区分求解器初始化不稳定、各rank独立非嵌套估计、
fold间有效维数异质和共享方向失败。R006V0F仅在已开封inner角色上做重复初始化失败分解，父STOP不可变。

## 冻结设计

- 6个outer-fold作为block，rank `{4,8,16,32}`为fold内条件；
- 每个fold/rank使用8个预注册random-only初始化；
- 每个seed单独求解，禁止PCA初始化与best-of-restarts掩盖初始化方差；
- primary/replay同GPU顺序运行；正式roster为`[0,1,6,7]`；
- recording groups仍是科学独立单位；seed和fold-pair仅为诊断重复，不用于扩大显著性；
- outer-confirmation及所有非source角色封存。

## 模块准入五问

1. **触发失败：** R006V0的最小rank为`[8,32,4,32,4,32]`，固定rank共识失败。
2. **数学对象：** 估计器条件分布 (\widehat P_{f,r,s})、qualification flip与每seed最小rank；不改变
   planner pullback metric、coverage或父科学对象。
3. **最简单替代为何不足：** 事后固定rank32只能绕过STOP，无法区分真实异质性与估计不稳定。
4. **可否定实验：** 若同fold/rank在seed间不翻转且fold modes稳定不同，则“纯优化不稳”解释被否定；
   若大量seed翻转，则尚无资格提出adaptive-rank方法。
5. **删除预测：** 删除random-only重复后，无法把best-of-three的单次结果分解为优化方差与fold异质性。

## 权威文件

- 预注册：`experiments/R006/R006V0F_FAILURE_DECOMPOSITION_PREREGISTRATION.md`；
- 配置：`configs/experiment/r006v0f_failure_decomposition.yaml`；
- runner：`scripts/run_r006v0f_chain.py`；
- 独立聚合审计：`scripts/audit_r006v0f_chain.py`。

正式运行已完成；结果见`RESULTS.md`、`aggregate_audit.json`与`EXPERIMENT_AUDIT.md`。

真实GPU smoke由`scripts/run_r006v0f_gpu_smoke.py`执行：仅使用一个已登记discovery group、一个
selection group、rank 4、一个冻结随机种子与两个优化步，并在同一GPU上做精确回放。它不输出科学
指标，只用于核验checkpoint、cache、planner tail、CUDA/JVP、random-only初始化和确定性回放。

最终聚合审计还必须逐fold核验冻结物理GPU映射、所有rank/seed共享完全相同的coverage分母，并报告
`total_task_drift_sq<=1e-12`的近零分母比例。该诊断不改变冻结coverage公式或科学路线，只防止把
近零task drift上的数值高coverage误写成有意义的机制覆盖。

冻结运行未包含recording-group bootstrap refit，因此projector overlap只能描述性报告；禁止对相关fold
pair做伪bootstrap，最终审计必须标记`UNAVAILABLE_NO_GROUP_LEVEL_REFITS`且`method_unlock=false`。

2026-10-27烟测与正式运行均基于提交`a0675d60f5093540fff3954ba6802c30b9c6a6a7`；正式目录为
`/data/khsong/aorta_runs/r006v0f_a0675d6_formal`。6/6 folds、same-GPU replay及逐折审计完成，聚合审计
83/83 PASS。4/6 folds出现同fold/rank的seed间资格翻转，按冻结规则裁决
`RANK_ESTIMATION_UNSTABLE`；其余fold 1与3的8个seed均无合格rank。该结果只授权预注册凸estimator
诊断，不授权adaptive reader、DINO reader、CPTR、outer intervention或NavDream。父实验
`STOP_GLOBAL_SHARED_INTERFACE`保持不变。
