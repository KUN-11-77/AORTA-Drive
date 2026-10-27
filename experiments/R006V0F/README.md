# R006V0F experiment record

状态：`IMPLEMENTED__GPU_SMOKE_PENDING__FORMAL_NOT_STARTED`。

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

正式运行前必须完成真实GPU smoke、固定实现commit、归档hash与服务器路径；结果完成后在本目录增加
`RESULTS.md`与审计报告，并更新claim–evidence matrix。

真实GPU smoke由`scripts/run_r006v0f_gpu_smoke.py`执行：仅使用一个已登记discovery group、一个
selection group、rank 4、一个冻结随机种子与两个优化步，并在同一GPU上做精确回放。它不输出科学
指标，只用于核验checkpoint、cache、planner tail、CUDA/JVP、random-only初始化和确定性回放。
