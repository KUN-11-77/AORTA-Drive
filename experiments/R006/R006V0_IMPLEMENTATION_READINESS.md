# R006V0 实现资格清单

状态：`DESIGN_READY__IMPLEMENTATION_NOT_READY`。

本文件只管理实现来源与机械资格，不改变 R006V0 科学问题、数据、层、rank、控制或裁决阈值。

## 已冻结

- 理论对象：task-weighted channel basis 与逐位置 coefficient field；
- 数据角色：42 source-fit groups；其余角色全部封存；
- 6-fold nested group-held-out 选择程序；
- candidate layer、rank 字典、外观算子与科学 arms；
- 16 个固定 random/noise seeds；
- group-level bootstrap 统计和 GO/STOP 结构；
- canonical 协议：`experiments/R006/R006V0_CAUSAL_VISUAL_MEDIATION_PREREGISTRATION.md`；
- canonical config：`configs/experiment/r006v0_causal_visual_mediation.yaml`。

## 正式 smoke 前必须写入 manifest 的不可缺项

1. 单一 planner checkpoint 的绝对路径与 SHA-256；
2. planner 源码提交 SHA 与 dirty-state；
3. hook 名称、shape、token/BEV 顺序和空间坐标 schema 的 SHA-256；
4. 图像预处理、受控 appearance 实现和 source manifest SHA-256；
5. `coverage_epsilon`：在归一化 W 度量中冻结，不得由 science outcome 调整；
6. synthetic dense-Jacobian oracle 的 objective、gradient 和 projector tolerance；
7. bootstrap seed、CI 类型与有效重采样比例；
8. dose OLS 截距的具体语义；
9. GPU roster、软件环境、最大并发、预计 wall time、异常停止与恢复规则。

## 当前代码资产边界

已有外部 instrumentation 原型提供 `early visual / fused BEV / planner memory / trajectory query` hooks，以及 same-scene / wrong-scene full patch。它只能作为移植参考：

- 不直接在原 GTRS checkout 中修改；
- 必须移植到隔离 worktree；
- 不沿用其 NavDream manifest；
- R006V0S 仅使用 source-controlled appearance；
- full patch 仅是机械 smoke，不进入科学主张。

## 解除阻塞的唯一顺序

1. 移植 hook 与 tail replay，生成 schema；
2. 冻结所有 provenance；
3. 运行显式 Jacobian synthetic oracle；
4. 运行 1–2 anchors 的 identity/full-patch smoke；
5. 用 smoke 的非科学耗时冻结正式预算；
6. 重新生成 config/manifest SHA；
7. 一次性执行 R006V0，不在途中读取部分科学指标。

任何一项未完成，状态保持 `IMPLEMENTATION_NOT_READY`。
