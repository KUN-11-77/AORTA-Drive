# Refinement Report

## 保留

- DINOv3-H+ 作为统一冻结视觉底座；
- appearance orbit、功能保持、可恢复性和 clean/no-harm 分离；
- 完全冻结 planner 的研究目标；
- 组级统计、角色封存、matched baseline 与 kill rules。

## 删除

- 共享固定 rank 作为主方法；
- Fantope 作为主线前置；
- 从 R006 失败直接推出 VJP；
- 强迫唯一失效归因；
- 把 clean residual penalty 写成 no-harm 保证。

## 新增

- 信息保持与信息损失外观的边界；
- native path × DINO path 因子化交换；
- full/compressed/skip token 组外信息审计；
- amplification/compression/mixed/unresolved 的结果驱动方法树。

## 当前裁决

GO 到接口审计预注册与实现；FQB/VJP 训练 HOLD；NavDream/test/target 继续封存。
