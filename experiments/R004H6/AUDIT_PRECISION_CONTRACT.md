# R004H6A 精度审计合同

R004H6R1 不进行第三次实验运行。本阶段只审计同一批冻结结果与 artifacts。

## 已观察问题

原 runner 使用 source label artifact 中的 `float64` teacher 计算所有指标，同时为便携输出保存 `float32` teacher。旧 auditor 只读取序列化 `float32` teacher。由于 descriptive salient-edge mask 使用硬阈值 `abs(edge)>=0.50m`，一个阈值附近的 edge 在转换后改变集合成员，造成 1/384 metric bundle 的单一 secondary scalar 不一致。

## 冻结修复

1. 重新读取原始、哈希冻结的 `float64` label artifact；
2. 要求每个序列化 teacher 与 `exact_teacher.astype(float32)` bit-exact 相等；
3. 使用 authoritative `float64` teacher 重放 runner 的原始指标语义；
4. 保持 `0.50m` 阈值、`1e-6` 比较容差、统计方案、scientific decision rules 和所有输入不变；
5. 只写新的 audit artifact，不覆盖原 result、fields 或旧 audit failure。

## PASS 条件

- 384/384 artifact hashes 匹配；
- 384/384 teacher float32 projections bit-exact；
- 384/384 metric bundles 在原容差内复算一致；
- macro summary 与 `SUPPORTED/MIXED/UNSUPPORTED` 决策完全复现。

该审计若通过，只能解除机械 fidelity HOLD；不会增加样本、改善指标或扩展到 untouched test、NavDream、目标域、驾驶收益或安全结论。
