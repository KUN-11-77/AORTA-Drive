# R004H7-BS4B0A Experiment Audit

**Date:** 2026-10-25  
**Scope:** forward96 exact-teacher artifact qualification  
**External reviewer:** `REVIEW_UNAVAILABLE`  
**Mechanical verdict:** `PASS`  
**Scientific integrity verdict:** `PROVISIONAL_SCOPE_WARN`

按 `experiment-audit` 流程检查时，当前会话没有可用的跨模型 reviewer backend，因此不得把
本报告表述为独立科学复核。以下是可重复的机械审计与范围审计。

## A. Ground-truth provenance：PASS（限定对象）

teacher 来自冻结 NAVSIM metric-cache polygon cover，不从模型输出生成。builder 只枚举冻结
cache 并调用精确 polygon teacher（`scripts/build_r004h7bs4b0a_forward96_labels.py:46-100`）。
但它是 dataset-derived map geometry，只能作为该 cache 对象的 teacher，不能代表现实道路真值。

## B. Score normalization：PASS

本实验不计算模型分数、性能指标或归一化分数，只构建 signed-distance label 与误差账本。

## C. Result existence and replay：PASS

primary/replay manifest 与 audit JSON 均存在；记录数 164、节点数 1,007,616，审计报告
`records_exact=true`、artifact failure=0、array failure=0。独立本地入口再次读取两个 manifest
并确认 record ledger 完全相同。

## D. Dead-code path：PASS

正式命令实际调用 builder 两次和 audit 一次；audit 重新打开每个 NPZ，并检查数组名、形状、
有限性、误差方向与两次数组相等（`scripts/audit_r004h7bs4b0a_forward96_labels.py:72-111`）。

## E. Scope：WARN

164-cache 范围足以审查生成器确定性，但与 H7BS4A-D0 的 672 个 fit token 不相同。任何把
该结果升级为 D0 teacher-criticality 或视觉机制证据的表述均不成立。项目已把精确 token 对齐
单列为 B0A2，不用跨 token 标签补洞。

## F. Evaluation type

`real_gt`（相对于冻结 dataset cache 的地图几何），不是模型性能评估；物理地图与现实道路误差
不在该类型的保证范围内。

## Claim impact

- 生成器确定性：`SUPPORTED_MECHANICALLY`；
- D0 对应 fit 标签可用：`UNSUPPORTED_UNTIL_B0A2`；
- 视觉、决策、OOD 与安全结论：`NOT_EVALUATED`。
