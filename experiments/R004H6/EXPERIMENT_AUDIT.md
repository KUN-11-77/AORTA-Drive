# R004H6R1 Experiment Audit

**Date**: 2026-10-24  
**Auditor**: `REVIEW_UNAVAILABLE`  
**Overall Verdict**: `WARN`  
**Integrity Status**: `warn`

规定的独立 cross-model reviewer backend 在当前会话中不可用，因此没有把内部机械复算冒充独立实验完整性审查。本文件只记录可复现的机械事实与审计缺口。

## Checks

### A. Ground Truth Provenance: WARN

已登记的 teacher/data provenance、role manifest 和 input replay 均存在，但未获独立 reviewer 逐行复核。

### B. Score Normalization: WARN

冻结脚本与配置已保存，未获独立 reviewer 对全部 metric 路径逐行复核。

### C. Result File Existence: WARN

`result.json`、384 个 field artifacts 与 provenance hashes 均存在；冻结机械 auditor 因一个 secondary scalar replay mismatch 未 PASS。

### D. Dead Code Detection: WARN

未获独立 reviewer 复核。

### E. Scope Assessment: PASS WITH QUALIFIER

范围明确限定为 24 个 historically exposed、current-model-held-out development groups，不写成 untouched test、NavDream/target、population、driving 或 safety 结论。

### F. Evaluation Type

`real_gt` / held-out development mechanism evaluation；不是最终 benchmark test，也不是驾驶收益评估。

## Claim Impact

- held-out visual task mechanism：`supported_provisional`；
- correct-image scene conditioning：`supported_provisional`；
- untouched/OOD generalization：`unsupported_by_scope`；
- driving/safety benefit：`unsupported_by_scope`；
- integrity PASS：`not established`。

## Action Items

- 保留 frozen mechanical audit 的 `metrics=1` 失败记录；
- 单独登记 dtype/threshold audit-contract 处置，不修改本次科学合同，不重跑第三次实验；
- 独立 reviewer 可用后再执行一次只读 cross-model audit；
- 在此之前保持 `HOLD`。
