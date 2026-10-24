# R004H6R1 Experiment Audit

**Date**: 2026-10-24  
**Auditor**: `REVIEW_UNAVAILABLE`  
**Overall Verdict**: `REVIEW_UNAVAILABLE`
**Integrity Status**: `mechanical_pass_external_unavailable`

规定的独立 cross-model reviewer backend 在当前会话中不可用，因此没有把内部机械复算冒充独立实验完整性审查。本文件只记录可复现的机械事实与审计缺口。

## Checks

### A. Ground Truth Provenance: WARN

已登记的 teacher/data provenance、role manifest 和 input replay 均存在，但未获独立 reviewer 逐行复核。

### B. Score Normalization: WARN

冻结脚本与配置已保存，未获独立 reviewer 对全部 metric 路径逐行复核。

### C. Result File Existence: WARN

`result.json`、384 个 field artifacts 与 provenance hashes 均存在。legacy float32 replay 曾有一个 secondary scalar mismatch；独立登记的 exact-teacher audit 已在不改变 tolerance、threshold 或结果的条件下通过 384/384 hashes、projections 与 metric bundles。

### D. Dead Code Detection: WARN

未获独立 reviewer 复核。

### E. Scope Assessment: PASS WITH QUALIFIER

范围明确限定为 24 个 historically exposed、current-model-held-out development groups，不写成 untouched test、NavDream/target、population、driving 或 safety 结论。

### F. Evaluation Type

`real_gt` / held-out development mechanism evaluation；不是最终 benchmark test，也不是驾驶收益评估。

## Claim Impact

- held-out visual task mechanism：`supported_on_heldout_development`；
- correct-image scene conditioning：`supported_on_heldout_development`；
- untouched/OOD generalization：`unsupported_by_scope`；
- driving/safety benefit：`unsupported_by_scope`；
- internal mechanical fidelity：`pass`；
- independent cross-model integrity review：`not established`。

## Action Items

- 保留 legacy `metrics=1` 失败记录及 R004H6A exact-teacher PASS；
- 独立 reviewer 可用后再执行一次只读 cross-model audit；
- R004H7 只允许预注册；calibration/test/target 与驾驶收益保持 `HOLD`。
