# R004H7-B 实验完整性审查状态

## 状态：`REVIEW_UNAVAILABLE`

已按 experiment-audit 流程收集冻结 runner、auditor、配置、raw result、机械审计、实验
登记和 claim-evidence 文档路径。但本会话没有该流程要求的独立
`mcp__codex__codex` / manual-review reviewer backend，因而不能给出跨模型
`PASS/WARN/FAIL`，也不能让执行者自行宣告独立审查通过。

可确认的机械事实仅限于：raw result 与本地同步副本哈希一致；384 个 panel artifacts
存在；冻结 auditor 独立复算 384 anchors，artifact 与 metric failure 均为 0，并复现
`INSUFFICIENT_PANEL_COVERAGE`。这些事实记录于 `H7B_MECHANICAL_AUDIT.json`，不替代
独立科学复核。

审查范围文件：

- `scripts/run_r004h7b_shadow_decision.py`
- `scripts/audit_r004h7b_shadow_decision.py`
- `src/aorta_drive/evaluation/shadow_decision.py`
- `configs/experiment/r004h7_decision_quotient_preregistration.yaml`
- `experiments/R004H7/artifacts/r004h7b_run2/results/result.json`
- `experiments/R004H7/artifacts/r004h7b_run2/audit/mechanical_audit.json`
- `experiments/R004H7/PREREGISTRATION.md`
- `experiments/registry.csv`
- `docs/CLAIM_EVIDENCE_MATRIX.md`

因此所有 H7-B 科学解释保持 provisional，且当前 coverage fail 已足以阻止 H7-C。
