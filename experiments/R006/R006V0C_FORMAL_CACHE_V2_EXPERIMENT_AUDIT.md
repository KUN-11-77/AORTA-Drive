# R006V0C 正式缓存 V2 完整性审计

**日期**：2026-10-26

**外部审稿后端**：`REVIEW_UNAVAILABLE`（当前会话未提供experiment-audit要求的独立reviewer工具）

**总体裁决**：`WARN`；**机械完整性**：`PASS`

## 检查

- Ground truth：N/A；本实验只物化内部activation，不计算标签或方法效果；
- Score normalization：PASS；没有科学分数；
- Result existence：PASS；runner/audit exit code、V2 manifest与audit均存在；
- Provenance：PASS；checkpoint、planner commit、clean `navsim/` execution subtree及tree object完整；
- Replay：PASS；两次完整运行42/42 shard hashes逐组一致；
- Scope：PASS for cache mechanics only；42 groups、672 anchors、四种条件完整；
- Evaluation type：`mechanical_cache_qualification`。

## WARN边界

独立外部reviewer后端不可用，因此不将本报告描述为cross-model审查。该限制不改变机械PASS，也不允许
把缓存资格写成科学证据。

