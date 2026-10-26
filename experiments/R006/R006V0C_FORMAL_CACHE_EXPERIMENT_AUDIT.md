# R006V0C 正式缓存完整性审计

**日期**：2026-10-26  
**外部审稿后端**：`REVIEW_UNAVAILABLE`（当前会话未提供experiment-audit要求的独立reviewer工具）  
**总体裁决**：`WARN`

## 机械检查

- A. ground-truth provenance：N/A。本实验不计算标签、性能或科学endpoint；
- B. score normalization：PASS。未产生分数或归一化科学指标；
- C. result existence：PASS。manifest、42个远端shards、runner/audit exit code与audit文件存在；
- D. dead code：PASS。独立审计器实际执行并生成261项检查；
- E. scope：PASS for cache mechanics。42 groups、672 anchors与四种冻结外观完整，不外推为方法结果；
- F. evaluation type：`mechanical_cache_qualification`。

## WARN原因

planner commit已记录，但第一次完整运行的manifest把dirty state记为`null`。post-run检查只能确认当时之后
的`navsim/`执行子树clean，不能完全替代run-time provenance。另因独立reviewer工具不可用，本报告不
冒充cross-model审查。

## 处置

修复runner，使其通过显式git-dir/work-tree读取并记录整个仓库dirty、`navsim/`子树dirty与子树Git
tree object；V2审计将execution subtree clean作为硬门。修复后重新生成完整缓存，正式科学运行继续HOLD。

