# Refinement Report

日期：2026-10-25  
轮数：1  
最终状态：`REVISE__WAIT_FOR_R005E2`

## 产物

- 最终蓝图：`AORTA_BEST_PAPER_FIRST_PRINCIPLES_BLUEPRINT_V1_CN.md`
- 审查总结：`REVIEW_SUMMARY.md`
- 内部评分：`score-history.md`

## 方法演化

1. 从“appearance-robust scorer”提升为由目标条件分布定义的 Decision Quotient；
2. 将 differential support 从完整效用代理收紧为提出比较问题的 query geometry；
3. 将 global visual evidence 保留为非加性上下文的回答通道；
4. 将 uncertainty/fallback 收紧为可辨识性下的条件性从属机制；
5. 增加 supply regret / selection regret 分解，限定 reranker 的系统责任；
6. 将整篇论文压缩为一个主贡献、一个从属机制和三个决定性 discovery blocks。

## Anchor Check

- 原始问题保持：仍解决外观变化下冻结 planner 的候选决策鲁棒性；
- 未漂移到地图重建、候选生成或通用世界模型；
- 未为追求“前沿”加入 LLM、RL、diffusion 或额外 foundation model；
- NavDream 继续只作最终零样本评估。

## Simplicity Check

- 主贡献：Decision-Quotient Visual Learning via Differential Evidence；
- 新 trainable components：differential query encoder、distributional relation head；
- post-hoc component：group-simultaneous calibration；
- 删除/拒绝：旧 field/SDF 主线、几何证书扩展、Hodge 包装、并列风险头、开放式复杂架构搜索。

## Remaining Weaknesses

理论工具大多是标准结果；真正的新颖性只能由新问题重述、差异架构的 matched advantage 和行为层
发现共同产生。当前这三者均未完全得到经验支持，因此禁止写成已成立贡献。

## Next Step

等待 R005E2 primary、replay 和审计。只有 GO 才进入 generic-vs-differential 的预注册；MIXED/STOP
按蓝图中的失败降级表执行，不自动增加模块。

